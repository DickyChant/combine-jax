"""Independently re-evaluate a saved Combine fit point before a separate restart."""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def interrupted_best_point(parent, previous):
    """Require the preserved raw point and a hash-bound interruption audit."""
    assert previous["status"] == "interrupted"
    interruption = previous["interruption"]
    raw_path = Path(interruption["raw_report"])
    raw = raw_path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    assert checksum == interruption["raw_report_sha256"]
    original = json.loads(raw)
    assert original["status"] == previous["producer_last_status"] == "running"
    audit = json.loads(Path(interruption["audit"]).read_bytes())
    assert audit["status"] == "interrupted_without_final_fit"
    assert audit["annotated_reports_sha256"][str(parent)] == hashlib.sha256(parent.read_bytes()).hexdigest()
    assert audit["raw_reports"][str(parent)] == dict(path=str(raw_path), sha256=checksum)
    assert audit["sources_unchanged"]
    assert not audit["final_minimum_present"] and not audit["final_covariance_present"]
    assert audit["tool_session"] == interruption["tool_session"]
    assert audit["tool_command_exit_code"] == interruption["tool_command_exit_code"]
    assert "final" not in previous and "arrays" not in previous
    for key, value in original.items():
        if key != "status":
            assert previous[key] == value, key
    point = np.asarray(original["best_seen_values"], dtype=np.float64)
    expected = float(original["best_seen_zero_point_nll"])
    assert np.isfinite(point).all() and np.isfinite(expected)
    return point, expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "hhvv"), default="hhvv")
    parser.add_argument("--parent-tag", choices=("", "restart", "restart_strategy1",
                                               "restart_centered_iterated", "restart_best_native",
                                               "restart_best_hesse1"), default="",
                        help="Check a separately recorded restart instead of the original fit")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    started = time.perf_counter()
    suffix = "_"+arguments.parent_tag if arguments.parent_tag else ""
    parent = Path("reports")/(arguments.model+"_combine_fit"+suffix+".json")
    parent_bytes = parent.read_bytes()
    parent_sha256 = hashlib.sha256(parent_bytes).hexdigest()
    previous = json.loads(parent_bytes)
    output = Path("reports")/(arguments.model+"_combine_fit"+
        (suffix+"_candidate_check" if suffix else "_restart_check")+".json")
    if output.exists():
        raise FileExistsError("Preserve previous restart evidence")
    assert previous["model"] == arguments.model
    assert previous["status"] in ("reference_ready", "interrupted") or (
        previous["status"] == "failed" and "TimeoutError" in previous["exception"])
    likelihood = load_export(Path("build/gpu_exports")/(arguments.model+".pkl"), previous["source"])
    np.testing.assert_array_equal(likelihood.parameter_names, previous["parameter_names"])
    np.testing.assert_array_equal(likelihood.initial, previous["initial"])
    if previous["status"] == "reference_ready":
        artifact = Path(previous["arrays"])
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == previous["arrays_sha256"]
        with np.load(artifact, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["parameter_names"], likelihood.parameter_names)
            point = arrays["values"].copy()
        np.testing.assert_array_equal(point, previous["final"]["values"])
        expected_zero = previous["final"]["nll_zero_point"]
    elif previous["status"] == "interrupted":
        point, expected_zero = interrupted_best_point(parent, previous)
    else:
        point = np.asarray(previous["best_seen_values"])
        expected_zero = previous["best_seen_zero_point_nll"]
    assert point.shape == likelihood.initial.shape and np.isfinite(point).all()
    assert np.all(point >= [p.minimum for p in likelihood.parameters])
    assert np.all(point <= [p.maximum for p in likelihood.parameters])
    for index, parameter in enumerate(likelihood.parameters):
        if parameter.minimum == parameter.maximum:
            assert point[index] == likelihood.initial[index]
    workspace_path = Path("inputs/workspace_MX2000.root" if arguments.model == "hhvv" else "build/gpu_exports/counting.root")
    with workspace_path.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == previous["source"]["sha256"]
    root = load_caching_nll()
    report = dict(status="running", model=arguments.model,
        scope="Independent restart-point check only; no minimum, covariance, impact or fit timing acceptance",
        parent=str(parent), parent_sha256=parent_sha256,
        parent_status=previous["status"],
        point_source="final" if previous["status"] == "reference_ready" else "best_seen",
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), source=previous["source"],
        combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
        runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
        parameter_names=list(likelihood.parameter_names), point=point.tolist(),
        discrete_choices=json.loads(json.dumps(likelihood.discrete_choices)),
        channel_masks=json.loads(json.dumps(likelihood.mask_configuration)),
        parameter_bounds_checked=True, fixed_parameters_unchanged=True, probes=[])
    assert report["combine_version"] == previous["combine_version"]
    assert report["root_version"] == previous["root_version"]
    assert report["runtime_flags"] == previous["runtime_flags"]
    assert report["discrete_choices"] == previous["discrete_choices"]
    assert report["channel_masks"] == previous["channel_masks"]

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    old_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    try:
        with open_workspace(workspace_path) as workspace:
            if arguments.model == "hhvv":
                assert configure_hhvv(workspace) == previous["source"]["configuration"]
            variables = [workspace.var(name) for name in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
            if model.GetPdf().ClassName() == "RooSimultaneousOpt":
                nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()

            def evaluate(theta, label):
                for variable, coordinate in zip(variables, theta, strict=True):
                    variable.setVal(float(coordinate))
                root.RooAbsReal.clearEvalErrorLog()
                value = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                report["probes"].append(dict(label=label, nll=value, roofit_evaluation_errors=errors))
                assert np.isfinite(value) and errors == 0
                return value

            nominal = evaluate(likelihood.initial, "canonical_initial")
            assert abs(nominal-previous["canonical_initial_nll"]) < 1e-6
            nll.setZeroPoint()
            assert abs(evaluate(likelihood.initial, "zero_point_initial")) < 1e-6
            zero = evaluate(point, "saved_point_zero_offset")
            report["saved_zero_point_absolute_error"] = abs(zero-expected_zero)
            assert report["saved_zero_point_absolute_error"] < 1e-6
            nll.clearZeroPoint()
            canonical = evaluate(point, "saved_point_canonical")
            report["canonical_offset_absolute_error"] = abs(canonical-(nominal+zero))
            assert report["canonical_offset_absolute_error"] < 1e-6
            restored = evaluate(likelihood.initial, "restored_initial")
            report["canonical_restoration_absolute_error"] = abs(restored-nominal)
            assert report["canonical_restoration_absolute_error"] < 1e-6
            report.update(canonical_candidate_nll=canonical, zero_point_candidate_nll=zero)
            del nll
        assert hashlib.sha256(parent.read_bytes()).hexdigest() == parent_sha256, "Parent changed during check"
        report["status"] = "candidate_checked"
        save()
        print(report["status"], report["saved_zero_point_absolute_error"], report["canonical_offset_absolute_error"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
