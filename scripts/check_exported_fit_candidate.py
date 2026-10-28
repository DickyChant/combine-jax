"""Evaluate a saved bounded-JAX fit candidate with Combine v11 at the same point.

Use an immutable report snapshot for a running fit. This checks a point only;
it does not accept a minimum, covariance, discrete envelope or fit timing.
"""

import argparse
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core_digest():
    return hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()


def checked_point(parent, likelihood, selection):
    """Keep point validation separate from any optimizer convergence claim."""
    assert parent["memory_policy"] == "resident_value_spooled_gradient"
    assert parent["float64"] and not parent["root_imported"]
    np.testing.assert_array_equal(parent["parameter_names"], likelihood.parameter_names)
    assert parent["discrete_choices"] == json.loads(json.dumps(likelihood.discrete_choices))
    assert parent["channel_masks"] == json.loads(json.dumps(likelihood.mask_configuration))
    if selection == "final":
        assert parent["status"] in ("minimum_computed", "fit_incomplete")
        assert parent["stage"] == "finished" and parent["evaluation_counters"]["closed"]
        point, expected = parent["final"]["values"], parent["final"]["nll"]
    elif selection == "best_seen":
        point, expected = parent["best_seen_values"], parent["best_seen_nll"]
    else:
        raise ValueError(f"Unknown point selection: {selection}")
    point = np.asarray(point, dtype=np.float64)
    assert point.shape == likelihood.initial.shape and np.isfinite(point).all()
    assert np.isfinite(expected)
    for value, parameter, nominal in zip(point, likelihood.parameters, likelihood.initial, strict=True):
        assert parameter.minimum <= value <= parameter.maximum
        if parameter.minimum == parameter.maximum:
            assert value == nominal
    return point, float(expected)


def candidate_errors(nominal, zero, canonical, restored, expected):
    assert np.isfinite([nominal, zero, canonical, restored, expected]).all()
    errors = dict(
        canonical_jax_absolute_error=abs(canonical-expected),
        saved_zero_point_absolute_error=abs(zero-(expected-nominal)),
        canonical_offset_absolute_error=abs(canonical-(nominal+zero)),
        canonical_restoration_absolute_error=abs(restored-nominal))
    return errors, all(value < 1e-6 for value in errors.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--point", choices=("final", "best_seen"), default="final")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use a new reports/*.json output")
    if args.output.exists():
        raise FileExistsError("Preserve earlier point checks")
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    started = time.perf_counter()
    raw = args.parent.read_bytes()
    parent_hash = hashlib.sha256(raw).hexdigest()
    parent = json.loads(raw)
    assert parent["runner_sha256"] == digest("scripts/fit_bounded_cpu.py")
    assert parent["source_sha256"] == core_digest()
    assert all(digest(p) == h for p, h in parent["frozen_source_files"].items())
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    assert digest(manifest_path) == parent["manifest_sha256"]
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][parent["model"]]
    assert model["source"] == parent["source"]
    export = Path("build/gpu_exports")/model["cache"]
    assert digest(export) == parent["export_sha256"]
    workspace_path = (Path("inputs/workspace_MX2000.root") if parent["model"] == "hhvv"
                      else Path("build/gpu_exports")/(parent["model"]+".root"))
    assert digest(workspace_path) == parent["source"]["sha256"]
    from hhvv_export_cache import load_export
    from hhvv_setup import configure_hhvv
    from combine_jax.discrete import extract_multipdf
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS

    likelihood = load_export(export, parent["source"])
    point, expected = checked_point(parent, likelihood, args.point)
    if args.point == "final":
        assert digest(parent["arrays"]) == parent["arrays_sha256"]
        with np.load(parent["arrays"], allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["parameter_names"], likelihood.parameter_names)
            np.testing.assert_array_equal(saved["values"], point)
    root = load_caching_nll()
    stack = dict(combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()))
    assert stack == parent["export_stack"]
    assert stack == {k: manifest[k] for k in stack}
    assert stack == dict(combine_version="11.0.0", root_version="6.40.00")
    frozen = {p: digest(p) for p in ("scripts/check_exported_fit_candidate.py",
        "scripts/hhvv_export_cache.py", "scripts/hhvv_setup.py")}
    report = dict(status="running", model=parent["model"], parent=str(args.parent),
        parent_sha256=parent_hash, parent_status=parent["status"], point_source=args.point,
        runner_sha256=digest(__file__), frozen_sources=frozen,
        source=parent["source"], source_sha256=parent["source_sha256"],
        export_sha256=parent["export_sha256"], manifest_sha256=parent["manifest_sha256"],
        **stack, runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
        parameter_names=list(likelihood.parameter_names), point=point.tolist(),
        discrete_choices=parent["discrete_choices"], channel_masks=parent["channel_masks"],
        parameter_bounds_checked=True, fixed_parameters_unchanged=True,
        saved_jax_nll=expected, probes=[],
        scope="Same-point Combine v11 NLL check only. The parent may be a running-fit "
        "snapshot; no minimum, covariance, profile, discrete-envelope or fit-timing acceptance.")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    old_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    save()
    try:
        with open_workspace(workspace_path) as workspace:
            configuration = configure_hhvv(workspace) if parent["model"] == "hhvv" else {}
            assert configuration == parent["source"]["configuration"]
            for name, choice in likelihood.discrete_choices.items():
                category = workspace.cat(choice["category"])
                assert category
                category.setIndex(choice["index"])
                _, actual = extract_multipdf(workspace.pdf(name))
                assert json.loads(json.dumps(actual)) == parent["discrete_choices"][name]
            for mask in likelihood.mask_configuration.values():
                actual = workspace.arg(mask["name"])
                assert actual and actual.ClassName() == mask["class_name"]
                assert float(actual.getVal()) == mask["value"]
            variables = [workspace.var(name) for name in likelihood.parameter_names]
            assert all(variables)
            model_config = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model_config.GetPdf(), workspace.data("data_obs"),
                model_config.GetNuisanceParameters())
            if model_config.GetPdf().ClassName() == "RooSimultaneousOpt":
                nll.setChannelMasks(model_config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint()
            nll.clearZeroPoint()

            def evaluate(theta, label):
                for variable, coordinate in zip(variables, theta, strict=True):
                    variable.setVal(float(coordinate))
                root.RooAbsReal.clearEvalErrorLog()
                value = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                report["probes"].append(dict(label=label,
                    nll=value if np.isfinite(value) else None, roofit_evaluation_errors=errors))
                assert np.isfinite(value) and errors == 0
                return value

            nominal = evaluate(likelihood.initial, "canonical_initial")
            report["nominal_reference_absolute_error"] = abs(nominal-model["reference"]["cases"][0]["combine_nll"])
            assert report["nominal_reference_absolute_error"] < 1e-6
            nll.setZeroPoint()
            report["initial_zero_point_absolute_error"] = abs(evaluate(likelihood.initial, "zero_point_initial"))
            assert report["initial_zero_point_absolute_error"] < 1e-6
            zero = evaluate(point, "candidate_zero_offset")
            nll.clearZeroPoint()
            canonical = evaluate(point, "candidate_canonical")
            restored = evaluate(likelihood.initial, "restored_initial")
            errors, passed = candidate_errors(nominal, zero, canonical, restored, expected)
            report.update(errors, canonical_candidate_nll=canonical, zero_point_candidate_nll=zero)
            assert passed, "Same-point NLL or offset/restoration check failed"
            del nll
        assert digest(args.parent) == parent_hash, "Parent changed; use an immutable snapshot"
        assert digest(export) == parent["export_sha256"] and digest(manifest_path) == parent["manifest_sha256"]
        assert digest(workspace_path) == parent["source"]["sha256"]
        assert core_digest() == parent["source_sha256"]
        assert all(digest(p) == h for p, h in {**parent["frozen_source_files"], **frozen}.items())
        report["status"] = "candidate_checked"
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)
        save()
    print(args.output, report["status"], report["canonical_jax_absolute_error"], flush=True)


if __name__ == "__main__":
    main()
