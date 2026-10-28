"""Reference-only stable density probes for selected unresolved HHVV gradients."""

from collections import Counter
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

from combine_jax.discrete import active_dependencies
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from stable_root_bins import LD, constraint_delta
from stable_root_intensities import RootIntensities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement-plan", type=Path,
                        help="Add a fixed FD-only subdivision plan to the preserved original scan")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    selection_path = Path("reports/gpu_random_intensity_selection.json")
    raw = selection_path.read_bytes()
    selection = json.loads(raw)
    reference_raw = Path(selection["reference"]).read_bytes()
    assert hashlib.sha256(reference_raw).hexdigest() == selection["reference_sha256"]
    reference = json.loads(reference_raw)
    assert reference["source"] == selection["source"]
    previous = plan = None
    if arguments.refinement_plan:
        plan_raw = arguments.refinement_plan.read_bytes()
        plan = json.loads(plan_raw)
        previous_raw = Path(plan["reference"]).read_bytes()
        assert hashlib.sha256(previous_raw).hexdigest() == plan["reference_sha256"]
        previous = json.loads(previous_raw)
        assert previous["status"] == "reference_uncertain"
        assert previous["source"] == plan["source"] == selection["source"]
        assert previous["selection_sha256"] == hashlib.sha256(raw).hexdigest()
        assert {(c["point_number"],c["index"]) for c in plan["cases"]} == {
            (c["point_number"],c["index"]) for c in previous["points"] if c["status"] != "reference_ready"}
    likelihood = load_export("build/gpu_exports/hhvv.pkl", reference["source"])
    dependencies = {c.name:active_dependencies(likelihood.nodes,[c.pdf]) for c in likelihood.channels}
    assert np.finfo(LD).nmant >= 63
    output = Path("reports/gpu_random_intensity_fd_dense.json" if plan else "reports/gpu_random_intensity_fd.json")
    if output.exists():
        raise FileExistsError(str(output))
    report = dict(status="running", source=reference["source"], selection=str(selection_path),
                  selection_sha256=hashlib.sha256(raw).hexdigest(), reference=selection["reference"],
                  reference_sha256=selection["reference_sha256"],
                  script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__), Path("scripts/stable_root_intensities.py"),
                                  Path("scripts/stable_root_bins.py"), Path("scripts/probe_hhvv_stable_fd.py"))},
                  combine_version=version("higgs-combine-tool"), root_version=reference["root_version"],
                  runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  longdouble_mantissa_bits=np.finfo(LD).nmant,
                  fractions=selection["fractions"], policy=selection["policy"],
                  scope=selection["scope"]+" Reference diagnostics only; no AD acceptance.", points=[])
    if plan:
        report.update(refinement_plan=str(arguments.refinement_plan),
                      refinement_plan_sha256=hashlib.sha256(plan_raw).hexdigest(),
                      extends_reference=plan["reference"], extends_reference_sha256=plan["reference_sha256"],
                      refinement_rule=plan["rule"],
                      timing_scope="Only new probes and setup; converged parent cases and original probes are reused unchanged")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        fit = json.loads(Path("reports/hhvv_combine_fit.json").read_text())
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if fit["status"] == "running":
            peak += fit["peak_rss_mib"]
        if peak > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed 11,000 MiB")

    root = load_caching_nll()
    assert str(root.gROOT.GetVersion()) == reference["root_version"]
    assert report["combine_version"] == reference["combine_version"]
    old_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == reference["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
            nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()

            def value():
                guard()
                root.RooAbsReal.clearEvalErrorLog()
                result = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                if not np.isfinite(result) or errors:
                    raise ValueError(f"Invalid Combine NLL: value={result}, RooFit error count={errors}")
                return result

            for selected in selection["cases"]:
                previous_point = (next(p for p in previous["points"] if p["point_number"] == selected["point_number"]
                                       and p["index"] == selected["index"]) if previous else None)
                if previous_point and previous_point["status"] == "reference_ready":
                    report["points"].append(previous_point)
                    continue
                fractions = selection["fractions"]
                if plan:
                    extension = next(c for c in plan["cases"] if c["point_number"] == selected["point_number"]
                                     and c["index"] == selected["index"])
                    assert extension["name"] == selected["name"] and extension["attempt"] == selected["attempt"]
                    first = None
                    for j,p in enumerate(previous_point["probes"][:-1]):
                        values = list(p["stable"].values())
                        if (max(values)-min(values))/max(min(abs(v) for v in values),1e-8) < 1e-5:
                            first = j
                            break
                    assert first == extension["first_individually_consistent_probe"] and first is not None
                    high, low = (p["fraction"] for p in previous_point["probes"][first:first+2])
                    assert extension["interval_fractions"] == [high,low] and plan["subdivisions"] == 10
                    fractions = [low+(high-low)*k/10 for k in range(9,0,-1)]
                    assert fractions == extension["added_fractions"]
                    assert not set(fractions) & {p["fraction"] for p in previous_point["probes"]}
                old = next(p for p in reference["points"] if p["point_number"] == selected["point_number"])
                assert old["attempt"] == selected["attempt"]
                for variable, coordinate in zip(variables, old["theta"], strict=True):
                    variable.setVal(float(coordinate))
                nll.clearZeroPoint()
                canonical = value()
                assert abs(canonical-old["canonical_nll"]) < 1e-6
                nll.setZeroPoint()
                zero = value()
                assert abs(zero) < 1e-6
                point = dict(point_number=old["point_number"], attempt=old["attempt"], theta=old["theta"],
                             canonical_nll=canonical, canonical_parent_absolute_error=abs(canonical-old["canonical_nll"]),
                             name=selected["name"], index=selected["index"], channels=[], probes=[])
                if previous_point:
                    assert previous_point["theta"] == point["theta"]
                    point.update(probes=list(previous_point["probes"]), added_fractions=fractions,
                                 original_probe_count=len(previous_point["probes"]))
                report["points"].append(point)
                i, name = selected["index"], selected["name"]
                parameter, variable, coordinate = likelihood.parameters[i], variables[i], old["theta"][i]
                assert parameter.name == name
                predictors = []
                for channel in likelihood.channels:
                    if name not in dependencies[channel.name]:
                        continue
                    assert not channel.masked
                    assert not any(p.propagator in dependencies[channel.name] for p in likelihood.profiles)
                    predictor = RootIntensities(root, workspace.pdf(channel.pdf),
                        [workspace.var(n) for n in channel.observables],
                        np.column_stack([channel.data.coordinates[n] for n in channel.observables]), channel.data.weights)
                    predictors.append((predictor,predictor.predict()))
                    point["channels"].append(dict(channel=channel.name, **predictor.report))
                assert predictors
                scale = parameter.error if parameter.error > 0 else max(abs(parameter.value), 1.)
                point.update(coordinate=coordinate, scale=scale)
                save()
                for fraction in fractions:
                    step = fraction*scale
                    assert parameter.minimum <= coordinate-4*step < coordinate < coordinate+4*step <= parameter.maximum
                    probe = dict(fraction=fraction, step=step, delta_checks=[])
                    stable, full = {}, {}
                    for multiple in (-4,-3,-2,-1,1,2,3,4):
                        shifted = coordinate+multiple*step
                        variable.setVal(float(shifted))
                        actual = value()-zero
                        delta = constraint_delta(likelihood, workspace, name, coordinate, shifted)
                        for predictor, baseline in predictors:
                            delta += predictor.delta(baseline)
                        errors = int(root.RooAbsReal.numEvalErrors())
                        assert errors == 0 and np.isfinite(delta)
                        error = abs(float(delta)-actual)
                        probe["delta_checks"].append(dict(multiple=multiple, coordinate=shifted,
                            stable_delta=float(delta), combine_delta=actual, absolute_error=error, roofit_evaluation_errors=errors))
                        assert error < 1e-6, (name,probe["delta_checks"][-1])
                        stable[multiple], full[multiple] = delta, LD(actual)
                    for kind, values in (("stable",stable), ("full_nll",full)):
                        central = (8*(values[1]-values[-1])-(values[2]-values[-2]))/(12*LD(step))
                        forward = sum(LD(a)*values[k] for a,k in zip((48,-36,16,-3),(1,2,3,4),strict=True))/(12*LD(step))
                        backward = sum(LD(a)*values[-k] for a,k in zip((48,-36,16,-3),(1,2,3,4),strict=True))/(-12*LD(step))
                        probe[kind] = dict(central=float(central), forward=float(forward), backward=float(backward))
                    point["probes"].append(probe)
                    variable.setVal(float(coordinate))
                    assert abs(value()-zero) < 1e-6
                    print(point["point_number"],name,step,probe["stable"],flush=True)
                    save()
                point["probes"].sort(key=lambda p:p["fraction"], reverse=True)
                point.update(status="reference_uncertain", selected_first_probe_index=None)
                for first in range(len(point["probes"])-2):
                    values = [v for p in point["probes"][first:first+3] for v in p["stable"].values()]
                    spread = (max(values)-min(values))/max(min(abs(v) for v in values), 1e-8)
                    if spread < 1e-5:
                        point.update(status="reference_ready", selected_first_probe_index=first, selected_relative_spread=spread)
                        break
                nll.clearZeroPoint()
                point["canonical_restoration_absolute_error"] = abs(value()-canonical)
                assert point["canonical_restoration_absolute_error"] < 1e-6
                print("complete",point["point_number"],name,point["status"],flush=True)
            del nll
        report["status_counts"] = dict(Counter(p["status"] for p in report["points"]))
        report["status"] = "reference_ready" if all(p["status"] == "reference_ready" for p in report["points"]) else "reference_uncertain"
        report["max_combine_delta_absolute_error"] = max(d["absolute_error"] for p in report["points"] for q in p["probes"] for d in q["delta_checks"])
        save()
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
