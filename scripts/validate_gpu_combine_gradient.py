"""Build per-parameter v11 finite differences for the nominal GPU comparison."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def add_step_quality(report):
    ready = [c for c in report["cases"] if c["status"] == "reference_ready"]
    for case in ready:
        derivative = case["probes"][-1]["derivative"]
        case["step_halving_relative_change"] = case["step_halving_difference"]/max(abs(derivative), 1e-8)
    report["step_quality"] = dict(
        max_relative_change=max(c["step_halving_relative_change"] for c in ready),
        max_scaled_absolute_change=max(c["step_halving_difference"]*c["scale"] for c in ready),
        names_above_1e_4=[c["name"] for c in ready if c["step_halving_relative_change"] >= 1e-4],
        note="Step disagreement is reference uncertainty, not evidence of an AD error; report it alongside AD comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--refine", action="store_true", help="Probe additional steps for uncertain saved references")
    mode.add_argument("--fine", action="store_true", help="Probe the same fixed fine-step sequence for every nonfixed parameter, preserving previous references")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    started = time.perf_counter()
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    source = manifest["models"]["hhvv"]["source"]
    likelihood = load_export("build/gpu_exports/hhvv.pkl", source)
    initial = likelihood.initial
    report = dict(status="running", source=source, combine_version=manifest["combine_version"],
                  root_version=manifest["root_version"], runtime_flags=REFERENCE_FLAGS,
                  parameter_names=list(likelihood.parameter_names), initial=initial.tolist(),
                  scope="Nominal independent per-parameter Combine finite differences; no AD acceptance claim until compared",
                  offset="Canonical NLL checked first; setZeroPoint for per-channel cancellation reduction in FD",
                  cpu_affinity=sorted(os.sched_getaffinity(0)), cases=[], offset_checks=[])
    output = Path("reports/gpu_combine_gradient_fine.json" if arguments.fine else "reports/gpu_combine_gradient_refined.json" if arguments.refine else "reports/gpu_combine_gradient.json")
    previous = None
    if arguments.refine or arguments.fine:
        previous_path = Path("reports/gpu_combine_gradient.json")
        previous_report = json.loads(previous_path.read_text())
        assert previous_report["source"] == source
        if arguments.refine:
            previous = {c["name"]: c for c in previous_report["cases"]
                        if c.get("step_halving_relative_change", 0.) >= 1e-4}
        report.update(parent_reference=str(previous_path),
                      parent_reference_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                      selection=("All nonfixed parameters with the same fixed fine-step sequence; no AD values used for selection" if arguments.fine
                                 else "Parent FD step-halving relative change >= 1e-4; no AD values used for selection"))

    def save():
        report["elapsed_seconds"] = time.perf_counter()-started
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    root = load_caching_nll()
    with open_workspace("inputs/workspace_MX2000.root") as workspace:
        configure_hhvv(workspace)
        variables = [workspace.var(n) for n in likelihood.parameter_names]
        model = workspace.obj("ModelConfig")
        reference = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
        reference.setChannelMasks(model.GetPdf().channelMasks())
        reference.setAnalyticBarlowBeeston(True)
        reference.clearConstantZeroPoint(); reference.clearZeroPoint()
        nominal = float(reference.getVal())
        report["canonical_combine_nll"] = nominal
        assert abs(nominal-manifest["models"]["hhvv"]["reference"]["cases"][0]["combine_nll"]) < 1e-6
        reference.setZeroPoint()
        zero = float(reference.getVal())
        report["zero_point_nll"] = zero
        assert abs(zero) < 1e-6
        directions = json.loads(Path("reports/gpu_combine_directions.json").read_text())
        for case in directions["cases"][:2]:
            probe = case["probes"][0]
            theta = initial+probe["step"]*np.asarray(case["direction"])
            for variable, value in zip(variables, theta, strict=True):
                variable.setVal(float(value))
            actual = float(reference.getVal())
            expected = probe["nll_values"][2]-nominal
            error = abs(actual-expected)
            report["offset_checks"].append(dict(label=case["label"], zero_point_value=actual,
                                                canonical_difference=expected, absolute_error=error))
            assert error < 1e-6
        for variable, value in zip(variables, initial, strict=True):
            variable.setVal(float(value))
        reference.getVal()

        for i, (p, variable) in enumerate(zip(likelihood.parameters, variables, strict=True)):
            if previous is not None and p.name not in previous:
                continue
            case = dict(index=i, name=p.name, initial=p.value, minimum=p.minimum, maximum=p.maximum,
                        scale=p.error if p.error > 0 else max(abs(p.value), 1.))
            if p.minimum == p.maximum:
                case.update(status="fixed_bounds", reason="No allowed finite-difference displacement")
            else:
                case["probes"] = []
                try:
                    left, right = p.value-p.minimum, p.maximum-p.value
                    fractions = (1e-5, 5e-6, 2e-6, 1e-6, 5e-7) if arguments.fine else (1e-3, 5e-4)
                    if previous is not None:
                        tiny = abs(previous[p.name]["probes"][-1]["derivative"]) < 1e-4
                        fractions = ((1e-3, 5e-4, .005, .01, .025, .05, .1) if tiny
                                     else (1e-3, 5e-4, 2e-4, 1e-4, 4e-5, 2e-5, 1e-5))
                        case["step_strategy"] = "Larger steps to inspect roundoff" if tiny else "Smaller steps to inspect truncation"
                    for fraction in fractions:
                        step = fraction*case["scale"]
                        if min(left, right) >= 2*step:
                            stencil, direction, multiples = "central", 1., (-2, -1, 1, 2)
                        else:
                            stencil, direction, multiples = "forward", (1. if right >= left else -1.), (1, 2, 3, 4)
                            step = min(step, max(left, right)/5)
                        assert step > 0 and p.value+direction*step != p.value
                        values = []
                        for multiple in multiples:
                            coordinate = p.value+direction*multiple*step
                            assert p.minimum <= coordinate <= p.maximum
                            variable.setVal(float(coordinate))
                            value = float(reference.getVal())
                            if not np.isfinite(value):
                                raise ValueError("Nonfinite Combine NLL")
                            values.append(value)
                        if stencil == "central":
                            derivative = (8*(values[2]-values[1])-(values[3]-values[0]))/(12*step)
                        else:
                            derivative = sum(c*(v-zero) for c,v in zip((48,-36,16,-3), values, strict=True))/(12*direction*step)
                        case["probes"].append(dict(step=step, stencil=stencil, direction=direction,
                                                   zero_point_nll_values=values, derivative=derivative))
                    case.update(status="reference_ready", step_halving_difference=abs(
                        case["probes"][-1]["derivative"]-case["probes"][-2]["derivative"]))
                    if previous is not None and not tiny:
                        a, b, c = case["probes"][-3:]
                        if (np.isclose(a["step"], 2*b["step"], rtol=1e-12, atol=0.)
                                and np.isclose(b["step"], 2*c["step"], rtol=1e-12, atol=0.)):
                            estimates = [2*b["derivative"]-a["derivative"], 2*c["derivative"]-b["derivative"]]
                            case["first_order_extrapolation"] = dict(estimates=estimates,
                                successive_difference=abs(estimates[1]-estimates[0]),
                                note="Diagnostic for an O(step) error; does not establish differentiability or replace the raw probes")
                except Exception:
                    case.update(status="failed", exception=traceback.format_exc())
                    print("Failed reference", p.name, case["exception"], flush=True)
                finally:
                    variable.setVal(float(p.value))
                    reference.getVal()
            report["cases"].append(case)
            if (i+1) % 100 == 0 or previous is not None:
                print(f"Reference {i+1}/{len(initial)}: {p.name}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
                save()
        reference.clearZeroPoint()
        report["canonical_nll_after_restore"] = float(reference.getVal())
        assert abs(report["canonical_nll_after_restore"]-nominal) < 1e-6
    report["status_counts"] = {status: sum(c["status"] == status for c in report["cases"])
                               for status in ("reference_ready", "fixed_bounds", "failed")}
    report["status"] = "reference_ready" if not report["status_counts"]["failed"] else "reference_incomplete"
    add_step_quality(report)
    save()
    print(report["status"], report["status_counts"], report["elapsed_seconds"], flush=True)


if __name__ == "__main__":
    main()
