"""Independent v11 finite differences for full HHVV gradient projections."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refine", action="store_true", help="Scan smaller fixed steps for all saved directions; preserve the original references")
    arguments = parser.parse_args()
    start = time.perf_counter()
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    source = manifest["models"]["hhvv"]["source"]
    likelihood = load_export("build/gpu_exports/hhvv.pkl", source)
    initial = likelihood.initial
    scales = np.array([p.error if p.error > 0 else max(abs(p.value), 1.) for p in likelihood.parameters])
    low = np.array([p.minimum for p in likelihood.parameters])
    high = np.array([p.maximum for p in likelihood.parameters])
    # Keep dense symmetric probes away from bounds; separate one-sided probes
    # can cover designated POIs if the saved nominal point is on a boundary.
    margin = np.minimum(initial-low, high-initial)
    interior = margin > .01*scales
    rng = np.random.default_rng(9052000)
    directions = [(f"random_{i}", rng.uniform(-1., 1., len(initial))*scales*interior, "central") for i in range(16)]
    for i, p in enumerate(likelihood.parameters):
        if p.role != "poi" and p.name != "CMS_B2G20008_scale_j":
            continue
        direction = np.zeros_like(initial)
        direction[i] = scales[i]
        if margin[i] <= .004*scales[i]:
            direction[i] *= 1 if high[i]-initial[i] >= initial[i]-low[i] else -1
            stencil = "forward"
        else:
            stencil = "central"
        directions.append((p.name, direction, stencil))
    report = dict(source=source, combine_version=manifest["combine_version"], root_version=manifest["root_version"],
                  runtime_flags=REFERENCE_FLAGS, seed=9052000, parameter_names=list(likelihood.parameter_names),
                  initial=initial.tolist(), parameters_in_dense_probes=int(interior.sum()),
                  excluded_near_bounds=[n for n, flag in zip(likelihood.parameter_names, interior, strict=True) if not flag],
                  scope="Full-model gradient projections; does not replace the main per-parameter FD acceptance gate",
                  cases=[], status="running")
    output = Path("reports/gpu_combine_directions_refined.json" if arguments.refine else "reports/gpu_combine_directions.json")
    steps = (1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6, 1e-6) if arguments.refine else (1e-3, 5e-4)
    if arguments.refine:
        parent_path = Path("reports/gpu_combine_directions.json")
        parent = json.loads(parent_path.read_text())
        assert parent["source"] == source and parent["status"] == "reference_ready"
        np.testing.assert_array_equal(parent["initial"], initial)
        np.testing.assert_array_equal(parent["parameter_names"], likelihood.parameter_names)
        for saved, (label, direction, stencil) in zip(parent["cases"], directions, strict=True):
            assert saved["label"] == label and saved["stencil"] == stencil
            np.testing.assert_array_equal(saved["direction"], direction)
        report.update(parent_reference=str(parent_path), parent_reference_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                      refinement_policy="All 18 original directions, same fixed smaller-step sequence; no direction removal or AD-based step selection",
                      steps=list(steps))

    def save():
        report["elapsed_seconds"] = time.perf_counter()-start
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    root = load_caching_nll()
    with open_workspace("inputs/workspace_MX2000.root") as workspace:
        configure_hhvv(workspace)
        variables = [workspace.var(n) for n in likelihood.parameter_names]
        model = workspace.obj("ModelConfig")
        reference = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
        reference.setChannelMasks(model.GetPdf().channelMasks())
        reference.setAnalyticBarlowBeeston(True)
        reference.clearConstantZeroPoint(); reference.clearZeroPoint()

        def value(theta):
            if np.any(theta < low) or np.any(theta > high):
                raise ValueError("Finite difference leaves an allowed parameter range")
            for variable, coordinate in zip(variables, theta, strict=True):
                variable.setVal(float(coordinate))
            result = float(reference.getVal())
            if not np.isfinite(result):
                raise ValueError("Nonfinite Combine NLL in directional probe")
            return result

        nominal = value(initial)
        report["combine_nll"] = nominal
        assert abs(nominal-manifest["models"]["hhvv"]["reference"]["cases"][0]["combine_nll"]) < 1e-6
        for label, direction, stencil in directions:
            case = dict(label=label, direction=direction.tolist(), stencil=stencil, probes=[])
            for step in steps:
                if stencil == "central":
                    values = [value(initial+t*step*direction) for t in (-2,-1,1,2)]
                    derivative = (8*(values[2]-values[1])-(values[3]-values[0]))/(12*step)
                else:
                    values = [nominal]+[value(initial+t*step*direction) for t in (1,2,3,4)]
                    derivative = sum(c*(v-nominal) for c,v in zip((48,-36,16,-3),values[1:],strict=True))/(12*step)
                case["probes"].append(dict(step=step, nll_values=values, derivative=derivative))
            case["step_halving_difference"] = abs(case["probes"][-1]["derivative"]-case["probes"][-2]["derivative"])
            case["step_halving_relative_change"] = case["step_halving_difference"]/max(abs(case["probes"][-1]["derivative"]),1e-8)
            report["cases"].append(case)
            print(label, [p["derivative"] for p in case["probes"]], flush=True)
            save()
        value(initial)
    report["status"] = "reference_ready"
    save()


if __name__ == "__main__":
    main()
