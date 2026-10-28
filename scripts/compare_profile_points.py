"""Compare reoptimized native/JAX profile samples on one frozen physical grid."""

import argparse
import json
from pathlib import Path

import numpy as np

from profile_plan import digest, validated_comparison


def load_profile(report, plan):
    assert report["status"] == "profile_computed"
    for key in ("model", "source", "host", "parameter_names", "parameter", "poi",
                "discrete_choices", "channel_masks", "initial"):
        assert report[key] == plan[key], key
    assert report["plan_helper_sha256"] == plan["runner_sha256"]
    assert report["baseline_nll"] == plan["canonical_baseline_nll"]
    assert 0 <= report["baseline_nll_absolute_error"] < 1e-6
    assert digest(report["arrays"]) == report["arrays_sha256"]
    with np.load(report["arrays"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(arrays["parameter_names"], plan["parameter_names"])
    np.testing.assert_array_equal(arrays["initial"], plan["initial"])
    points, n = len(plan["points"]), len(plan["parameter_names"])
    assert arrays["values"].shape == (points, n) and arrays["canonical_nll"].shape == (points,)
    for key in ("values", "canonical_nll", "fixed_values"):
        assert arrays[key].dtype == np.float64 and np.isfinite(arrays[key]).all(), key
    np.testing.assert_array_equal(arrays["fixed_values"], [p["value"] for p in plan["points"]])
    np.testing.assert_array_equal(arrays["values"][:, plan["parameter_index"]], arrays["fixed_values"])
    for name in plan["fixed_bounds"]:
        index = plan["parameter_names"].index(name)
        np.testing.assert_array_equal(arrays["values"][:, index], np.full(points, plan["initial"][index]))
    assert len(report["points"]) == points
    for index, (record, point) in enumerate(zip(report["points"], plan["points"], strict=True)):
        assert record["valid_minimum"] and record["index"] == index
        assert record["value"] == point["value"] and record["offset_sigma"] == point["offset_sigma"]
        assert record["canonical_nll"] == arrays["canonical_nll"][index]
        assert record["delta_nll"] == arrays["canonical_nll"][index]-plan["canonical_baseline_nll"]
        assert record["poi_value"] == arrays["values"][index, plan["poi_index"]]
        assert record["poi_shift"] == record["poi_value"]-plan["initial"][plan["poi_index"]]
        if report["engine"] == "combine":
            assert record["native_final_evaluation_errors"] == 0
            assert 0 <= record["offset_restoration_error"] < 1e-6
    return arrays


def compare_arrays(jax, native, plan):
    names = plan["parameter_names"]
    free = [i for i, name in enumerate(names)
            if name not in plan["fixed_bounds"] and name != plan["parameter"]]
    errors = np.asarray(plan["baseline_errors"])
    assert np.isfinite(errors).all() and np.all(errors[free] > 0.)
    poi = plan["poi_index"]
    assert poi in free
    results = []
    for i, point in enumerate(plan["points"]):
        coordinate_errors = abs(jax["values"][i, free]-native["values"][i, free])
        limits = np.maximum(.01*abs(native["values"][i, free]), .01*errors[free])
        shift_jax = jax["values"][i, poi]-plan["initial"][poi]
        shift_native = native["values"][i, poi]-plan["initial"][poi]
        shift_error = abs(shift_jax-shift_native)
        shift_limit = max(.01*abs(shift_native), .01*errors[poi])
        nll_error = abs(jax["canonical_nll"][i]-native["canonical_nll"][i])
        failed_coordinates = [names[j] for j, error, limit in zip(free, coordinate_errors, limits, strict=True)
                              if error > limit]
        passed = not failed_coordinates and shift_error <= shift_limit and nll_error < 1e-6
        results.append(dict(index=i, **point, status="passed" if passed else "failed",
            nll_absolute_error=float(nll_error), coordinate_max_absolute_error=float(coordinate_errors.max(initial=0.)),
            failed_coordinates=failed_coordinates, poi_shift_jax=float(shift_jax), poi_shift_native=float(shift_native),
            poi_shift_absolute_error=float(shift_error), poi_shift_limit=float(shift_limit),
            delta_nll_jax=float(jax["canonical_nll"][i]-plan["canonical_baseline_nll"]),
            delta_nll_native=float(native["canonical_nll"][i]-plan["canonical_baseline_nll"])))
    return results


def compare(jax_path, native_path):
    jax, native = [json.loads(p.read_bytes()) for p in (jax_path, native_path)]
    assert jax["engine"] in ("cpu", "gpu") and native["engine"] == "combine"
    assert jax["plan_sha256"] == native["plan_sha256"]
    for report in (jax, native):
        assert digest(report["plan"]) == report["plan_sha256"]
    plan = json.loads(Path(jax["plan"]).read_bytes())
    assert plan["status"] == "profile_plan_ready"
    assert digest(plan["comparison"]) == plan["comparison_sha256"]
    _, _, baseline = validated_comparison(Path(plan["comparison"]))
    assert plan["reference_sha256"] == digest(plan["reference"])
    assert plan["reference_arrays_sha256"] == baseline["arrays_sha256"]
    for key in ("initial", "parameter_names", "source", "discrete_choices", "channel_masks"):
        expected = baseline["final"]["values"] if key == "initial" else baseline[key]
        assert plan[key] == expected, key
    assert plan["baseline_errors"] == baseline["final"]["errors"]
    assert plan["canonical_baseline_nll"] == baseline["canonical_fitted_nll"]
    for point in plan["points"]:
        assert point["value"] == (plan["initial"][plan["parameter_index"]]
                                  + point["offset_sigma"]*plan["baseline_errors"][plan["parameter_index"]])
    for key in ("core_source_sha256", "runner_sha256", "optimizer"):
        assert jax[key] == native[key], key
    result = dict(status="incomplete", scope=plan["scope"], host=plan["host"], model=plan["model"],
        benchmark=str(jax_path), benchmark_sha256=digest(jax_path),
        reference=str(native_path), reference_sha256=digest(native_path),
        plan=jax["plan"], plan_sha256=jax["plan_sha256"], checker_sha256=digest(__file__),
        producer_statuses=dict(jax=jax["status"], native=native["status"]),
        tolerance="Canonical/Delta NLL difference <1e-6; every reoptimized coordinate and POI shift within max(1% absolute native result, 0.01 native baseline Minuit sigma).")
    if jax["status"] != "profile_computed" or native["status"] != "profile_computed":
        result["unresolved"] = "Both complete profile grids must contain valid minima"
        return result
    assert jax["float64"] and not jax["root_imported"] and native["root_imported"]
    for key in ("combine_version", "root_version", "runtime_flags"):
        assert native["native_reference"][key] == baseline[key]
    ja, na = load_profile(jax, plan), load_profile(native, plan)
    points = compare_arrays(ja, na, plan)
    result.update(status="passed" if all(p["status"] == "passed" for p in points) else "failed",
                  points=points, arrays_sha256=jax["arrays_sha256"],
                  reference_arrays_sha256=native["arrays_sha256"],
                  nll_max_absolute_error=max(p["nll_absolute_error"] for p in points),
                  coordinate_max_absolute_error=max(p["coordinate_max_absolute_error"] for p in points),
                  poi_shift_max_absolute_error=max(p["poi_shift_absolute_error"] for p in points),
                  observed_timings=dict(scope="Recorded times on the same host and initial vector; no isolated speedup ratio.",
                      jax_elapsed_seconds=jax["elapsed_seconds"], native_elapsed_seconds=native["elapsed_seconds"],
                      jax_point_seconds=[p["elapsed_seconds"] for p in jax["points"]],
                      native_point_seconds=[p["elapsed_seconds"] for p in native["points"]],
                      jax_affinity=jax["cpu_affinity"], native_affinity=native["cpu_affinity"],
                      jax_timing_scope=jax.get("timing_scope", "Legacy profile phase after device/export/plan initialization"),
                      native_timing_scope=native.get("timing_scope", "Legacy profile phase after device/export/plan initialization"),
                      jax_version=jax["jax_version"], jaxlib_version=jax["jaxlib_version"],
                      jax_iminuit_version=jax["iminuit_version"], native_iminuit_version=native["iminuit_version"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous profile comparisons")
    result = compare(args.benchmark, args.reference)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output, result["status"])
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
