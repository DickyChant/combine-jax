"""Freeze common profile points from an accepted JAX/native fit comparison."""

import argparse
import json
from pathlib import Path

import numpy as np

from compare_exported_fit import compare, digest, load_arrays


MANIFEST = Path("build/gpu_exports/manifest_random.json")


def validated_comparison(path):
    report = json.loads(path.read_bytes())
    assert report["status"] == "passed", "A profile requires accepted baseline fits"
    jax_path, native_path = Path(report["benchmark"]), Path(report["reference"])
    assert digest(jax_path) == report["benchmark_sha256"]
    assert digest(native_path) == report["reference_sha256"]
    assert report["checker_sha256"] == digest(Path(__file__).with_name("compare_exported_fit.py"))
    # Recompute acceptance; tiny BLAS rounding differences between the two
    # runtimes need not reproduce a diagnostic residual bit-for-bit.
    assert compare(jax_path, native_path)["status"] == "passed", "Baseline comparison no longer passes"
    return report, json.loads(jax_path.read_bytes()), json.loads(native_path.read_bytes())


def make_plan(comparison_path, likelihood, parameter, poi, offsets):
    comparison, jax, native = validated_comparison(comparison_path)
    arrays = load_arrays(native)
    names = list(likelihood.parameter_names)
    np.testing.assert_array_equal(names, native["parameter_names"])
    assert json.loads(json.dumps(likelihood.discrete_choices)) == native["discrete_choices"]
    assert json.loads(json.dumps(likelihood.mask_configuration)) == native["channel_masks"]
    if parameter not in names or poi not in names or parameter == poi:
        raise ValueError("Choose distinct active nuisance and POI names")
    if parameter in native["fixed_bounds"] or poi in native["fixed_bounds"]:
        raise ValueError("Profile nuisance and POI must be free in the baseline")
    offsets = np.asarray(offsets, dtype=np.float64)
    if (offsets.ndim != 1 or len(offsets) < 3 or not np.isfinite(offsets).all()
            or len(set(offsets.tolist())) != len(offsets) or 0. not in offsets
            or not np.any(offsets < 0) or not np.any(offsets > 0)):
        raise ValueError("Use distinct finite offsets including zero and both sides")
    selected = names.index(parameter)
    baseline = arrays["values"]
    errors = arrays["errors"]
    parameter_info = likelihood.parameters[selected]
    assert np.isfinite(errors[selected]) and errors[selected] > 0.
    points = baseline[selected]+offsets*errors[selected]
    if not np.all((points >= parameter_info.minimum) & (points <= parameter_info.maximum)):
        raise ValueError("Requested profile points exceed nuisance bounds; points are not clipped")
    for p, value in zip(likelihood.parameters, baseline, strict=True):
        assert p.minimum <= value <= p.maximum
    plan = dict(
        status="profile_plan_ready", model=native["model"], source=native["source"], host=native["host"],
        comparison=str(comparison_path), comparison_sha256=digest(comparison_path),
        comparison_checker_sha256=comparison["checker_sha256"], runner_sha256=digest(__file__),
        benchmark=comparison["benchmark"], benchmark_sha256=comparison["benchmark_sha256"],
        reference=comparison["reference"], reference_sha256=comparison["reference_sha256"],
        reference_arrays=native["arrays"], reference_arrays_sha256=native["arrays_sha256"],
        parameter_names=names, discrete_choices=native["discrete_choices"], channel_masks=native["channel_masks"],
        fixed_bounds=native["fixed_bounds"], initial=baseline.tolist(),
        canonical_baseline_nll=native["canonical_fitted_nll"], baseline_errors=errors.tolist(),
        parameter=parameter, parameter_index=selected, poi=poi, poi_index=names.index(poi),
        points=[dict(offset_sigma=float(offset), value=float(value))
                for offset, value in zip(offsets, points, strict=True)],
        point_definition="Native fitted nuisance value plus offset times its native Minuit-reported error. Both engines use these identical physical values and the same native fitted initial vector.",
        scope="Reoptimized profile points and POI shifts at fixed nuisance values. The offsets are not claims of likelihood-based confidence endpoints or a full discrete envelope.",
        export_stack=dict(combine_version=native["combine_version"], root_version=native["root_version"]))
    return plan


def load_plan(path, likelihood, source, *, host):
    """Recheck the accepted baseline, point construction and export before use."""
    plan = json.loads(path.read_bytes())
    assert plan["status"] == "profile_plan_ready" and plan["source"] == source
    assert plan["host"] == host, "Run the profile on the baseline fit host"
    assert digest(plan["comparison"]) == plan["comparison_sha256"]
    assert digest(MANIFEST) == plan["manifest_sha256"]
    manifest = json.loads(MANIFEST.read_bytes())
    model = manifest["models"][plan["model"]]
    assert model["source"] == source
    export = Path("build/gpu_exports")/model["cache"]
    assert str(export) == plan["export"] and digest(export) == plan["export_sha256"]
    expected = make_plan(Path(plan["comparison"]), likelihood, plan["parameter"], plan["poi"],
                         [p["offset_sigma"] for p in plan["points"]])
    for key, value in expected.items():
        assert plan[key] == value, key
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--parameter", required=True)
    parser.add_argument("--poi", required=True)
    parser.add_argument("--offsets", type=float, nargs="+", default=[-2., -1., 0., 1., 2.])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve existing profile plans")
    comparison, _, reference = validated_comparison(args.comparison)
    manifest = json.loads(MANIFEST.read_bytes())
    model = manifest["models"][reference["model"]]
    assert model["source"] == reference["source"]
    from hhvv_export_cache import load_export
    export = Path("build/gpu_exports")/model["cache"]
    likelihood = load_export(export, model["source"])
    plan = make_plan(args.comparison, likelihood, args.parameter, args.poi, args.offsets)
    plan.update(manifest=str(MANIFEST), manifest_sha256=digest(MANIFEST),
                export=str(export), export_sha256=digest(export))
    with args.output.open("x") as stream:
        stream.write(json.dumps(plan, indent=2, allow_nan=False)+"\n")
    print(args.output, plan["status"], len(plan["points"]), "points")


if __name__ == "__main__":
    main()
