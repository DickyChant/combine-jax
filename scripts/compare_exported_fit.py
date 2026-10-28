"""Compare saved JAX/Combine fits on one host, with GOAL.md tolerances.

This reads numerical artifacts only. It does not run a fit or import ROOT/JAX.
Conditional fits are compared at the recorded discrete choices and masks.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_arrays(report):
    assert digest(report["arrays"]) == report["arrays_sha256"], "arrays_sha256"
    with np.load(report["arrays"], allow_pickle=False) as arrays:
        result = {key: arrays[key] for key in arrays.files}
    np.testing.assert_array_equal(result["parameter_names"], report["parameter_names"])
    np.testing.assert_array_equal(result["values"], report["final"]["values"])
    return result


def checked_start(report, arrays=None):
    point = np.asarray(report.get("fit_initial", report["initial"]), dtype=np.float64)
    if report.get("initial_check"):
        path = Path(report["initial_check"])
        assert digest(path) == report["initial_check_sha256"]
        checked = json.loads(path.read_bytes())
        assert checked["status"] == "candidate_checked" and checked["source"] == report["source"]
        assert digest(checked["parent"]) == checked["parent_sha256"]
        for key in ("parameter_names", "discrete_choices", "channel_masks"):
            assert checked[key] == report[key], key
        np.testing.assert_array_equal(point, checked["point"])
    if arrays is not None:
        np.testing.assert_array_equal(point, arrays["initial"])
    assert point.shape == (len(report["parameter_names"]),) and np.isfinite(point).all()
    return point


def compare(jax_path, reference_path):
    jax = json.loads(jax_path.read_bytes())
    native = json.loads(reference_path.read_bytes())
    for key in ("model", "source", "host", "parameter_names", "discrete_choices", "channel_masks"):
        assert jax[key] == native[key], key
    assert len(set(jax["parameter_names"])) == len(jax["parameter_names"])
    assert native["combine_version"] == jax["export_stack"]["combine_version"] == "11.0.0"
    assert native["root_version"] == jax["export_stack"]["root_version"] == "6.40.00"
    assert jax["float64"]
    assert jax["optimizer"]["errordef"] == native["optimizer"]["errordef"] == .5
    result = dict(
        status="incomplete", benchmark=str(jax_path), benchmark_sha256=digest(jax_path),
        reference=str(reference_path), reference_sha256=digest(reference_path),
        checker_sha256=digest(__file__), source=jax["source"], host=jax["host"], model=jax["model"],
        scope="Conditional continuous best-fit and full AD covariance comparison at identical recorded categories/masks. No discrete-envelope or profile/impact acceptance, and no isolated speedup claim.",
        tolerance=dict(coordinates="max(0.01*abs(reference value), 0.01*reference Minuit error)",
                       uncertainties="0.01*reference Minuit error for each nonfixed parameter",
                       canonical_nll_absolute=1e-6, hessian_covariance_identity_absolute=1e-6,
                       scaled_matrix_asymmetry_absolute=1e-6),
        uncertainty_reference="Native Minuit-reported errors; covariance-diagonal sigmas are also reported as a distinct diagnostic.",
        versions=dict(jax=jax["jax_version"], jaxlib=jax["jaxlib_version"],
                      jax_iminuit=jax["iminuit_version"], combine=native["combine_version"],
                      root=native["root_version"], native_iminuit=native["iminuit_version"]),
        execution=dict(backend=jax["backend"], devices=jax["devices"], lcg_view=jax.get("lcg_view"),
                       gpu_metadata=jax.get("gpu_metadata"), jax_affinity=jax["cpu_affinity"],
                       native_affinity=native["cpu_affinity"], core_source_sha256=jax["source_sha256"],
                       jax_runner_sha256=jax["runner_sha256"], native_runner_sha256=native["runner_sha256"]),
        producer_statuses=dict(jax=jax["status"], native=native["status"]))
    jf, nf = jax.get("final", {}), native.get("final", {})
    not_ready = []
    if jax["status"] != "fit_computed" or not jf.get("valid_minimum"):
        not_ready.append("JAX fit has no accepted minimum with requested covariance")
    if jf.get("covariance_status") not in ("positive_definite", "all_fixed"):
        not_ready.append("JAX covariance is unavailable or not positive definite")
    if (native["status"] != "reference_ready"
            or not all(nf.get(key) for key in ("is_valid", "has_accurate_covar", "has_posdef_covar"))
            or any(nf.get(key) for key in ("has_made_posdef_covar", "is_above_max_edm",
                                          "hesse_failed", "has_reached_call_limit"))):
        not_ready.append("Native reference lacks a valid minimum and accurate positive covariance")
    if not_ready:
        result["unresolved"] = not_ready
        return result

    assert not jax["root_imported"]
    ja, na = load_arrays(jax), load_arrays(native)
    jstart, nstart = checked_start(jax, ja), checked_start(native)
    names = jax["parameter_names"]
    n = len(names)
    fixed_names = set(jf["fixed"])
    assert fixed_names == set(jax["fixed_bounds"]) == set(native["fixed_bounds"]), "fixed settings"
    assert fixed_names <= set(names)
    free = np.asarray([i for i, name in enumerate(names) if name not in fixed_names], dtype=int)
    fixed = np.asarray([i for i, name in enumerate(names) if name in fixed_names], dtype=int)
    for arrays, matrix_keys in ((ja, ("hessian", "covariance")), (na, ("covariance",))):
        assert arrays["values"].shape == (n,)
        assert arrays["values"].dtype == np.float64 and np.isfinite(arrays["values"]).all()
        for key in matrix_keys:
            matrix = arrays[key]
            assert matrix.shape == (n, n) and matrix.dtype == np.float64 and np.isfinite(matrix).all(), key
        assert np.all(arrays["covariance"][fixed, :] == 0.)
        assert np.all(arrays["covariance"][:, fixed] == 0.)
    assert ja["gradient"].shape == (n,) and ja["gradient"].dtype == np.float64
    assert np.isfinite(ja["gradient"]).all()
    np.testing.assert_array_equal(ja["values"][fixed], na["values"][fixed])
    np.testing.assert_array_equal(ja["values"], jax["curvature_point"])
    np.testing.assert_array_equal(na["errors"], nf["errors"])
    reference_error = na["errors"][free]
    assert reference_error.shape == (len(free),) and np.isfinite(reference_error).all()
    assert np.all(reference_error > 0.)
    for arrays in (ja, na):
        assert np.all(np.diag(arrays["covariance"])[free] > 0.)
    covariance, reference_covariance = [arrays["covariance"][np.ix_(free, free)] for arrays in (ja, na)]
    hessian = ja["hessian"][np.ix_(free, free)]
    sigma, reference_sigma = np.sqrt(np.diag(covariance)), np.sqrt(np.diag(reference_covariance))
    sigma_outer, reference_outer = np.outer(sigma, sigma), np.outer(reference_sigma, reference_sigma)
    matrix_asymmetries = dict(
        hessian=float(np.max(abs(hessian-hessian.T)*sigma_outer, initial=0.)),
        covariance=float(np.max(abs(covariance-covariance.T)/sigma_outer, initial=0.)),
        reference_covariance=float(np.max(abs(reference_covariance-reference_covariance.T)/reference_outer, initial=0.)))
    positive = {}
    for label, matrix in (("hessian", hessian), ("covariance", covariance),
                          ("reference_covariance", reference_covariance)):
        try:
            np.linalg.cholesky(matrix)
            positive[label] = True
        except np.linalg.LinAlgError:
            positive[label] = False
    identity_error = float(np.max(abs(hessian @ covariance-np.eye(len(free))), initial=0.))
    values, ref_values = ja["values"][free], na["values"][free]
    coordinate_errors = abs(values-ref_values)
    coordinate_limits = np.maximum(.01*abs(ref_values), .01*reference_error)
    sigma_errors = abs(sigma-reference_error)
    covariance_sigma_errors = abs(sigma-reference_sigma)/reference_sigma
    nll_error = abs(float(jf["nll"])-float(native["canonical_fitted_nll"]))
    restoration_error = abs(native["canonical_fitted_nll"]-(
        native["canonical_initial_nll"]+nf["nll_zero_point"]))
    np.testing.assert_equal(restoration_error, native["canonical_restoration_error"])
    checks = dict(
        coordinates=bool(np.all(coordinate_errors <= coordinate_limits)),
        uncertainties=bool(np.all(sigma_errors <= .01*reference_error)),
        canonical_nll=bool(np.isfinite(nll_error) and nll_error < 1e-6),
        initial_nll=bool(0 <= jax["initial_combine_nll_absolute_error"] < 1e-6),
        native_offset_restoration=bool(restoration_error < 1e-6),
        full_ad_covariance_identity=bool(identity_error < 1e-6),
        matrix_symmetry=all(v < 1e-6 for v in matrix_asymmetries.values()),
        positive_curvature_and_covariances=all(positive.values()))
    settings = {
        key: {"jax": jax["optimizer"].get(key), "native": native["optimizer"].get(key)}
        for key in ("errordef", "tolerance", "strategy", "max_calls", "migrad_iterate", "use_simplex")
        if jax["optimizer"].get(key) != native["optimizer"].get(key)}
    result.update(
        status="passed" if all(checks.values()) else "failed", checks=checks,
        arrays_sha256=jax["arrays_sha256"], reference_arrays_sha256=native["arrays_sha256"],
        discrete_choices=jax["discrete_choices"], channel_masks=jax["channel_masks"],
        fixed_parameters=[names[i] for i in fixed], parameters=n, compared_free_parameters=len(free),
        at_bounds=dict(jax=jf["at_bounds"], native_has_parameters_at_limit=nf["has_parameters_at_limit"]),
        errors=dict(nll_absolute=nll_error, coordinate_max_absolute=float(coordinate_errors.max(initial=0.)),
                    coordinate_max_in_reference_sigma=float((coordinate_errors/reference_error).max(initial=0.)),
                    uncertainty_max_relative=float((sigma_errors/reference_error).max(initial=0.)),
                    covariance_diagonal_sigma_max_relative=float(covariance_sigma_errors.max(initial=0.)),
                    hessian_covariance_identity_max=identity_error, matrix_scaled_asymmetries=matrix_asymmetries),
        per_parameter=[dict(name=names[i], jax=float(values[k]), native=float(ref_values[k]),
            coordinate_absolute_error=float(coordinate_errors[k]), coordinate_limit=float(coordinate_limits[k]),
            coordinate_pass=bool(coordinate_errors[k] <= coordinate_limits[k]),
            jax_ad_sigma=float(sigma[k]), native_minuit_error=float(reference_error[k]),
            native_covariance_sigma=float(reference_sigma[k]), uncertainty_relative_error=float(sigma_errors[k]/reference_error[k]),
            uncertainty_pass=bool(sigma_errors[k] <= .01*reference_error[k]))
            for k, i in enumerate(free)],
        observed_timings=dict(
            scope="Recorded whole-run and stage times only; no speedup ratio. Different settings, starts, runtime versions, affinity and shared-node load must be considered.",
            same_host=True, same_initial_point=bool(np.array_equal(jstart, nstart)),
            optimizer_setting_differences=settings,
            native_hesse_strategy=native["optimizer"].get("hesse_strategy_used"),
            native_hesse_settings=native.get("hesse_settings"),
            jax_elapsed_seconds=jax["elapsed_seconds"], native_elapsed_seconds=native["elapsed_seconds"],
            jax_stages=jax["timings"], native_stages=native["timings"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve previous fit comparisons")
    result = compare(args.benchmark, args.reference)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output, result["status"])
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
