"""Audit a terminal native minimum and its full saved covariance, without ROOT/JAX."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from compare_exported_fit import checked_start, digest, load_arrays


def check_final(report, arrays):
    assert report["status"] == "reference_ready" and report["stage"] == "finished"
    final = report["final"]
    for key in ("is_valid", "has_accurate_covar", "has_posdef_covar"):
        assert final[key] is True, key
    for key in ("has_made_posdef_covar", "is_above_max_edm", "hesse_failed", "has_reached_call_limit"):
        assert final[key] is False, key
    assert np.isfinite(final["edm"]) and final["edm"] >= 0
    names = report["parameter_names"]
    assert len(set(names)) == len(names)
    fixed_names = report["fixed_bounds"]
    assert len(set(fixed_names)) == len(fixed_names) and set(fixed_names) <= set(names)
    fixed = np.array([i for i, name in enumerate(names) if name in fixed_names], dtype=int)
    free = np.array([i for i, name in enumerate(names) if name not in fixed_names], dtype=int)
    n = len(names)
    for key in ("values", "errors"):
        assert arrays[key].shape == (n,) and arrays[key].dtype == np.float64, key
        assert np.isfinite(arrays[key]).all(), key
        np.testing.assert_array_equal(arrays[key], final[key])
    np.testing.assert_array_equal(arrays["parameter_names"], names)
    initial = np.asarray(report["initial"], dtype=float)
    assert initial.shape == (n,) and np.isfinite(initial).all()
    np.testing.assert_array_equal(arrays["values"][fixed], initial[fixed])
    covariance = arrays["covariance"]
    assert covariance.shape == (n, n) and covariance.dtype == np.float64
    assert np.isfinite(covariance).all()
    assert np.all(covariance[fixed, :] == 0.) and np.all(covariance[:, fixed] == 0.)
    assert len(free) and np.all(arrays["errors"][free] > 0.)
    matrix = covariance[np.ix_(free, free)]
    assert np.all(np.diag(matrix) > 0.)
    sigma = np.sqrt(np.diag(matrix))
    normalized = matrix/np.outer(sigma, sigma)
    asymmetry = float(np.max(abs(normalized-normalized.T)))
    assert asymmetry < 1e-6, "Asymmetric native covariance"
    # Check every entry. Symmetrization is used only for this diagnostic.
    np.linalg.cholesky(.5*(normalized+normalized.T))
    restored = report["canonical_initial_nll"]+final["nll_zero_point"]
    error = abs(report["canonical_fitted_nll"]-restored)
    assert np.isfinite(error) and error < 1e-6
    np.testing.assert_equal(error, report["canonical_restoration_error"])
    shift = report["optimizer"]["constant_objective_shift"]
    assert np.isfinite(shift)
    np.testing.assert_equal(final["nll_zero_point"], final["optimizer_fval"]+shift)
    assert 0 <= report["optimizer_offset_restoration_error"] < 1e-6
    assert report["optimizer"]["errordef"] == .5 and report["optimizer"]["tolerance"] == 1e-4
    assert report["timings"]["hesse_seconds"] > 0, "Explicit full HESSE required"
    assert all(np.isfinite(value) and value >= 0 for value in report["timings"].values())
    return dict(parameters=n, free_parameters=len(free), fixed_parameters=fixed_names,
        fixed_covariance_rows_columns_zero=True, fixed_coordinates_unchanged=True,
        fixed_minuit_error_fields={names[i]: float(arrays["errors"][i]) for i in fixed},
        fixed_error_scope="Stored Minuit step/error fields; not uncertainties of fixed coordinates",
        full_free_covariance_positive_definite=True, normalized_covariance_asymmetry=asymmetry,
        native_error_vs_covariance_sigma_max_relative=float(
            np.max(abs(arrays["errors"][free]-sigma)/arrays["errors"][free])),
        uncertainty_scope="Native Minuit errors and covariance-diagonal sigmas are distinct; no agreement required here",
        canonical_restoration_absolute_error=error, edm=final["edm"],
        has_parameters_at_limit=final["has_parameters_at_limit"], nfcn=final["nfcn"])


def audit(path, manifest_path, driver_path=None):
    before = digest(path)
    report = json.loads(path.read_bytes())
    manifest_checksum = digest(manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][report["model"]]
    assert report["source"] == model["source"]
    assert report["combine_version"] == manifest["combine_version"] == "11.0.0"
    assert report["root_version"] == manifest["root_version"] == "6.40.00"
    assert report["runner_sha256"] == digest("scripts/fit_combine_reference.py")
    reference = model["reference"]
    names = reference.get("parameter_names", reference.get("parameters"))
    np.testing.assert_array_equal(report["parameter_names"], names)
    np.testing.assert_array_equal(report["initial"], reference["cases"][0]["theta"])
    assert abs(report["canonical_initial_nll"]-reference["cases"][0]["combine_nll"]) < 1e-6
    checked_start(report)
    arrays = load_arrays(report)
    checks = check_final(report, arrays)
    driver_checksum = None
    if driver_path is not None:
        driver_checksum = digest(driver_path)
        driver = json.loads(driver_path.read_bytes())
        assert driver["status"] == "producer_completed" and driver["exit_code"] == 0
        assert driver["producer_status"] == report["status"] == "reference_ready"
        assert driver["producer"] == str(path) and driver["producer_sha256"] == before
        assert driver["sources_unchanged"]
        assert all(digest(p) == h for p, h in driver["frozen_sources"].items())
        assert digest(driver["producer_log"]) == driver["log_sha256"]
        assert digest(driver_path) == driver_checksum
    assert digest(report["arrays"]) == report["arrays_sha256"]
    assert digest(path) == before and digest(manifest_path) == manifest_checksum
    return dict(status="passed", reference=str(path), reference_sha256=before,
        arrays=report["arrays"], arrays_sha256=report["arrays_sha256"],
        manifest=str(manifest_path), manifest_sha256=manifest_checksum,
        driver=str(driver_path) if driver_path else None, driver_sha256=driver_checksum,
        checker_sha256=digest(__file__), helper_sha256=digest("scripts/compare_exported_fit.py"),
        source=report["source"], combine_version=report["combine_version"],
        root_version=report["root_version"], host=report["host"], checks=checks,
        elapsed_seconds=report["elapsed_seconds"], timings=report["timings"],
        peak_rss_mib=report["peak_rss_mib"],
        scope="Terminal native state and complete saved covariance only. Canonical restoration is "
              "reconstructed from saved fields; a fresh native point check remains separate. "
              "No JAX minimum/covariance, profile, discrete-envelope or fit-speed acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("build/gpu_exports/manifest_random.json"))
    parser.add_argument("--driver", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use a fresh reports/*.json output")
    if args.output.exists():
        raise FileExistsError("Preserve earlier native audits")
    try:
        result = audit(args.reference, args.manifest, args.driver)
    except Exception:
        result = dict(status="failed", reference=str(args.reference),
            reference_sha256=digest(args.reference), checker_sha256=digest(__file__),
            exception=traceback.format_exc())
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(args.output, result["status"])
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
