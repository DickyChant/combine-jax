"""Saved fit acceptance must follow convergence, covariance and source evidence."""

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import compare_exported_fit as checker


@pytest.fixture
def evidence(tmp_path):
    names = ["large", "near_zero", "fixed"]
    initial, values = [0., 0., 7.], [10., 0., 7.]
    common = dict(model="synthetic", source={"sha256": "synthetic-only"}, host="same-host",
                  parameter_names=names, initial=initial, discrete_choices={"choice": 1},
                  channel_masks={"mask": 0.}, fixed_bounds=["fixed"],
                  optimizer=dict(errordef=.5, tolerance=1e-4, strategy=1, max_calls=100,
                                 migrad_iterate=5, use_simplex=True),
                  cpu_affinity=[0], iminuit_version="synthetic", runner_sha256="synthetic",
                  timings={"synthetic": 1.}, elapsed_seconds=2.)
    jax = copy.deepcopy(common)
    jax.update(status="fit_computed", float64=True, root_imported=False,
               export_stack=dict(combine_version="11.0.0", root_version="6.40.00"),
               backend="cpu", devices=["cpu:0"], jax_version="synthetic", jaxlib_version="synthetic",
               source_sha256="synthetic", curvature_point=values,
               initial_combine_nll_absolute_error=0.,
               final=dict(values=values, nll=12., valid_minimum=True,
                          covariance_status="positive_definite", fixed={"fixed": 7.}, at_bounds=["fixed"]))
    native = copy.deepcopy(common)
    native.update(status="reference_ready", combine_version="11.0.0", root_version="6.40.00",
                  canonical_initial_nll=20., canonical_fitted_nll=12., canonical_restoration_error=0.,
                  final=dict(values=values, errors=[2., 1., .1], is_valid=True,
                             has_accurate_covar=True, has_posdef_covar=True, has_made_posdef_covar=False,
                             is_above_max_edm=False, hesse_failed=False, has_reached_call_limit=False,
                             has_parameters_at_limit=False, nll_zero_point=-8.))
    ja = dict(parameter_names=np.asarray(names), initial=np.asarray(initial), values=np.asarray(values),
              gradient=np.zeros(3), hessian=np.diag([.25, 1., 0.]), covariance=np.diag([4., 1., 0.]))
    na = dict(parameter_names=np.asarray(names), values=np.asarray(values),
              errors=np.asarray(native["final"]["errors"]), covariance=np.diag([4., 1., 0.]))

    def save():
        reports = []
        for name, report, arrays in (("jax", jax, ja), ("native", native, na)):
            artifact = tmp_path/(name+".npz")
            np.savez_compressed(artifact, **arrays)
            report.update(arrays=str(artifact), arrays_sha256=checker.digest(artifact))
            path = tmp_path/(name+".json")
            path.write_text(json.dumps(report))
            reports.append(path)
        return reports
    return jax, native, ja, na, save


def test_valid_full_covariance_and_fixed_coordinate(evidence):
    *_, save = evidence
    result = checker.compare(*save())
    assert result["status"] == "passed" and all(result["checks"].values())
    assert result["fixed_parameters"] == ["fixed"] and result["compared_free_parameters"] == 2
    assert result["errors"]["hessian_covariance_identity_max"] == 0.
    assert [p["coordinate_limit"] for p in result["per_parameter"]] == [.1, .01]
    assert result["observed_timings"]["same_initial_point"]
    assert not result["observed_timings"]["optimizer_setting_differences"]


@pytest.mark.parametrize("field", ["source", "host", "parameter_names", "discrete_choices", "channel_masks"])
def test_reject_incompatible_fit_definition(evidence, field):
    jax, native, ja, na, save = evidence
    native[field] = "incompatible"
    with pytest.raises(AssertionError, match=field):
        checker.compare(*save())


@pytest.mark.parametrize("which,field,value", [
    ("jax", "valid_minimum", False), ("jax", "covariance_status", "nonpositive_curvature"),
    ("native", "is_valid", False), ("native", "has_accurate_covar", False),
    ("native", "has_made_posdef_covar", True), ("native", "has_reached_call_limit", True)])
def test_unfinished_or_inaccurate_fit_never_passes(evidence, which, field, value):
    jax, native, ja, na, save = evidence
    report = jax if which == "jax" else native
    report["final"][field] = value
    paths = save()
    Path(jax["arrays"]).unlink()
    Path(native["arrays"]).unlink()
    result = checker.compare(*paths)
    assert result["status"] == "incomplete" and result["unresolved"]
    assert "observed_timings" not in result


@pytest.mark.parametrize("fault,failed_check", [
    ("coordinate_large", "coordinates"), ("coordinate_zero", "coordinates"),
    ("sigma", "uncertainties"), ("inverse", "full_ad_covariance_identity"),
    ("nonpositive", "positive_curvature_and_covariances"),
    ("asymmetric", "matrix_symmetry"), ("nll", "canonical_nll")])
def test_numerical_disagreements_are_failed_reports(evidence, fault, failed_check):
    jax, native, ja, na, save = evidence
    if fault.startswith("coordinate"):
        i = 0 if fault == "coordinate_large" else 1
        ja["values"][i] += .11 if i == 0 else .011
        jax["final"]["values"] = jax["curvature_point"] = ja["values"].tolist()
    elif fault == "sigma":
        ja["covariance"][0, 0] = 4.41
        ja["hessian"][0, 0] = 1/4.41
    elif fault == "inverse":
        ja["hessian"][0, 0] = .3
    elif fault == "nonpositive":
        ja["hessian"][0, 0] = -.25
    elif fault == "asymmetric":
        ja["hessian"][0, 1] = .1
    elif fault == "nll":
        jax["final"]["nll"] += 1e-4
    result = checker.compare(*save())
    assert result["status"] == "failed" and not result["checks"][failed_check]


@pytest.mark.parametrize("fault", ["checksum", "array_order", "curvature_point", "fixed_covariance", "offset"])
def test_reject_inconsistent_saved_evidence(evidence, fault):
    jax, native, ja, na, save = evidence
    if fault == "array_order":
        ja["parameter_names"] = ja["parameter_names"][::-1]
    elif fault == "curvature_point":
        jax["curvature_point"] = [11., 0., 7.]
    elif fault == "fixed_covariance":
        ja["covariance"][2, 2] = 1.
    elif fault == "offset":
        native["canonical_restoration_error"] = 1.
    paths = save()
    if fault == "checksum":
        Path(jax["arrays"]).write_bytes(b"tampered")
    with pytest.raises(AssertionError):
        checker.compare(*paths)


def test_minuit_error_and_covariance_diagonal_remain_distinct(evidence):
    jax, native, ja, na, save = evidence
    na["errors"][0] = 2.04
    native["final"]["errors"] = na["errors"].tolist()
    result = checker.compare(*save())
    assert result["status"] == "failed" and not result["checks"]["uncertainties"]
    assert result["errors"]["covariance_diagonal_sigma_max_relative"] == 0.
    assert result["errors"]["uncertainty_max_relative"] > .01


def test_different_settings_and_starts_do_not_become_speed_claim(evidence):
    jax, native, ja, na, save = evidence
    jax["optimizer"]["strategy"] = 0
    jax["initial"] = [.1, 0., 7.]
    ja["initial"] = np.asarray(jax["initial"])
    result = checker.compare(*save())
    assert result["status"] == "passed"  # Accepted numerical minima remain comparable.
    timings = result["observed_timings"]
    assert not timings["same_initial_point"]
    assert timings["optimizer_setting_differences"] == {"strategy": {"jax": 0, "native": 1}}
    assert not any("ratio" in key for key in timings)


def test_unverified_restart_rejected(evidence, tmp_path):
    jax, native, ja, na, save = evidence
    path = tmp_path/"check.json"
    path.write_text('{"status":"not_checked"}')
    jax.update(initial_check=str(path), initial_check_sha256=checker.digest(path))
    with pytest.raises(AssertionError):
        checker.compare(*save())
