"""Native acceptance needs a full covariance and retains fixed error fields."""

from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_native_fit import audit, check_final, digest


@pytest.fixture
def native():
    final = dict(is_valid=True, has_accurate_covar=True, has_posdef_covar=True,
        has_made_posdef_covar=False, is_above_max_edm=False, hesse_failed=False,
        has_reached_call_limit=False, edm=1e-9, nfcn=100, has_parameters_at_limit=True,
        values=[1., 2., 3.], errors=[1., 2., 10.], nll_zero_point=-2., optimizer_fval=0.)
    report = dict(status="reference_ready", stage="finished", final=final,
        parameter_names=["x", "y", "fixed"], fixed_bounds=["fixed"], initial=[0., 0., 3.],
        canonical_initial_nll=100., canonical_fitted_nll=98., canonical_restoration_error=0.,
        optimizer_offset_restoration_error=0.,
        optimizer=dict(constant_objective_shift=-2., errordef=.5, tolerance=1e-4),
        timings=dict(setup_seconds=1., migrad_seconds=2., hesse_seconds=3.))
    arrays = dict(parameter_names=np.array(report["parameter_names"]),
        values=np.array(final["values"]), errors=np.array(final["errors"]),
        covariance=np.diag([1., 4., 0.]))
    return report, arrays


def test_complete_covariance_allows_nonzero_fixed_error_field(native):
    result = check_final(*native)
    assert result["full_free_covariance_positive_definite"]
    assert result["free_parameters"] == 2
    assert result["fixed_minuit_error_fields"] == {"fixed": 10.}


@pytest.mark.parametrize("key,value", [("status", "fit_incomplete"), ("stage", "hesse")])
def test_only_terminal_accepted_native_state_is_allowed(native, key, value):
    native[0][key] = value
    with pytest.raises(AssertionError):
        check_final(*native)


@pytest.mark.parametrize("key", ["is_valid", "has_accurate_covar", "has_posdef_covar",
    "has_made_posdef_covar", "is_above_max_edm", "hesse_failed", "has_reached_call_limit"])
def test_invalid_final_flags_are_rejected(native, key):
    native[0]["final"][key] = not native[0]["final"][key]
    with pytest.raises(AssertionError):
        check_final(*native)


@pytest.mark.parametrize("kind", ["nonfinite", "missing_fixed", "fixed_row", "fixed_column",
                                "asymmetric", "indefinite", "float32"])
def test_malformed_or_unphysical_covariance_is_rejected(native, kind):
    report, arrays = native
    matrix = arrays["covariance"]
    if kind == "nonfinite":
        matrix[2, 2] = np.nan
    elif kind == "missing_fixed":
        arrays["covariance"] = matrix[:2, :2]
    elif kind == "fixed_row":
        matrix[2, 0] = .1
    elif kind == "fixed_column":
        matrix[0, 2] = .1
    elif kind == "asymmetric":
        matrix[0, 1] = .1
    elif kind == "indefinite":
        matrix[0, 1] = matrix[1, 0] = 3.
    elif kind == "float32":
        arrays["covariance"] = matrix.astype(np.float32)
    with pytest.raises((AssertionError, np.linalg.LinAlgError)):
        check_final(report, arrays)


def test_fixed_coordinate_must_remain_fixed(native):
    native[0]["final"]["values"][2] = 4.
    native[1]["values"][2] = 4.
    with pytest.raises(AssertionError):
        check_final(*native)


def test_migrad_only_checkpoint_is_not_a_full_hesse(native):
    del native[0]["timings"]["hesse_seconds"]
    with pytest.raises(KeyError):
        check_final(*native)


def test_offset_restoration_cannot_hide_shift(native):
    native[0]["optimizer"]["constant_objective_shift"] = -1.
    with pytest.raises(AssertionError):
        check_final(*native)


def test_error_and_covariance_sigma_are_distinct(native):
    native[0]["final"]["errors"][1] = native[1]["errors"][1] = 2.1
    result = check_final(*native)
    assert result["native_error_vs_covariance_sigma_max_relative"] > .01


@pytest.fixture
def stored_native(native, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for directory in ("reports", "build", "scripts"):
        Path(directory).mkdir()
    runner = Path("scripts/fit_combine_reference.py")
    runner.write_text("fixture runner\n")
    frozen_helper = Path("scripts/hhvv_setup.py")
    frozen_helper.write_text("fixture configuration\n")
    Path("scripts/compare_exported_fit.py").write_text("fixture helper\n")
    report, arrays = native
    artifact = Path("build/native.npz")
    np.savez_compressed(artifact, **arrays)
    source = dict(sha256="source fixture")
    report.update(model="counting", source=source, combine_version="11.0.0",
        root_version="6.40.00", runner_sha256=digest(runner), host="fixture-host",
        elapsed_seconds=7., peak_rss_mib=100., arrays=str(artifact),
        arrays_sha256=digest(artifact))
    path = Path("reports/native.json")
    path.write_text(json.dumps(report))
    manifest_path = Path("build/manifest.json")
    manifest_path.write_text(json.dumps(dict(combine_version="11.0.0", root_version="6.40.00",
        models=dict(counting=dict(source=source, reference=dict(parameters=report["parameter_names"],
            cases=[dict(theta=report["initial"], combine_nll=report["canonical_initial_nll"])]))))))
    log = Path("reports/native.log")
    log.write_text("fixture finished\n")
    driver_path = Path("reports/driver.json")
    driver_path.write_text(json.dumps(dict(status="producer_completed", exit_code=0,
        producer_status="reference_ready", producer=str(path), producer_sha256=digest(path),
        sources_unchanged=True, frozen_sources={
            str(runner): digest(runner), str(frozen_helper): digest(frozen_helper)},
        producer_log=str(log), log_sha256=digest(log))))
    return path, manifest_path, driver_path


def test_hash_bound_terminal_driver_and_artifact_pass(stored_native):
    result = audit(*stored_native)
    assert result["status"] == "passed" and result["driver_sha256"] == digest(stored_native[2])


@pytest.mark.parametrize("key,value", [
    ("status", "running"), ("exit_code", 86), ("producer_status", "fit_incomplete"),
    ("producer_sha256", "wrong"), ("sources_unchanged", False), ("log_sha256", "wrong")])
def test_unfinished_or_mismatched_driver_is_rejected(stored_native, key, value):
    path = stored_native[2]
    driver = json.loads(path.read_text())
    driver[key] = value
    path.write_text(json.dumps(driver))
    with pytest.raises(AssertionError):
        audit(*stored_native)


def test_changed_native_arrays_are_rejected(stored_native):
    Path("build/native.npz").write_bytes(b"corrupt")
    with pytest.raises(AssertionError, match="arrays_sha256"):
        audit(*stored_native)


def test_changed_frozen_native_source_is_rejected(stored_native):
    Path("scripts/hhvv_setup.py").write_text("changed configuration\n")
    with pytest.raises(AssertionError):
        audit(*stored_native)


def test_manifest_source_must_match_native(stored_native):
    path = stored_native[1]
    manifest = json.loads(path.read_text())
    manifest["models"]["counting"]["source"]["sha256"] = "changed source"
    path.write_text(json.dumps(manifest))
    with pytest.raises(AssertionError):
        audit(*stored_native)
