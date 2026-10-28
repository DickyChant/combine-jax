"""Complete curvature, fixed coordinates, and immutable minimum prerequisites."""

import json
from pathlib import Path
import sys
import platform
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import complete_bounded_cpu_covariance as continuation


def configuration():
    likelihood = SimpleNamespace(parameter_names=("x",), discrete_choices={
        "pdf": dict(category="index", index=1, alternatives=("a", "b"), selected="b", correction=1.)},
        mask_configuration={"channel": dict(name="mask", class_name="RooRealVar", value=0., masked=False)})
    report = dict(parameter_names=["x"], discrete_choices=json.loads(json.dumps(likelihood.discrete_choices)),
                  channel_masks=likelihood.mask_configuration.copy())
    return likelihood, report


def test_json_tuple_representation_preserves_discrete_configuration():
    continuation.check_configuration(*configuration())


@pytest.mark.parametrize("change", ["category", "alternatives", "mask"])
def test_changed_category_alternative_or_mask_is_rejected(change):
    likelihood, report = configuration()
    report = json.loads(json.dumps(report))
    if change == "category":
        report["discrete_choices"]["pdf"]["index"] = 0
    elif change == "alternatives":
        report["discrete_choices"]["pdf"]["alternatives"] = ["a", "c"]
    else:
        report["channel_masks"]["channel"]["value"] = 1.
    with pytest.raises(AssertionError):
        continuation.check_configuration(likelihood, report)


def test_complete_inverse_retains_free_correlations_and_zeroes_fixed_rows():
    hessian = np.array([[4., 9., 1.], [9., -5., 2.], [1., 2., 3.]])
    original = hessian.copy()
    free = np.array([0, 2])
    covariance, info = continuation.full_covariance(hessian, free, np.array([.1, 5., 2.]))
    assert info["status"] == "positive_definite"
    np.testing.assert_allclose(covariance[np.ix_(free, free)], np.linalg.inv(hessian[np.ix_(free, free)]))
    assert covariance[0, 2] != 0
    np.testing.assert_array_equal(covariance[1, :], 0.)
    np.testing.assert_array_equal(covariance[:, 1], 0.)
    np.testing.assert_array_equal(hessian, original)
    assert info["hessian_covariance_identity_absolute_error"] < 1e-12


@pytest.mark.parametrize("matrix", [np.array([[1., 0.], [0., 0.]]),
                                   np.array([[1., 2.], [2., 1.]])])
def test_singular_or_negative_curvature_does_not_produce_pseudoinverse(matrix):
    covariance, info = continuation.full_covariance(matrix, np.arange(2), np.ones(2))
    assert covariance is None and info["status"] == "nonpositive_curvature"
    assert info["nonpositive_scaled_eigenvalues"] == 1


def test_nonfinite_fixed_row_is_still_rejected():
    matrix = np.array([[2., np.nan], [0., 3.]])
    covariance, info = continuation.full_covariance(matrix, np.array([0]), np.ones(2))
    assert covariance is None and info["status"] == "nonfinite_curvature"


def test_asymmetric_curvature_is_not_replaced_by_symmetric_approximation():
    covariance, info = continuation.full_covariance(np.array([[2., 0.], [.1, 3.]]),
                                                    np.arange(2), np.ones(2))
    assert covariance is None and info["status"] == "asymmetric_curvature"


def test_unstable_inverse_does_not_pass(monkeypatch):
    monkeypatch.setattr(np.linalg, "inv", lambda matrix: np.eye(len(matrix)))
    covariance, info = continuation.full_covariance(np.diag([2., 3.]), np.arange(2), np.ones(2))
    assert covariance is None and info["status"] == "unstable_inverse"
    assert info["hessian_covariance_identity_absolute_error"] > 1e-6


def test_all_fixed_covariance_is_zero():
    covariance, info = continuation.full_covariance(np.diag([-1., 0.]), np.array([], dtype=int), np.ones(2))
    assert info["status"] == "all_fixed"
    np.testing.assert_array_equal(covariance, np.zeros((2, 2)))


def test_parameter_scaling_does_not_change_physical_inverse():
    matrix = np.array([[2e8, 1.], [1., 3e-8]])
    covariance, info = continuation.full_covariance(matrix, np.arange(2), np.array([1e-4, 1e4]))
    assert info["status"] == "positive_definite"
    np.testing.assert_allclose(matrix@covariance, np.eye(2), atol=1e-8)


@pytest.fixture
def minimum_path(tmp_path, monkeypatch):
    """Build integrity fixtures without relying on ignored benchmark artifacts."""
    monkeypatch.chdir(tmp_path)
    Path("scripts").mkdir()
    Path("scripts/fit_bounded_cpu.py").write_text("# integrity fixture\n")
    Path("src/combine_jax").mkdir(parents=True)
    Path("src/combine_jax/model.py").write_text("# integrity fixture\n")
    artifact = Path("minimum.npz")
    np.savez_compressed(artifact, parameter_names=np.array(["x"]), initial=np.array([0.]),
        values=np.array([1.]), gradient=np.array([0.]), hessian=np.empty((0, 0)), covariance=np.empty((0, 0)))
    validation = Path("validation.npz")
    np.savez_compressed(validation, validation_gradients=np.zeros((26, 1)))
    checkpoint = dict(status="passed", source={"fixture": True}, source_sha256=continuation.core_digest(),
        export_sha256="fixture-export", gradient_compiler_options=continuation.OPTIONS,
        arrays=str(validation), arrays_sha256=continuation.digest(validation),
        reference_validation=[dict(nll_absolute_error=0., value_gradient_nll_absolute_error=0.,
                                   gradient_max_relative_error=0.) for _ in range(26)])
    checkpoint_path = Path("checkpoint.json")
    checkpoint_path.write_text(json.dumps(checkpoint))
    report = dict(status="minimum_computed", stage="finished", model="counting", backend="cpu",
        host=platform.node(), float64=True, root_imported=False, memory_policy="resident_value_spooled_gradient",
        evaluation_counters=dict(closed=True, prepared=True),
        runner_sha256=continuation.digest("scripts/fit_bounded_cpu.py"),
        source=checkpoint["source"], source_sha256=continuation.core_digest(),
        export_sha256=checkpoint["export_sha256"],
        frozen_source_files={"scripts/fit_bounded_cpu.py": continuation.digest("scripts/fit_bounded_cpu.py")},
        final=dict(valid_minimum=True, covariance_status="not_requested", values=[1.], nll=0.,
                   edm=1e-10, objective_offset=1., shifted_nll=-1.),
        optimizer=dict(errordef=.5), initial_combine_nll_absolute_error=0.,
        arrays=str(artifact), arrays_sha256=continuation.digest(artifact),
        parameter_names=["x"], initial=[0.], gradient_checkpoint=str(checkpoint_path),
        gradient_checkpoint_sha256=continuation.digest(checkpoint_path), gradient_compiler_options=continuation.OPTIONS)
    path = Path("minimum.json")
    path.write_text(json.dumps(report))
    return path


def test_saved_minimum_gate_rechecks_numerical_artifacts(minimum_path):
    report, arrays = continuation.checked_minimum(minimum_path)
    assert report["final"]["valid_minimum"] and arrays["hessian"].shape == (0, 0)
    np.testing.assert_array_equal(arrays["values"], report["final"]["values"])


@pytest.mark.parametrize("keys,value", [
    (("status",), "running"),
    (("evaluation_counters", "closed"), False),
    (("final", "valid_minimum"), False),
    (("final", "nll"), 100.),
    (("gradient_checkpoint_sha256",), "changed"),
    (("arrays_sha256",), "changed"),
])
def test_incomplete_or_changed_minimum_cannot_start_curvature(tmp_path, minimum_path, keys, value):
    report = json.loads(minimum_path.read_bytes())
    target = report
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    path = tmp_path/"changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        continuation.checked_minimum(path)


def test_existing_output_prevents_initialization(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    output = Path("reports/old.json")
    output.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["complete_bounded_cpu_covariance.py",
        "--minimum-report", "missing.json", "--output", str(output)])
    with pytest.raises(FileExistsError):
        continuation.main()
    assert output.read_text() == "preserve"
