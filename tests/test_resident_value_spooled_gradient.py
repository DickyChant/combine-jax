"""Resident NLLs survive gradient loading; function-only calls need no gradient."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from resident_value_spooled_gradient import ResidentValueSpooledGradient, fit_separate_calls
from sharded_cpu_evaluator import NO_FUSION_OPTIONS


@pytest.fixture(scope="module")
def template_path(tmp_path_factory):
    from build_template_fixture import build_template_fixture
    return build_template_fixture(tmp_path_factory.mktemp("resident_template"))


@pytest.mark.parametrize("fixture", ["counting_path", "template_path"])
def test_prepared_batch_and_live_calls_preserve_every_term(request, fixture, tmp_path):
    from combine_jax.likelihood import from_workspace
    from combine_jax.root_io import open_workspace

    with open_workspace(request.getfixturevalue(fixture)) as workspace:
        likelihood = from_workspace(workspace)
    likelihood = likelihood.at_zero(likelihood.initial)
    shift = np.array([.1*p.error for p in likelihood.parameters])
    point = np.clip(likelihood.initial+shift,
                    [p.minimum for p in likelihood.parameters], [p.maximum for p in likelihood.parameters])
    points = np.asarray([likelihood.initial, point])
    events = []
    with ResidentValueSpooledGradient(likelihood, tmp_path/"executables",
            gradient_compiler_options=NO_FUSION_OPTIONS, on_progress=events.append) as evaluator:
        with pytest.raises(RuntimeError):
            evaluator.value(point)
        result = evaluator.prepare(points)
        assert events and len(evaluator.gradient_preparation) == len(evaluator.local.terms)
        assert all(r["validation_points"] == 2 and r["nominal_reload_bitwise_equal"]
                   for r in evaluator.gradient_preparation)
        for index, point in enumerate(points):
            expected_value, expected_gradient = likelihood.value_and_grad(point)
            np.testing.assert_allclose(result["validation_values"][index], expected_value, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(result["validation_vg_values"][index], expected_value, atol=1e-10, rtol=1e-10)
            np.testing.assert_allclose(result["validation_gradients"][index], expected_gradient, atol=1e-10, rtol=1e-10)
            before = [len(t.value_and_grad.calls) for t in evaluator.local.terms]
            np.testing.assert_allclose(evaluator.value(point), expected_value, atol=1e-10, rtol=1e-10)
            assert before == [len(t.value_and_grad.calls) for t in evaluator.local.terms]
            value, gradient = evaluator.value_and_grad(point)
            np.testing.assert_array_equal(value, result["validation_vg_values"][index])
            np.testing.assert_array_equal(gradient, result["validation_gradients"][index])
        with pytest.raises(RuntimeError, match="once"):
            evaluator.prepare(points)
    assert evaluator.closed
    with pytest.raises(RuntimeError):
        evaluator.value_and_grad(point)


def test_minimum_matches_combined_callback_without_gradients_on_every_function_call(counting_path, tmp_path):
    from combine_jax.fitting import fit
    from combine_jax.likelihood import from_workspace
    from combine_jax.root_io import open_workspace

    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    expected = fit(likelihood, covariance=False, strategy=1, max_calls=10000)
    with ResidentValueSpooledGradient(likelihood, tmp_path/"executables",
                                     gradient_compiler_options=NO_FUSION_OPTIONS) as evaluator:
        prepared = evaluator.prepare([likelihood.initial])
        result = fit_separate_calls(likelihood, evaluator, strategy=1, max_calls=10000,
                                    expected_initial_nll=float(prepared["validation_values"][0]))
        assert result["valid_minimum"] and result["covariance_status"] == "not_requested"
        np.testing.assert_allclose(result["values"], expected.values, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(result["nll"], expected.nll, atol=1e-8, rtol=0)
        assert evaluator.gradient_calls < evaluator.value_calls
        assert evaluator.gradient_calls <= result["gradient_calls"]+2
        assert result["maximum_resident_streamed_nll_difference"] < 1e-6
        assert result["nonfinite_value_calls"] == result["nonfinite_gradient_calls"] == 0
        assert result["objective_offset"] == result["initial_nll"]
        assert result["canonical_restoration_absolute_error"] < 1e-10


@pytest.mark.parametrize("policy", [{}, {"migrad_iterate": 1, "use_simplex": False}])
def test_large_canonical_nll_is_centered_and_restored(policy):
    likelihood = SimpleNamespace(parameter_names=("x",), initial=np.array([0.]),
        parameters=[SimpleNamespace(name="x", value=0., minimum=-5., maximum=5., error=1.)])
    gradients = []

    def value(point):
        return 1e7+(point[0]-1.)**2

    def value_and_grad(point):
        derivative = np.array([2.*(point[0]-1.)])
        gradients.append(derivative.copy())
        return value(point), derivative

    evaluator = SimpleNamespace(parameter_names=likelihood.parameter_names, value=value,
                                value_and_grad=value_and_grad)
    result = fit_separate_calls(likelihood, evaluator, expected_initial_nll=10000001.,
                               strategy=1, max_calls=1000, tolerance=1e-6, **policy)
    assert result["valid_minimum"]
    assert result["objective_offset"] == 10000001.
    np.testing.assert_allclose(result["values"], [1.], atol=1e-4, rtol=0)
    np.testing.assert_allclose(result["nll"], 1e7, atol=1e-7, rtol=0)
    np.testing.assert_allclose(result["shifted_nll"], -1., atol=1e-7, rtol=0)
    assert result["canonical_restoration_absolute_error"] < 1e-8
    np.testing.assert_array_equal(gradients[0], [-2.])
    np.testing.assert_allclose(result["gradient"], 2.*(result["values"]-1.), atol=0, rtol=0)


@pytest.mark.parametrize("iterations", [0, -1, 1.5])
def test_invalid_retry_count_is_rejected_before_evaluation(iterations):
    with pytest.raises(ValueError, match="positive integer iterations"):
        fit_separate_calls(None, None, migrad_iterate=iterations)


def test_existing_executables_are_not_overwritten(counting_path, tmp_path):
    from combine_jax.likelihood import from_workspace
    from combine_jax.root_io import open_workspace

    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    directory = tmp_path/"preserved"
    directory.mkdir()
    with pytest.raises(FileExistsError):
        ResidentValueSpooledGradient(likelihood, directory)
