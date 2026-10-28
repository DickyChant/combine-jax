"""Fix only the selected coordinate and keep the complete validated objective."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from combine_jax.graph import Parameter
from bounded_cpu_profile import fit_point, fixed_problem, verify_reference_batch
import run_profile_points as runner


def problem():
    likelihood = SimpleNamespace(parameter_names=("nuisance", "poi", "fixed"), initial=np.array([0., 0., 4.]),
        parameters=tuple(Parameter(name, value, low, high, 1., "nuisance") for name, value, low, high in (
            ("nuisance", 0., -3., 3.), ("poi", 0., -10., 10.), ("fixed", 4., 4., 4.))))
    plan = dict(parameter_names=list(likelihood.parameter_names), initial=[0., 0., 4.],
                parameter="nuisance", parameter_index=0)
    return likelihood, plan


def test_fixed_optimizer_view_does_not_mutate_the_likelihood_or_common_start():
    likelihood, plan = problem()
    view, initial = fixed_problem(likelihood, plan, .7)
    assert view.parameters[0].minimum == view.parameters[0].maximum == .7
    assert likelihood.parameters[0].minimum == -3. and likelihood.parameters[0].maximum == 3.
    np.testing.assert_array_equal(initial, [.7, 0., 4.])
    np.testing.assert_array_equal(plan["initial"], [0., 0., 4.])
    np.testing.assert_array_equal(likelihood.initial, [0., 0., 4.])
    assert view.parameters[1:] == likelihood.parameters[1:]


@pytest.mark.parametrize("held", [-.8, .7])
@pytest.mark.parametrize("iterate,simplex", [(5, True), (1, False)])
def test_correlated_profile_reoptimizes_poi_and_retains_original_fixed_coordinate(
        monkeypatch, held, iterate, simplex):
    from iminuit import Minuit
    original_migrad = Minuit.migrad
    recorded = []

    def migrad(self, *args, **kwargs):
        recorded.append(kwargs)
        return original_migrad(self, *args, **kwargs)

    monkeypatch.setattr(Minuit, "migrad", migrad)
    likelihood, plan = problem()
    calls = dict(value=0, gradient=0)

    def value(point):
        calls["value"] += 1
        x, y, z = point
        return .5*x*x+.5*(y-2*x)**2+.5*(z-4.)**2

    def value_gradient(point):
        calls["gradient"] += 1
        x, y, z = point
        return value(point), np.array([x-2*(y-2*x), y-2*x, z-4.])

    evaluator = SimpleNamespace(parameter_names=likelihood.parameter_names,
                               value=value, value_and_grad=value_gradient)
    args = SimpleNamespace(max_calls=1000, strategy=1, tolerance=1e-6,
                           migrad_iterate=iterate, no_simplex=not simplex)
    values, nll, record = fit_point(likelihood, evaluator, plan, held, args)
    assert record["valid_minimum"] and record["fixed"] == {"nuisance": held, "fixed": 4.}
    assert values[0] == held and values[2] == 4.
    np.testing.assert_allclose(values[1], 2*held, atol=1e-4, rtol=0)
    np.testing.assert_allclose(nll, .5*held**2, atol=1e-8, rtol=0)
    assert record["canonical_restoration_absolute_error"] < 1e-10
    assert calls["gradient"] < calls["value"]
    assert recorded == [dict(ncall=1000, iterate=iterate, use_simplex=simplex)]


@pytest.mark.parametrize("iterate,simplex", [(5, True), (1, False)])
def test_native_profile_forwards_retry_policy_and_restores_baseline(monkeypatch, iterate, simplex):
    from iminuit import Minuit
    original_migrad = Minuit.migrad
    recorded = []

    def migrad(self, *args, **kwargs):
        recorded.append(kwargs)
        return original_migrad(self, *args, **kwargs)

    monkeypatch.setattr(Minuit, "migrad", migrad)
    likelihood, plan = problem()
    state = dict(point=np.asarray(plan["initial"]), offset=50.)

    def get_value():
        x, y, z = state["point"]
        return 50.+.5*x*x+.5*(y-2*x)**2+.5*(z-4.)**2-state["offset"]

    def evaluate(point):
        state["point"] = np.asarray(point)
        return get_value()

    reference = SimpleNamespace(getVal=get_value,
        clearZeroPoint=lambda: state.update(offset=0.),
        setZeroPoint=lambda: state.update(offset=get_value()+state["offset"]))
    root = SimpleNamespace(RooAbsReal=SimpleNamespace(
        clearEvalErrorLog=lambda: None, numEvalErrors=lambda: 0))
    args = SimpleNamespace(max_calls=1000, strategy=1, tolerance=1e-6,
                           migrad_iterate=iterate, no_simplex=not simplex)
    values, nll, record = runner.fit_native(root, reference,
        SimpleNamespace(evaluate=evaluate), 50., likelihood, plan, .7, args)
    assert recorded == [dict(ncall=1000, iterate=iterate, use_simplex=simplex)]
    assert record["valid_minimum"] and record["offset_restoration_error"] < 1e-10
    np.testing.assert_allclose(values, [.7, 1.4, 4.], atol=1e-4, rtol=0)
    np.testing.assert_allclose(nll, 50.+.5*.7**2, atol=1e-8, rtol=0)
    np.testing.assert_array_equal(state["point"], plan["initial"])
    assert state["offset"] == 50. and get_value() == 0.


@pytest.mark.parametrize("value", [-3.1, 3.1, np.nan, np.inf])
def test_out_of_range_or_nonfinite_profile_values_are_rejected(value):
    with pytest.raises(ValueError, match="outside"):
        fixed_problem(*problem(), value)


def batch():
    reference = dict(parameter_names=np.array(["a", "b"]), validation_values=np.zeros(26),
                     validation_vg_values=np.zeros(26), validation_gradients=np.zeros((26, 2)))
    actual = {key: value.copy() for key, value in reference.items() if key != "parameter_names"}
    return actual, reference


def test_complete_fresh_reference_batch_passes():
    verify_reference_batch(*batch())


@pytest.mark.parametrize("change", ["gradient", "nll", "missing", "float32", "nan"])
def test_changed_or_incomplete_reference_batch_cannot_start_profile(change):
    actual, reference = batch()
    if change == "gradient":
        actual["validation_gradients"][25, 1] = 1e-14
    elif change == "nll":
        actual["validation_values"][0] = 1e-14
    elif change == "missing":
        actual.pop("validation_gradients")
    elif change == "float32":
        actual["validation_gradients"] = actual["validation_gradients"].astype(np.float32)
    else:
        actual["validation_gradients"][0, 0] = np.nan
    with pytest.raises(AssertionError):
        verify_reference_batch(actual, reference)


@pytest.mark.parametrize("engine", ["combine", "gpu"])
def test_bounded_mode_is_cpu_only_before_any_runtime_initialization(monkeypatch, engine):
    monkeypatch.setattr(sys, "argv", ["run_profile_points.py", "--plan", "missing",
        "--engine", engine, "--output", "reports/missing.json", "--bounded-minimum", "missing"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


@pytest.mark.parametrize("engine,options", [
    ("combine", ["--migrad-iterate", "0"]),
    ("combine", ["--migrad-iterate", "-1"]),
    ("cpu", ["--migrad-iterate", "1"]),
    ("gpu", ["--no-simplex"]),
])
def test_invalid_or_unsupported_retry_controls_fail_before_runtime(monkeypatch, engine, options):
    monkeypatch.setattr(sys, "argv", ["run_profile_points.py", "--plan", "missing",
        "--engine", engine, "--output", "reports/missing.json", *options])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2
