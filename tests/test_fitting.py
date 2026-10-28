from iminuit import Minuit
import numpy as np
import pytest

from combine_jax.fitting import fit, profile
from combine_jax.likelihood import from_workspace
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.graph import Parameter


@pytest.mark.parametrize("memory_policy", ("retain","release"))
def test_counting_fit_covariance_and_profile_against_combine(counting_path,memory_policy):
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
        evaluator = likelihood.local_evaluator()
        result = fit(likelihood, evaluator=evaluator, tolerance=1e-6,memory_policy=memory_policy)
        assert result.valid_minimum
        assert result.covariance_status == "positive_definite"
        reference = root.cacheutils.CachingSimNLL(workspace.pdf("model_s"), workspace.data("data_obs"),
                                                workspace.obj("ModelConfig").GetNuisanceParameters())
        reference.clearConstantZeroPoint(); reference.clearZeroPoint()
        variables = [workspace.var(n) for n in likelihood.parameter_names]
        def objective(*point):
            for variable, value in zip(variables, point, strict=True): variable.setVal(float(value))
            return float(reference.getVal())
        minuit = Minuit(objective, *likelihood.initial, name=likelihood.parameter_names)
        minuit.errordef = .5; minuit.tol = 1e-6
        for p in likelihood.parameters:
            minuit.limits[p.name] = (p.minimum, p.maximum)
            minuit.errors[p.name] = p.error if p.error > 0 else .1
        minuit.migrad(); minuit.hesse()
        assert minuit.valid
        np.testing.assert_allclose(result.values, np.array(minuit.values), atol=1e-4, rtol=1e-4)
        np.testing.assert_allclose(result.nll, minuit.fval, atol=1e-8)
        np.testing.assert_allclose(result.covariance, np.array(minuit.covariance), atol=1e-4, rtol=1e-3)
        index = likelihood.parameter_names.index("bkg")
        endpoints = result.values[index]+np.array([-1., 1.])*result.errors[index]
        _, points = profile(likelihood, "bkg", endpoints, best_fit=result, evaluator=evaluator,
                            tolerance=1e-6,memory_policy=memory_policy)
        for value, delta, fitted in points:
            assert fitted.valid_minimum and delta > 0
            minuit.values = result.values
            minuit.values["bkg"] = value; minuit.fixed["bkg"] = True
            minuit.migrad()
            assert minuit.valid
            np.testing.assert_allclose(fitted.nll, minuit.fval, atol=1e-8)
            np.testing.assert_allclose(fitted.values, np.array(minuit.values), atol=1e-4, rtol=1e-4)
            assert np.all(fitted.covariance[index] == 0.)


def test_fit_fixed_coordinates_and_input_guards(counting_path):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    result = fit(likelihood, fixed=dict(zip(likelihood.parameter_names, likelihood.initial, strict=True)))
    assert result.valid_minimum and result.covariance_status == "all_fixed"
    np.testing.assert_array_equal(result.errors, np.zeros_like(likelihood.initial))
    with pytest.raises(ValueError, match="Unknown fixed"):
        fit(likelihood, fixed={"missing": 1.})
    with pytest.raises(ValueError,match="memory_policy"):
        fit(likelihood,memory_policy="unknown")
    with pytest.raises(ValueError, match="outside the range"):
        fit(likelihood, fixed={"r": -1000.})
    with pytest.raises(ValueError, match="fixed in the baseline"):
        profile(likelihood, "bkg", [.2], best_fit=result)


def test_profile_preserves_other_fixed_parameters(counting_path):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    baseline, points = profile(likelihood, "bkg", [-.5, .5], fixed={"lumi": .2}, covariance=False)
    assert baseline.valid_minimum
    for _, _, result in points:
        assert result.valid_minimum
        assert result.values[likelihood.parameter_names.index("lumi")] == .2
        assert result.covariance_status == "not_requested"


@pytest.mark.parametrize("curvature", [0., -1.])
def test_nonpositive_curvature_is_reported_without_covariance(curvature):
    from types import SimpleNamespace

    parameter = Parameter("x", .1, -1., 1., .1, "nuisance")
    likelihood = SimpleNamespace(parameters=(parameter,), parameter_names=("x",), initial=np.array([.1]))
    evaluator = SimpleNamespace(parameter_names=("x",),
                                value_and_grad=lambda x: (.5*curvature*x[0]**2, curvature*x),
                                hessian=lambda x, **kw: np.array([[curvature]]))
    result = fit(likelihood, evaluator=evaluator)
    assert result.covariance_status == "nonpositive_curvature"
    assert result.covariance is None and result.errors is None
