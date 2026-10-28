import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dataclasses import replace

from combine_jax.bin_statistics import barlow_beeston_solution, error_moments
from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.likelihood import from_workspace
from combine_jax.root_io import open_workspace, load_caching_nll
from combine_jax.validation import validate_workspace


@pytest.fixture(scope="module")
def template_path(tmp_path_factory):
    path = Path(__file__).resolve().parents[1] / "scripts/build_template_fixture.py"
    spec = importlib.util.spec_from_file_location("build_template_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_template_fixture(tmp_path_factory.mktemp("template"))


@pytest.mark.parametrize("profile", [False, True])
def test_automcstats_nll_gradients_and_hessians(template_path, profile):
    with open_workspace(template_path) as workspace:
        likelihood = from_workspace(workspace, analytic_barlow_beeston=profile)
        types = likelihood.nodes["prop_binch"].attrs["bin_types"]
        assert {t for row in types for t in row} == {0, 1, 2, 3, 4}
        report = validate_workspace(workspace, random_points=25, seed=1819, analytic_barlow_beeston=profile)
    assert report["max_nll_absolute_error"] < 1e-6
    assert report["max_channel_absolute_error"] < 1e-6
    assert report["max_gradient_relative_error"] < 1e-4
    assert report["max_hessian_absolute_error"] < 1e-6
    assert bool(report["profiled_parameters"]) == profile
    for case in report["cases"]:
        for values in case.get("profiled", {}).values():
            assert abs(values["jax"]-values["combine"]) < 1e-12


def test_profile_hessian_is_schur_complement_and_includes_constraint(template_path):
    with open_workspace(template_path) as workspace:
        full = from_workspace(workspace, analytic_barlow_beeston=False)
        profiled = from_workspace(workspace)
    point = profiled.initial.copy()
    point[profiled.parameter_names.index("r")] = 1.7
    values = dict(zip(profiled.parameter_names, point, strict=True)) | profiled.profile_values(point)
    full_point = np.array([values[name] for name in full.parameter_names])
    value, gradient = full.value_and_grad(full_point)
    idx = full.parameter_names.index("prop_binch_bin0")
    assert abs(gradient[idx]) < 1e-12
    np.testing.assert_allclose(profiled.value(point), value, atol=1e-12)
    h = np.asarray(full.hessian(full_point))
    keep = [full.parameter_names.index(n) for n in profiled.parameter_names]
    schur = h[np.ix_(keep, keep)] - np.outer(h[keep, idx], h[idx, keep])/h[idx, idx]
    np.testing.assert_allclose(profiled.hessian(point), schur, atol=1e-11, rtol=1e-11)
    assert abs(float(profiled.at_zero(point).value(point))) < 1e-12


def test_prepared_profiled_terms_preserve_full_derivatives(template_path):
    from test_local import compare_local
    with open_workspace(template_path) as workspace:
        likelihood = from_workspace(workspace)
    point = likelihood.initial.copy()
    point[likelihood.parameter_names.index("r")] = 1.4
    compare_local(likelihood,[likelihood.initial,point],constraint_batch_size=32,
                  hessian_chunk_size=2,prepared=True)


@pytest.mark.parametrize("batch_size", [1, 32])
@pytest.mark.parametrize("hessian_chunk_size", [None, 1, 2, 8])
def test_local_profile_constraint_and_hessian_assembly(template_path, batch_size, hessian_chunk_size):
    from test_local import compare_local

    with open_workspace(template_path) as workspace:
        likelihood = from_workspace(workspace)
    point = likelihood.initial.copy()
    point[likelihood.parameter_names.index("r")] = 1.7
    local = compare_local(likelihood, [likelihood.initial, point], constraint_batch_size=batch_size,
                          hessian_chunk_size=hessian_chunk_size)
    assert len([t for t in local.terms if t.profiles]) == 1
    assert all(len(t.indices)<len(likelihood.parameters) for t in local.terms if not t.profiles)


def test_masked_channel_disables_profiling_and_retains_constraints(template_path):
    root=load_caching_nll()
    with open_workspace(template_path) as workspace:
        mask=root.RooRealVar("mask_template","",1.,0.,1.)
        getattr(workspace,"import")(mask)
        workspace.pdf("model_s").addChannelMasks(root.RooArgList(workspace.var("mask_template")))
        likelihood=from_workspace(workspace)
        assert not likelihood.profiles
        assert "prop_binch_bin0" in likelihood.parameter_names
        assert all(c.masked for c in likelihood.channels)
        report=validate_workspace(workspace,random_points=4,seed=7712)
    assert report["max_nll_absolute_error"]<1e-6
    assert report["max_gradient_relative_error"]<1e-4


def test_nonzero_global_observation_and_frozen_statistical_parameter(template_path):
    root = load_caching_nll()
    with open_workspace(template_path) as workspace:
        workspace.var("prop_binch_bin0_In").setVal(.8)
        report = validate_workspace(workspace, random_points=4)
        assert report["max_nll_absolute_error"] < 1e-6
        assert report["max_gradient_relative_error"] < 1e-4
        frozen = workspace.var("prop_binch_bin0")
        frozen.setVal(.4); frozen.setConstant(True)
        likelihood = from_workspace(workspace)
        assert not likelihood.profiles
        reference = root.cacheutils.CachingSimNLL(workspace.pdf("model_s"), workspace.data("data_obs"),
                                                 workspace.obj("ModelConfig").GetNuisanceParameters())
        reference.setAnalyticBarlowBeeston(True)
        reference.clearZeroPoint(); reference.clearConstantZeroPoint()
        np.testing.assert_allclose(likelihood.value(likelihood.initial), reference.getVal(), atol=1e-12)
        del reference


def test_quadratic_solve_zero_data_zero_error_and_cancellation():
    for mean, error, observed, gobs in [(20., 2., 0., 0.), (1e6, .01, 1e6+1, .2), (4., 2., 4., 0.)]:
        value = float(barlow_beeston_solution(mean, error, observed, gobs))
        assert np.isfinite(value)
        b = error + mean/error - gobs
        c = mean - observed - (mean/error)*gobs
        assert abs(value*value + b*value + c) < 1e-7 * max(1., abs(c))
    function = lambda t: barlow_beeston_solution(t[0], t[1], 10., .3)
    zero_error = jnp.array([10., 0.])
    assert np.isfinite(function(zero_error))
    assert np.all(np.isfinite(jax.grad(function)(zero_error)))


def test_profile_outside_root_bounds_is_invalid_not_clipped(template_path):
    with open_workspace(template_path) as workspace:
        workspace.var("prop_binch_bin0").setRange(-.001, .001)
        likelihood = from_workspace(workspace)
        assert np.isnan(likelihood.value(likelihood.initial))
        # Invoke the real updater directly: the boundary belongs to its setVal,
        # independent of RooAbsReal's outer NLL cache and zero-point state.
        reference = workspace.function("prop_binch")
        reference.setData(workspace.data("data_obs"))
        reference.setAnalyticBarlowBeeston(True)
        with pytest.raises(Exception, match="outside the default range"):
            reference.runBarlowBeeston()
        reference.setAnalyticBarlowBeeston(False)


def test_observable_dependent_error_coefficient_is_rejected(template_path):
    with open_workspace(template_path) as workspace:
        likelihood = from_workspace(workspace)
    node = likelihood.nodes["prop_binch"]
    invalid = replace(node, attrs={**node.attrs, "coefs": ("CMS_th1x", node.attrs["coefs"][1])})
    context = Evaluation(likelihood.nodes, likelihood.registry, likelihood.parameter_names, likelihood.initial)
    with pytest.raises(UnsupportedModelError, match="observable-dependent histogram coefficients"):
        error_moments(invalid, context)


def test_zero_error_profile_is_history_dependent_in_combine(root, template_path):
    x = root.RooRealVar("zero_error_x", "", .5, 0., 1.)
    x.setBins(1)
    hist = root.TH1D("one_bin", "", 1, 0., 1.)
    hist.SetDirectory(0); hist.SetBinContent(1, 10.); hist.SetBinError(1, 1.)
    statistical = root.CMSHistFunc("statistical", "", x, hist)
    hist.SetBinError(1, 0.)
    exact = root.CMSHistFunc("exact", "", x, hist)
    rate = root.RooRealVar("rate", "", 1., 0., 3.)
    one = root.RooConstVar("one", "", 1.)
    prop = root.CMSHistErrorPropagator("zero_error_prop", "", x, root.RooArgList(statistical, exact), root.RooArgList(rate, one))
    binpars = prop.setupBinPars(0.)
    nuisance = binpars.at(0)
    global_obs = root.RooRealVar(nuisance.GetName()+"_In", "", 0., -7., 7.)
    global_obs.setConstant(True)
    constraint = root.SimpleGaussianConstraint("bin_constraint", "", nuisance, global_obs, one)
    weight = root.RooRealVar("weight", "", 22., 0., 100.)
    variables = root.RooArgSet(x, weight)
    data = root.RooDataSet("one_bin_data", "", variables, root.RooFit.WeightVar(weight))
    data.add(variables, 22.)
    prop.setData(data); prop.setAnalyticBarlowBeeston(True)
    prop.runBarlowBeeston(); first = nuisance.getVal()
    rate.setVal(0.); prop.runBarlowBeeston()
    assert nuisance.getVal() == first
    rate.setVal(2.); prop.runBarlowBeeston(); second = nuisance.getVal()
    rate.setVal(0.); prop.runBarlowBeeston()
    assert nuisance.getVal() == second
    assert abs(first-second) > .1
    prop.setAnalyticBarlowBeeston(False)

    with open_workspace(template_path) as workspace:
        likelihood = from_workspace(workspace)
    nodes = dict(likelihood.nodes)
    for name in nodes["prop_binch"].attrs["functions"]:
        node = nodes[name]
        nodes[name] = replace(node, attrs={**node.attrs, "errors": np.zeros_like(node.attrs["errors"])})
    context = Evaluation(nodes, likelihood.registry, likelihood.parameter_names, likelihood.initial)
    assert np.isnan(likelihood.profiles[0].evaluate(context)["prop_binch_bin0"])
