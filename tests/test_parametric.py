"""Weighted event densities use continuous normalisation, including binned rows."""

import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize("binned", [False, True])
def test_weighted_gaussian_uniform_mixture_against_caching_nll(root, binned):
    root = load_caching_nll()
    x = root.RooRealVar("x", "", 0., -2., 3.)
    x.setBins(5)
    mean = root.RooRealVar("mean", "", .4, -.8, 1.2)
    sigma = root.RooRealVar("sigma", "", .7, .1, 3.)
    signal = root.RooGaussian("signal", "", x, mean, sigma)
    background = root.RooUniform("background", "", root.RooArgSet(x))
    ns = root.RooRealVar("ns", "", 3., .1, 10.)
    nb = root.RooRealVar("nb", "", 2., .1, 10.)
    pdf = root.RooAddPdf("mixture", "", root.RooArgList(signal, background), root.RooArgList(ns, nb))
    if binned:
        hist = root.TH1D("hist", "", 5, -2., 3.)
        hist.SetDirectory(0)
        for i, w in enumerate([.2, 0., 1.5, 2.3, .4], 1):
            hist.SetBinContent(i, w)
        data = root.RooDataHist("data", "", root.RooArgList(x), hist)
    else:
        weight = root.RooRealVar("weight", "", 1., 0., 5.)
        variables = root.RooArgSet(x, weight)
        data = root.RooDataSet("data", "", variables, root.RooFit.WeightVar(weight))
        for coordinate, w in [(-1.63, .2), (-.41, 0.), (.23, 1.5), (.57, 2.3), (2.85, .4)]:
            x.setVal(coordinate)
            data.add(variables, w)
    bound = bind_data(data, ["x"])
    registry = default_registry()
    nodes = registry.export([pdf], ["x"])
    parameter_info = parameters(pdf, ["x"])
    likelihood = Likelihood(nodes, registry, parameter_info, [Channel("one", "mixture", ("x",), bound)], [])
    # Direct CachingAddNLL must retain zero rows with generic CachingPdf:
    # realFill_ writes at original row indices. CachingSimNLL normally removes
    # zero rows when it splits the data. Keep aligned arrays in this direct probe.
    reference = root.cacheutils.CachingAddNLL("reference", "", pdf, data, True)
    reference.clearConstantZeroPoint()
    reference.clearZeroPoint()
    variables = {v.GetName(): v for v in [mean, sigma, ns, nb]}

    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
    from stable_root_intensities import RootIntensities
    predictor = RootIntensities(root, pdf, [x], bound.coordinates["x"][:,None], bound.weights)
    baseline = predictor.predict()
    baseline_nll = float(reference.getVal())

    def ref(point):
        for name, value in zip(likelihood.parameter_names, point, strict=True):
            variables[name].setVal(float(value))
        return float(reference.getVal())

    for point in np.random.default_rng(147).uniform([-.7, .5, .5, .3], [1.1, 5., 5., 2.], (15, 4)):
        value, gradient = likelihood.value_and_grad(point)
        np.testing.assert_allclose(value, ref(point), atol=1e-6, rtol=0)
        np.testing.assert_allclose(predictor.delta(baseline), float(reference.getVal())-baseline_nll,
                                   atol=1e-10, rtol=0)
        fd = finite_difference(ref, point, np.full(4, 1e-4))
        assert np.max(np.abs(np.asarray(gradient)-fd) / np.maximum(np.abs(fd), 1e-8)) < 1e-4
