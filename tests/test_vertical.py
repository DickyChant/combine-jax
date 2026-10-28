from array import array

import jax
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import UnsupportedModelError, parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_fast_vertical(root, *, logarithmic=False, active=False, region=.7, morphs=True, old=False):
    edges = [0., .5, 2., 4., 7.]
    x = root.RooRealVar("vx", "", 1., edges[0], edges[-1])
    x.setBinning(root.RooBinning(4, array("d", edges)))
    a = root.RooRealVar("va", "", 0., -3., 3.)
    b = root.RooRealVar("vb", "", 0., -3., 3.)
    rows = [[2., 5., 0., .2], [3.5, 4., .5, .2], [1., 6., 0., .2],
            [2.5, 6., .2, .3], [1.5, 4., .3, .1]]
    templates = root.TList()
    held = []
    for i, row in enumerate(rows if morphs else rows[:1]):
        hist = root.TH1D(f"vertical_stored_{i}", "", 4, array("d", edges))
        hist.SetDirectory(0)
        for j, value in enumerate(row, 1):
            hist.SetBinContent(j, value)
        templates.Add(hist)
        held.append(hist)
    coefs = root.RooArgList(a, b) if morphs else root.RooArgList()
    if old:
        functions = root.RooArgList()
        for i, hist in enumerate(list(held)):
            data = root.RooDataHist(f"vdata_{i}", "", root.RooArgList(x), hist)
            component = root.RooHistPdf(f"vcomponent_{i}", "", root.RooArgSet(x), data, 0)
            functions.add(component)
            held.extend((data, component))
        pdf = root.FastVerticalInterpHistPdf("vertical", "", x, functions, coefs, region, -1 if logarithmic else 1)
    else:
        pdf = root.FastVerticalInterpHistPdf2("vertical", "", x, templates, coefs, region, -1 if logarithmic else 1)
    if active:
        pdf.setActiveBins(3)
    return x, a, b, pdf, held


def make_vertical(root, algorithm=2, region=.7):
    x = root.RooRealVar("vx", "", .3, -4., 5.)
    a = root.RooRealVar("va", "", 0., -3., 3.)
    b = root.RooRealVar("vb", "", 0., -3., 3.)
    held = []
    functions = root.RooArgList()
    for i, (mean, sigma) in enumerate([(0., 1.1), (.1, 1.2), (-.15, 1.05), (.2, 1.12), (-.1, 1.07)]):
        m = root.RooConstVar(f"vmean_{i}", "", mean)
        s = root.RooConstVar(f"vsigma_{i}", "", sigma)
        p = root.RooGaussian(f"vgaussian_{i}", "", x, m, s)
        functions.add(p); held.extend((m, s, p))
    pdf = root.VerticalInterpPdf("vertical", "", functions, root.RooArgList(a, b), region, algorithm)
    return x, a, b, pdf, held


@pytest.mark.parametrize("algorithm,region", [(0, 0.), (0, .7), (1, .7), (2, .7)])
def test_vertical_pdf_values_integrals_and_gradients(root, algorithm, region):
    x, a, b, pdf, held = make_vertical(root, algorithm, region)
    registry = default_registry()
    nodes = registry.export([pdf], ["vx"])
    coords = np.array([-3.8, -1.1, .4, 2., 4.5])
    normalization = root.RooArgSet(x)
    integral = pdf.createIntegral(normalization)

    def value(theta):
        context = Evaluation(nodes, registry, ("va", "vb"), theta, {"vx": coords})
        return context.value("vertical"), context.integral("vertical", ("vx",)), context.density("vertical", ("vx",))

    value = jax.jit(value)
    gradient = jax.jit(jax.jacrev(lambda theta: value(theta)[2]))

    def reference(theta):
        a.setVal(float(theta[0])); b.setVal(float(theta[1]))
        vals, densities = [], []
        for coord in coords:
            x.setVal(float(coord))
            vals.append(float(pdf.getVal()))
            densities.append(float(pdf.getVal(normalization)))
        return np.array(vals), float(integral.getVal()), np.array(densities)

    points = [np.array([t, -.31]) for t in (-2., -.70001, -.7, -.69999, 0., .69999, .7, .70001, 2.)]
    points.extend(np.random.default_rng(773).uniform(-2., 2., (15, 2)))
    for theta in points:
        actual, expected = value(theta), reference(theta)
        for v, ref in zip(actual, expected, strict=True):
            np.testing.assert_allclose(v, ref, atol=3e-14, rtol=3e-14)
        # Algo 0 is not differentiable at the interpolation-region boundary.
        if algorithm == 0 and (abs(abs(theta[0])-region) < 1e-4):
            continue
        fd = finite_difference(lambda point: reference(point)[2], theta, np.full(2, 1e-6)).T
        np.testing.assert_allclose(gradient(theta), fd, atol=2e-9, rtol=2e-6)


@pytest.mark.parametrize("logarithmic,active,region,morphs,old", [
    (False, False, .7, True, False), (True, False, .7, True, False),
    (False, True, .7, True, False), (True, True, .7, True, False),
    (False, False, 0., True, False), (True, False, 0., True, False),
    (False, False, .7, False, False), (True, False, .7, False, False),
    (False, False, .7, True, True), (True, False, .7, True, True),
    (False, False, 0., True, True), (True, False, .7, False, True)])
def test_fast_vertical_stored_bins_and_gradients(root, logarithmic, active, region, morphs, old):
    x, a, b, pdf, held = make_fast_vertical(root, logarithmic=logarithmic, active=active, region=region, morphs=morphs, old=old)
    registry = default_registry()
    nodes = registry.export([pdf], ["vx"])
    coords = np.array([0., .25, .5, .500001, 1.25, 2., 3., 4., 5.5, 7.])
    edges = nodes["vertical"].attrs["edges"]
    normalization = root.RooArgSet(x)

    def value(theta):
        context = Evaluation(nodes, registry, ("va", "vb"), theta, {"vx": coords})
        return context.density("vertical", ("vx",)), context.integral("vertical", ("vx",))

    value = jax.jit(value)
    gradient = jax.jit(jax.jacrev(lambda theta: value(theta)[0]))

    def reference(theta):
        a.setVal(float(theta[0])); b.setVal(float(theta[1]))
        result = []
        for coord in coords:
            x.setVal(float(coord)); result.append(float(pdf.getVal(normalization)))
        # Exact quadrature on the known bins also includes the inactive tail.
        area = 0.
        for low, high in zip(edges[:-1], edges[1:], strict=True):
            x.setVal(float((low+high)/2))
            area += float(pdf.getVal())*(high-low)
        return np.array(result), area

    points = [np.array([t, -.31]) for t in (-2., -.70001, -.7, -.69999, 0., .69999, .7, .70001, 2.)]
    points.extend(np.random.default_rng(774).uniform(-2., 2., (15, 2)))
    for theta in points:
        actual, expected = value(theta), reference(theta)
        for v, ref in zip(actual, expected, strict=True):
            np.testing.assert_allclose(v, ref, atol=4e-14, rtol=4e-14)
        if region == 0 and theta[0] == 0:
            continue
        fd = finite_difference(lambda point: reference(point)[0], theta, np.full(2, 1e-6)).T
        np.testing.assert_allclose(gradient(theta), fd, atol=2e-9, rtol=2e-6)


@pytest.mark.parametrize("kind", ["fast", "fast_log", "fast_old", "fast_old_log", "quadratic0", "quadratic1", "polynomial"])
def test_vertical_extended_channel_against_combine(root, kind):
    root = load_caching_nll()
    if kind.startswith("fast"):
        x, a, b, shape, held = make_fast_vertical(root, logarithmic=kind.endswith("log"), old="old" in kind)
        rows = [(.25, 2.3), (1.25, 4.7), (3., 0.), (5.5, .2)]
    else:
        x, a, b, shape, held = make_vertical(root, {"quadratic0": 0, "quadratic1": 1, "polynomial": 2}[kind])
        rows = [(-1.5, 1.1), (-.3, 3.7), (1.4, 2.4), (2.5, 0.)]
    rate = root.RooRealVar("rate", "", 7., .1, 30.)
    pdf = root.RooAddPdf("vertical_sum", "", root.RooArgList(shape), root.RooArgList(rate))
    weight = root.RooRealVar("vertical_weight", "", 1., 0., 20.)
    variables = root.RooArgSet(x, weight)
    data = root.RooDataSet("vertical_data", "", variables, root.RooFit.WeightVar(weight))
    for coord, w in rows:
        x.setVal(coord); data.add(variables, w)
    registry = default_registry()
    nodes = registry.export([pdf], ["vx"])
    likelihood = Likelihood(nodes, registry, parameters(pdf, ["vx"]),
                            [Channel("vchannel", "vertical_sum", ("vx",), bind_data(data, ["vx"]))], [])
    reference = root.cacheutils.CachingAddNLL("vertical_reference", "", pdf, data, True)
    reference.clearConstantZeroPoint(); reference.clearZeroPoint()
    variables = {str(v.GetName()): v for v in (a, b, rate)}

    def ref(theta):
        for name, value in zip(likelihood.parameter_names, theta, strict=True):
            variables[name].setVal(float(value))
        return float(reference.getVal())

    for theta in np.random.default_rng(778).uniform([4., -2., -2.], [10., 2., 2.], (25, 3)):
        value, grad = likelihood.value_and_grad(theta)
        np.testing.assert_allclose(value, ref(theta), atol=1e-6, rtol=0)
        fd = finite_difference(ref, theta, np.full(3, 1e-5))
        assert np.max(np.abs(grad-fd)/np.maximum(np.abs(fd), 1e-8)) < 1e-4


@pytest.mark.parametrize("algorithm", [-1, 100, 102])
def test_numerical_vertical_normalization_rejected(root, algorithm):
    x, a, b, pdf, held = make_vertical(root, algorithm)
    with pytest.raises(UnsupportedModelError, match="vertical.*numerical/multiplicative"):
        default_registry().export([pdf], ["vx"])


def test_vertical_nonpositive_floor_is_not_a_clamp(root):
    theta = root.RooRealVar("floor_theta", "", 0., -3., 3.)
    central = root.RooConstVar("floor_central", "", 1.)
    high = root.RooConstVar("floor_high", "", .1)
    low = root.RooConstVar("floor_low", "", 3.)
    pdf = root.VerticalInterpPdf("floor_pdf", "", root.RooArgList(central, high, low), root.RooArgList(theta), 0., 0)
    pdf.setFloorVals(.05, .15)
    registry = default_registry()
    nodes = registry.export([pdf])
    for t in (-2., 0., 1.1, 2.):
        theta.setVal(t)
        context = Evaluation(nodes, registry, ("floor_theta",), [t])
        np.testing.assert_allclose(context.value("floor_pdf"), pdf.getVal(), atol=1e-15, rtol=0)
    theta.setVal(1.1)
    assert 0 < pdf.getVal() < .05


def test_old_fast_nonhistogram_template_rejected(root):
    x = root.RooRealVar("old_vx", "", 0., -1., 1.)
    shape = root.RooUniform("old_vuniform", "", root.RooArgSet(x))
    pdf = root.FastVerticalInterpHistPdf("old_vpdf", "", x, root.RooArgList(shape), root.RooArgList())
    with pytest.raises(UnsupportedModelError, match="old_vpdf.*fixed RooHistPdf"):
        default_registry().export([pdf], ["old_vx"])
