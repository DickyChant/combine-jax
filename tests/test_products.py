import itertools

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


def make_product(root, mode, nested=False):
    x = root.RooRealVar("px", "", .2, -2., 3.)
    y = root.RooRealVar("py", "", .5, -3., 4.)
    mean = root.RooRealVar("pm", "", .2, -.5, .7)
    sigma = root.RooRealVar("ps", "", 1., .4, 2.)
    slope = root.RooRealVar("pa", "", .4, -.7, .7)
    conditional_mean = root.RooFormulaVar("conditional_mean", "@0*@1", root.RooArgList(x, slope))
    gx = root.RooGaussian("gx", "", x, mean, sigma)
    gy = root.RooGaussian("gy", "", y, mean if mode == "independent" else conditional_mean, sigma)
    ux = root.RooUniform("ux", "", root.RooArgSet(x))
    fraction = root.RooConstVar("fraction", "", .7)
    mixture = root.RooAddPdf("px_mixture", "", root.RooArgList(gx, ux), root.RooArgList(fraction))
    parent = mixture if nested else gx
    if mode == "independent":
        pdf = root.RooProdPdf("product_pdf", "", root.RooArgList(parent, gy))
    else:
        selected = root.RooArgSet(y if mode == "nset" else x)
        pdf = root.RooProdPdf("product_pdf", "", root.RooArgSet(parent),
                              root.RooFit.Conditional(root.RooArgSet(gy), selected, mode == "cset"))
    return locals()  # Keep every RooFit server alive with the model.


@pytest.mark.parametrize("mode,nested", list(itertools.product(["independent", "nset", "cset"], [False, True])))
def test_product_densities_and_derivatives(root, mode, nested):
    model = make_product(root, mode, nested)
    x, y, pdf = (model[k] for k in ("x", "y", "pdf"))
    registry = default_registry()
    nodes = registry.export([pdf], ["px", "py"])
    pars = parameters(pdf, ["px", "py"])
    names = tuple(p.name for p in pars)
    variables = {v.GetName(): v for v in (model["mean"], model["sigma"], model["slope"])}
    coordinates = np.array([[-1.8, -2.7], [-.3, .6], [.4, 1.2], [2.6, 3.6]])
    data = dict(zip(("px", "py"), coordinates.T, strict=True))
    norm_sets = [(root.RooArgSet(x), ("px",)), (root.RooArgSet(y), ("py",)), (root.RooArgSet(x, y), ("px", "py"))]
    for norm, axes in norm_sets:
        def value(theta):
            return Evaluation(nodes, registry, names, theta, data).density("product_pdf", axes)
        value, grad = jax.jit(value), jax.jit(jax.jacrev(value))

        if mode == "cset" and axes == ("px",):
            # ROOT reverts to ordinary normalisation if cset removes all
            # requested variables; gx and gy then need a joint x integral.
            with pytest.raises(UnsupportedModelError, match="overlapping product"):
                value(np.array([p.value for p in pars]))
            continue

        def reference(theta):
            for name, val in zip(names, theta, strict=True): variables[name].setVal(float(val))
            result = []
            for xx, yy in coordinates:
                x.setVal(float(xx)); y.setVal(float(yy)); result.append(pdf.getVal(norm))
            return np.array(result)

        for theta in np.random.default_rng(7411).uniform([p.minimum for p in pars], [p.maximum for p in pars], (15, len(pars))):
            np.testing.assert_allclose(value(theta), reference(theta), atol=2e-12, rtol=2e-12, equal_nan=False)
            fd = finite_difference(reference, theta, np.full(len(pars), 1e-4)).T
            np.testing.assert_allclose(grad(theta), fd, atol=1e-8, rtol=1e-6, equal_nan=False)


@pytest.mark.parametrize("mode", ["independent", "nset", "cset"])
def test_product_raw_values_and_integrals(root, mode):
    model = make_product(root, mode)
    x, y, pdf = (model[k] for k in ("x", "y", "pdf"))
    registry = default_registry(); nodes = registry.export([pdf])
    context = Evaluation(nodes, registry, (), ())
    np.testing.assert_allclose(context.value("product_pdf"), pdf.getVal(), atol=2e-15, equal_nan=False)
    axes = [("py",)] if mode != "independent" else [("px",), ("py",), ("px", "py")]
    for axes_ in axes:
        norm = root.RooArgSet(*[x if a == "px" else y for a in axes_])
        integral = pdf.createIntegral(norm)
        np.testing.assert_allclose(context.integral("product_pdf", axes_), integral.getVal(), atol=2e-12, equal_nan=False)
    if mode != "independent":
        with pytest.raises(UnsupportedModelError, match="overlapping dependencies"):
            context.integral("product_pdf", ("px", "py"))


@pytest.mark.parametrize("mode,nested", list(itertools.product(["independent", "nset", "cset"], [False, True])))
def test_product_extended_nll(root, mode, nested):
    root = load_caching_nll()
    model = make_product(root, mode, nested)
    x, y, shape = (model[k] for k in ("x", "y", "pdf"))
    rate = root.RooRealVar("prate", "", 20., 10., 30.)
    pdf = root.RooAddPdf("product_sum", "", root.RooArgList(shape), root.RooArgList(rate))
    weight = root.RooRealVar("pweight", "", 1., 0., 100.)
    collection = root.RooArgSet(x, y, weight)
    data = root.RooDataSet("product_data", "", collection, root.RooFit.WeightVar(weight))
    for (xx, yy), w in zip([[-1.7,-2.1],[-.3,.6],[.4,1.1],[2.7,3.4],[1.4,.2]], [2., 0., 6.5, 3., 1.2]):
        x.setVal(xx); y.setVal(yy); data.add(collection, w)
    registry = default_registry(); nodes = registry.export([pdf], ["px", "py"])
    pars = parameters(pdf, ["px", "py"])
    likelihood = Likelihood(nodes, registry, pars,
                            [Channel("prod", "product_sum", ("px", "py"), bind_data(data, ["px", "py"]))], [])
    reference = root.cacheutils.CachingAddNLL("product_ref", "", pdf, data, True)
    reference.clearConstantZeroPoint(); reference.clearZeroPoint()
    variables = {v.GetName(): v for v in (model["mean"], model["sigma"], model["slope"], rate)}

    def ref(theta):
        for name, val in zip(likelihood.parameter_names, theta, strict=True): variables[name].setVal(float(val))
        return reference.getVal()

    for index, theta in enumerate(np.random.default_rng(7412).uniform([p.minimum+.01 for p in pars], [p.maximum-.01 for p in pars], (25, len(pars)))):
        actual, grad = likelihood.value_and_grad(theta)
        np.testing.assert_allclose(actual, ref(theta), atol=1e-6, rtol=0, equal_nan=False)
        fd = finite_difference(ref, theta, np.full(len(pars), 1e-4))
        assert np.max(np.abs(grad-fd)/np.maximum(np.abs(fd), 1e-8)) < 1e-4
        if index == 0:
            fd_hessian = finite_difference(lambda p: likelihood.value_and_grad(p)[1], theta, np.full(len(pars), 1e-4)).T
            np.testing.assert_allclose(likelihood.hessian(theta), fd_hessian, atol=1e-6, rtol=2e-7, equal_nan=False)


def test_product_omits_unrequested_factors(root):
    model = make_product(root, "nset")
    x, y, gx, gy = (model[k] for k in ("x", "y", "gx", "gy"))
    norm_y = root.RooArgSet(y)
    alone = root.RooProdPdf("conditional_only", "", root.RooArgSet(), root.RooFit.Conditional(root.RooArgSet(gy), norm_y))
    constraint = root.RooGaussian("product_constraint", "", model["mean"], model["fraction"], model["sigma"])
    with_constraint = root.RooProdPdf("with_constraint", "", root.RooArgList(gx, constraint))
    registry = default_registry(); nodes = registry.export([alone, with_constraint])
    for pdf in (alone, with_constraint):
        for axes, ns in [(("px",), root.RooArgSet(x)), (("py",), norm_y), (("px", "py"), root.RooArgSet(x,y))]:
            value = Evaluation(nodes, registry, (), ()).density(pdf.GetName(), axes)
            np.testing.assert_allclose(value, pdf.getVal(ns), atol=2e-15, equal_nan=False)


@pytest.mark.parametrize("nested", [False, True])
def test_three_axis_conditional_chain(root, nested):
    model = make_product(root, "nset")
    x, y, gx, gy = (model[k] for k in ("x", "y", "gx", "gy"))
    z = root.RooRealVar("pz", "", -.3, -4., 5.)
    gz = root.RooGaussian("gz", "", z, y, model["sigma"])
    if nested:
        pdf = root.RooProdPdf("chain", "", root.RooArgSet(model["pdf"]),
                             root.RooFit.Conditional(root.RooArgSet(gz), root.RooArgSet(z)))
    else:
        pdf = root.RooProdPdf("chain", "", root.RooArgSet(gx),
                             root.RooFit.Conditional(root.RooArgSet(gy), root.RooArgSet(y)),
                             root.RooFit.Conditional(root.RooArgSet(gz), root.RooArgSet(z)))
    norm = root.RooArgSet(x, y, z)
    registry = default_registry(); nodes = registry.export([pdf], ["px", "py", "pz"])
    for xx, yy, zz in np.random.default_rng(7413).uniform([-2., -3., -4.], [3., 4., 5.], (25, 3)):
        x.setVal(float(xx)); y.setVal(float(yy)); z.setVal(float(zz))
        ctx = Evaluation(nodes, registry, (), (), {"px": xx, "py": yy, "pz": zz})
        np.testing.assert_allclose(ctx.density("chain", ("px", "py", "pz")), pdf.getVal(norm), atol=2e-14, rtol=2e-13, equal_nan=False)


@pytest.mark.parametrize("mode", ["empty_nset", "irrelevant_nset", "empty_cset", "cset"])
def test_empty_effective_specification_reverts_to_ordinary(root, mode):
    model = make_product(root, "nset")
    x, y = model["x"], model["y"]
    g = root.RooGaussian("direct_conditional", "", y, x, model["sigma"])
    unused = root.RooRealVar("unrelated", "", .4, -1., 1.)
    selected = root.RooArgSet(x) if mode == "cset" else root.RooArgSet(unused) if mode == "irrelevant_nset" else root.RooArgSet()
    pdf = root.RooProdPdf("empty_spec", "", root.RooArgSet(),
                         root.RooFit.Conditional(root.RooArgSet(g), selected, mode.endswith("cset")))
    registry = default_registry(); nodes = registry.export([pdf])
    ns = root.RooArgSet(x)
    ctx = Evaluation(nodes, registry, (), ())
    np.testing.assert_allclose(ctx.density("empty_spec", ("px",)), pdf.getVal(ns), atol=2e-15, equal_nan=False)


def test_overlapping_and_cyclic_products_are_rejected(root):
    model = make_product(root, "nset")
    x, y, gx, gy = (model[k] for k in ("x", "y", "gx", "gy"))
    overlap = root.RooProdPdf("overlap", "", root.RooArgList(gx, gy))
    reverse = root.RooGaussian("reverse", "", x, y, model["sigma"])
    cycle = root.RooProdPdf("cycle", "", root.RooFit.Conditional(root.RooArgSet(gy), root.RooArgSet(y)),
                            root.RooFit.Conditional(root.RooArgSet(reverse), root.RooArgSet(x)))
    registry = default_registry()
    for pdf, message in [(overlap, "overlapping product"), (cycle, "cyclic conditional")]:
        nodes = registry.export([pdf])
        with pytest.raises(UnsupportedModelError, match=message):
            Evaluation(nodes, registry, (), ()).density(pdf.GetName(), ("px", "py"))


@pytest.mark.parametrize("mode", ["cutoff", "range", "reference", "default", "self_normalised", "numeric"])
def test_product_nonstandard_options_rejected(root, mode):
    model = make_product(root, "independent")
    pdf = model["pdf"]
    if mode == "cutoff": pdf = root.RooProdPdf("cutoff_product", "", root.RooArgList(model["gx"], model["gy"]), .1)
    if mode == "range": pdf.setNormRange("custom")
    if mode == "reference": pdf.fixRefRange("custom")
    if mode == "default": pdf.setDefNormSet(root.RooArgSet(model["x"]))
    if mode == "self_normalised": pdf.setSelfNormalized(False)
    if mode == "numeric": pdf.forceNumInt(True)
    with pytest.raises(UnsupportedModelError, match="Product cutoff"):
        default_registry().export([pdf])
