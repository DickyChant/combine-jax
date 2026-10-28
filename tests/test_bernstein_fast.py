import jax
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_fast_bernstein(root, degree):
    x = root.RooRealVar("fbx", "", .4, -2., 3.)
    coefficients = [root.RooRealVar(f"fbc{i}", "", 0., -.1, 4.) for i in range(degree)]
    pdf = root.RooBernsteinFast[degree]("fast_bernstein", "", x, root.RooArgList(*coefficients))
    return x, coefficients, pdf


@pytest.mark.parametrize("degree", range(1, 8))
def test_fast_bernstein_values_integrals_and_gradients(root, degree):
    x, coefficients, pdf = make_fast_bernstein(root, degree)
    registry = default_registry()
    nodes = registry.export([pdf], ["fbx"])
    names = tuple(str(c.GetName()) for c in coefficients)
    coordinates = np.linspace(-2., 3., 9)
    norm = root.RooArgSet(x)
    integral = pdf.createIntegral(norm)

    def value(theta):
        context = Evaluation(nodes, registry, names, theta, {"fbx": coordinates})
        return context.value("fast_bernstein"), context.integral("fast_bernstein", ("fbx",)), context.density("fast_bernstein", ("fbx",))

    value = jax.jit(value)
    gradient = jax.jit(jax.jacrev(lambda t: value(t)[2]))

    def ref(theta):
        for c, v in zip(coefficients, theta, strict=True): c.setVal(float(v))
        raw, density = [], []
        for coordinate in coordinates:
            x.setVal(float(coordinate))
            density.append(pdf.getVal(norm)); raw.append(pdf.getVal())
        return np.array(raw), integral.getVal(), np.array(density)

    points = [np.zeros(degree)]
    points.extend(np.random.default_rng(6212).uniform(.3, 2., (20, degree)))
    for theta in points:
        for actual, expected in zip(value(theta), ref(theta), strict=True):
            np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=2e-11, equal_nan=False)
        if np.all(theta == 0):
            continue  # Negative finite-difference probes at an endpoint can make the PDF negative.
        fd = finite_difference(lambda t: ref(t)[2], theta, np.full(degree, 1e-4)).T
        np.testing.assert_allclose(gradient(theta), fd, atol=2e-8, rtol=2e-6, equal_nan=False)


@pytest.mark.parametrize("degree", [1, 4, 7])
def test_fast_bernstein_extended_nll(root, degree):
    root = load_caching_nll()
    x, coefficients, shape = make_fast_bernstein(root, degree)
    for c in coefficients: c.setVal(1.)
    rate = root.RooRealVar("fbrate", "", 20., 1., 40.)
    pdf = root.RooAddPdf("fast_bernstein_sum", "", root.RooArgList(shape), root.RooArgList(rate))
    weight = root.RooRealVar("fbweight", "", 1., 0., 100.)
    collection = root.RooArgSet(x, weight)
    data = root.RooDataSet("fbdata", "", collection, root.RooFit.WeightVar(weight))
    for coordinate, w in zip(np.linspace(-1.8, 2.8, 9), [2., 3., 1., 0., 2.1, 4., 2.9, 1., 4.]):
        x.setVal(float(coordinate)); data.add(collection, w)
    registry = default_registry()
    nodes = registry.export([pdf], ["fbx"])
    likelihood = Likelihood(nodes, registry, parameters(pdf, ["fbx"]),
                            [Channel("fb", "fast_bernstein_sum", ("fbx",), bind_data(data, ["fbx"]))], [])
    reference = root.cacheutils.CachingAddNLL("fbref", "", pdf, data, True)
    reference.clearConstantZeroPoint(); reference.clearZeroPoint()
    variables = {str(c.GetName()): c for c in (*coefficients, rate)}

    def ref(theta):
        for name, val in zip(likelihood.parameter_names, theta, strict=True): variables[name].setVal(float(val))
        return reference.getVal()

    for theta in np.random.default_rng(6213).uniform([.3]*degree+[12.], [2.]*degree+[30.], (25, degree+1)):
        actual, grad = likelihood.value_and_grad(theta)
        np.testing.assert_allclose(actual, ref(theta), atol=1e-6, rtol=0, equal_nan=False)
        fd = finite_difference(ref, theta, np.full(degree+1, 1e-3))
        assert np.max(np.abs(grad-fd)/np.maximum(np.abs(fd), 1e-8)) < 1e-4
