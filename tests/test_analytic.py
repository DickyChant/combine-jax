import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize("negate", [False, True])
def test_exponential_values_normalisation_and_derivatives(root, negate):
    x = root.RooRealVar("ex", "", .4, -2., 5.)
    slope = root.RooRealVar("ec", "", -.2, -1., 1.)
    pdf = root.RooExponential("exponential", "", x, slope, negate)
    registry = default_registry()
    nodes = registry.export([pdf], ["ex"])
    coordinates = np.array([-2., -.3, 1.5, 5.])
    norm = root.RooArgSet(x)
    integral = pdf.createIntegral(norm)

    def value(theta):
        context = Evaluation(nodes, registry, ("ec",), theta, {"ex": coordinates})
        return context.density("exponential", ("ex",)), context.integral("exponential", ("ex",))

    value = jax.jit(value)
    gradient = jax.jit(jax.jacrev(lambda t: value(t)[0]))

    def ref(theta):
        slope.setVal(float(theta[0]))
        result = []
        for coordinate in coordinates:
            x.setVal(float(coordinate)); result.append(pdf.getVal(norm))
        return np.array(result)

    for t in (-.9, -.3, -.00001, 0., .00001, .2, .8):
        theta = np.array([t])
        expected = ref(theta)
        np.testing.assert_allclose(value(theta)[0], expected, atol=2e-12, rtol=2e-12, equal_nan=False)
        np.testing.assert_allclose(value(theta)[1], integral.getVal(), atol=3e-11, rtol=2e-12, equal_nan=False)
        fd = finite_difference(ref, theta, np.array([1e-4])).T
        np.testing.assert_allclose(gradient(theta), fd, atol=3e-9, rtol=2e-6, equal_nan=False)
    # A second derivative exists at the exactly uniform limit.
    hessian = jax.jacfwd(jax.jacrev(lambda t: value(t)[0]))(np.array([0.]))
    expected = ((coordinates-1.5)**2-7**2/12)/7
    np.testing.assert_allclose(np.asarray(hessian)[:, 0, 0], expected, atol=1e-13, rtol=1e-13)
    # RooExponential also advertises integration over its coefficient.
    nodes = registry.export([pdf], ["ec"])
    x.setVal(.4)
    context = Evaluation(nodes, registry, ("ex",), [.4])
    root_integral = pdf.createIntegral(root.RooArgSet(slope))
    np.testing.assert_allclose(context.integral("exponential", ("ec",)), root_integral.getVal(), atol=2e-14, rtol=2e-14)


@pytest.mark.parametrize("degree,reference", [(0, False), (1, False), (2, False), (4, False), (7, False), (4, True)])
def test_bernstein_values_integrals_and_gradients(root, degree, reference):
    x = root.RooRealVar("bx", "", 120., 100., 180.)
    coefficients = [root.RooRealVar(f"bc{i}", "", 1.+.1*i, .1, 3.) for i in range(degree+1)]
    pdf = root.RooBernstein("bernstein", "", x, root.RooArgList(*coefficients))
    if reference:
        x.setRange("bern_reference", 100., 180.)
        pdf.selectNormalizationRange("bern_reference", True)
        x.setRange(110., 170.)
    registry = default_registry()
    nodes = registry.export([pdf], ["bx"])
    names = tuple(str(v.GetName()) for v in coefficients)
    coordinates = np.linspace(x.getMin(), x.getMax(), 9)
    norm = root.RooArgSet(x)
    integral = pdf.createIntegral(norm)

    def value(theta):
        context = Evaluation(nodes, registry, names, theta, {"bx": coordinates})
        return context.value("bernstein"), context.integral("bernstein", ("bx",)), context.density("bernstein", ("bx",))

    value = jax.jit(value)
    gradient = jax.jit(jax.jacrev(lambda t: value(t)[2]))

    def ref(theta):
        for variable, val in zip(coefficients, theta, strict=True): variable.setVal(float(val))
        raw, density = [], []
        for coordinate in coordinates:
            x.setVal(float(coordinate)); raw.append(pdf.getVal()); density.append(pdf.getVal(norm))
        return np.array(raw), integral.getVal(), np.array(density)

    for theta in np.random.default_rng(6112).uniform(.2, 2.5, (20, degree+1)):
        for actual, expected in zip(value(theta), ref(theta), strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-10, rtol=2e-12, equal_nan=False)
        fd = finite_difference(lambda t: ref(t)[2], theta, np.full(degree+1, 1e-4)).T
        np.testing.assert_allclose(gradient(theta), fd, atol=2e-10, rtol=2e-6, equal_nan=False)


def test_observable_dependent_analytic_parameters_rejected(root):
    x = root.RooRealVar("ax", "", .4, .1, .9)
    dependent = root.RooFormulaVar("dependent", "2*@0", root.RooArgList(x))
    registry = default_registry()
    for pdf in (root.RooExponential("exp_dependent", "", x, dependent),
                root.RooBernstein("bern_dependent", "", x, root.RooArgList(dependent))):
        nodes = registry.export([pdf], ["ax"])
        with pytest.raises(UnsupportedModelError, match="depend"):
            Evaluation(nodes, registry, (), ()).integral(pdf.GetName(), ("ax",))


def test_infinite_exponential_integral_rejected(root):
    x = root.RooRealVar("infinite_x", "", 1., 0., float("inf"))
    slope = root.RooConstVar("infinite_slope", "", -1.)
    pdf = root.RooExponential("infinite_exp", "", x, slope)
    registry = default_registry()
    nodes = registry.export([pdf], ["infinite_x"])
    with pytest.raises(UnsupportedModelError, match="finite positive range"):
        Evaluation(nodes, registry, (), ()).integral("infinite_exp", ("infinite_x",))
