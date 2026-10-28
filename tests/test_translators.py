import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry


def compile_node(node, parameter_names, observables=()):
    registry = default_registry()
    graph = registry.export([node], observables)

    def context(theta, data=None):
        return Evaluation(graph, registry, parameter_names, theta, data)

    return context


@pytest.mark.parametrize("low,high", [(.8, 1.15), (1/1.2, 1.2), (1., 1.), (1.1, 1.3)])
def test_asympow_value_gradient_and_hessian(root, low, high):
    theta = root.RooRealVar("theta", "", 0., -5., 5.)
    lo, hi = root.RooConstVar("low", "", low), root.RooConstVar("high", "", high)
    pdf = root.AsymPow("power", "", lo, hi, theta)
    context = compile_node(pdf, ("theta",))
    value = jax.jit(lambda t: context(t).value("power"))
    gradient = jax.jit(jax.grad(value))
    hessian = jax.jit(jax.hessian(value))
    for t in (-2., -.50001, -.5, -.49999, -.2, 0., .2, .49999, .5, .50001, 2.):
        theta.setVal(t)
        np.testing.assert_allclose(value(jnp.array([t])), pdf.getVal(), rtol=1e-14)
        h = 1e-5
        theta.setVal(t+h)
        upper = pdf.getVal()
        theta.setVal(t-h)
        lower = pdf.getVal()
        np.testing.assert_allclose(gradient(jnp.array([t]))[0], (upper-lower)/(2*h), atol=1e-9, rtol=1e-7)
        # The function is C2 at +/-0.5 but its third derivative jumps. The
        # centred derivative of the gradient then has O(h), not O(h^2), error.
        def second(step):
            return (gradient(jnp.array([t+step])) - gradient(jnp.array([t-step]))) / (2*step)
        fd2 = 2*second(h/2) - second(h) if abs(t) == .5 else second(h)
        np.testing.assert_allclose(hessian(jnp.array([t]))[0], fd2, atol=1e-8, rtol=1e-6)


def test_ordered_repeated_dependencies(root):
    theta = root.RooRealVar("theta", "", .3, -3., 3.)
    shared = root.RooConstVar("shared", "", 1.2)
    pdf = root.AsymPow("power", "", shared, shared, theta)
    context = compile_node(pdf, ("theta",))
    assert context(jnp.array([.3])).nodes["power"].inputs == ("shared", "shared", "theta")
    np.testing.assert_allclose(context(jnp.array([.3])).value("power"), pdf.getVal(), rtol=1e-14)
    addition = root.RooAddition("sum", "", root.RooArgList(theta, theta, shared))
    summed = compile_node(addition, ("theta",))
    np.testing.assert_allclose(summed(jnp.array([.3])).value("sum"), addition.getVal(), rtol=1e-14)


def test_process_normalization_all_factor_types(root):
    a = root.RooRealVar("a", "", 0., -3., 3.)
    b = root.RooRealVar("b", "", 0., -3., 3.)
    r = root.RooRealVar("r", "", 1., .1, 4.)
    norm = root.ProcessNormalization("yield", "", 17.3)
    norm.addLogNormal(1.07, a)
    norm.addAsymmLogNormal(.7, 1.25, b)
    norm.addAsymmLogNormal(1/1.3, 1.3, a)
    norm.addOtherFactor(r)
    context = compile_node(norm, ("a", "b", "r"))
    evaluate = jax.jit(lambda t: context(t).value("yield"))
    for point in np.random.default_rng(42).uniform([-2., -2., .1], [2., 2., 4.], (30, 3)):
        for variable, value in zip((a, b, r), point):
            variable.setVal(float(value))
        np.testing.assert_allclose(evaluate(point), norm.getVal(), rtol=2e-14)


def test_gaussian_raw_normalized_and_integral(root):
    x = root.RooRealVar("x", "", .2, -1.7, 2.6)
    mean = root.RooRealVar("mean", "", .1, -1., 1.)
    sigma = root.RooRealVar("sigma", "", .8, .1, 3.)
    pdf = root.RooGaussian("gauss", "", x, mean, sigma)
    obs = root.RooArgSet(x)
    integral = pdf.createIntegral(obs)
    context = compile_node(pdf, ("mean", "sigma"), ("x",))
    for m, s in np.random.default_rng(3).uniform([-.9, .2], [.9, 2.5], (20, 2)):
        mean.setVal(float(m)); sigma.setVal(float(s))
        c = context(jnp.array([m, s]), {"x": .2})
        np.testing.assert_allclose(c.value("gauss"), pdf.getVal(), rtol=1e-13)
        np.testing.assert_allclose(c.integral("gauss", ("x",)), integral.getVal(), rtol=1e-13)
        np.testing.assert_allclose(c.density("gauss", ("x",)), pdf.getVal(obs), rtol=1e-13)


def test_unknown_class_is_named_even_if_constant(root):
    x = root.RooRealVar("x", "", 1., 0., 2.)
    k = root.RooConstVar("k", "", -.2)
    pdf = root.RooChebychev("unsupported_cheby", "", x, root.RooArgList(k))
    with pytest.raises(UnsupportedModelError, match="RooChebychev: unsupported_cheby"):
        default_registry().export([pdf])


def test_missing_observable_binding_does_not_use_nominal(root):
    x = root.RooRealVar("x", "", .2, -1., 2.)
    context = compile_node(x, (), ("x",))
    with pytest.raises(ValueError, match="Unbound observable x"):
        context(jnp.array([])).value("x")


def test_parameter_dependent_integral_is_not_approximated(root):
    x = root.RooRealVar("x", "", .4, .1, 2.)
    zero = root.RooConstVar("zero", "", 0.)
    width = root.RooProduct("width", "", root.RooArgList(x, x))
    pdf = root.RooGaussian("dependent_width", "", x, zero, width)
    context = compile_node(pdf, (), ("x",))
    with pytest.raises(UnsupportedModelError, match="parameters depend"):
        context(jnp.array([]), {"x": .4}).integral("dependent_width", ("x",))


def test_distinct_nodes_with_same_name_are_rejected_across_roots(root):
    first = root.RooRealVar("duplicate", "", 1., 0., 3.)
    second = root.RooRealVar("duplicate", "", 2., 0., 3.)
    with pytest.raises(ValueError, match="Distinct graph nodes.*duplicate"):
        default_registry().export([first, second])
