from array import array

import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.splines import ALGORITHMS
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_spline(root, algorithm):
    knots = np.array([0., .12345678901234567, .7, 1.5, 2.2, 3.7, 5.])
    values = np.array([.2, 1.4, .6, -.1, .9, .4, .2])
    x = root.RooRealVar("spline_x", "", 2., 0., 5.)
    spline = root.RooSpline1D("spline", "", x, len(knots), array("d", knots), array("d", values), algorithm)
    interpolator = root.Math.Interpolator(knots.tolist(), values.tolist(), getattr(root.Math.Interpolation, "k"+algorithm))
    return x, spline, interpolator, knots, values


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_spline_values_derivatives_and_integral(root, algorithm):
    x, spline, reference, knots, stored_values = make_spline(root, algorithm)
    registry = default_registry()
    nodes = registry.export([spline])
    np.testing.assert_array_equal(nodes["spline"].attrs["knots"], knots)

    def value(t):
        return Evaluation(nodes, registry, ("spline_x",), [t]).value("spline")

    value, gradient, hessian = jax.jit(value), jax.jit(jax.grad(value)), jax.jit(jax.hessian(value))
    points = np.r_[knots, np.random.default_rng(6201).uniform(0., 5., 25)]
    for coordinate in points:
        x.setVal(float(coordinate))
        np.testing.assert_allclose(value(coordinate), spline.getVal(), atol=4e-12, rtol=4e-12, equal_nan=False)
        np.testing.assert_allclose(gradient(coordinate), reference.Deriv(float(coordinate)), atol=3e-11, rtol=3e-11, equal_nan=False)
        np.testing.assert_allclose(hessian(coordinate), reference.Deriv2(float(coordinate)), atol=2e-10, rtol=2e-10, equal_nan=False)
    for coordinate in points[len(knots):]:
        def ref(t):
            x.setVal(float(t[0])); return spline.getVal()
        fd = finite_difference(ref, np.array([coordinate]), np.array([1e-5]))[0]
        np.testing.assert_allclose(gradient(coordinate), fd, atol=2e-8, rtol=2e-7, equal_nan=False)
    for low, high in [(0., 5.), (.2, 3.1), (1.7, 1.9)]:
        x.setRange(low, high)
        exported = registry.export([spline])
        integral = Evaluation(exported, registry, (), ()).integral("spline", ("spline_x",))
        np.testing.assert_allclose(integral, reference.Integ(low, high), atol=1e-11, rtol=1e-11, equal_nan=False)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_spline_domain_is_not_extrapolated(root, algorithm):
    x, spline, reference, knots, stored_values = make_spline(root, algorithm)
    x.setRange(-1., 6.)
    registry = default_registry()
    nodes = registry.export([spline])
    for coordinate in (-.1, 5.1):
        x.setVal(coordinate)
        with pytest.warns(RuntimeWarning, match="input domain error"):
            assert np.isnan(spline.getVal())
        assert np.isnan(Evaluation(nodes, registry, ("spline_x",), [coordinate]).value("spline"))
    with pytest.raises(UnsupportedModelError, match="outside the knot domain"):
        Evaluation(nodes, registry, (), ()).integral("spline", ("spline_x",))


@pytest.mark.parametrize("knots,values,algorithm", [
    ([0., 1., 1.], [1., 2., 3.], "CSPLINE"),
    ([0., 1., 2.], [1., float("inf"), 3.], "CSPLINE"),
    ([0., 1.], [1., 2.], "AKIMA"),
    ([0., 1., 2.], [1., 2., 3.], "UNKNOWN")])
def test_invalid_spline_tables_are_named(root, knots, values, algorithm):
    x = root.RooRealVar("bad_x", "", .3, 0., 2.)
    spline = root.RooSpline1D("bad_spline", "", x, len(knots), array("d", knots), array("d", values), algorithm)
    with pytest.raises(UnsupportedModelError, match="bad_spline"):
        default_registry().export([spline])


def test_spline_composes_with_formula_arguments(root):
    theta = root.RooRealVar("spline_theta", "", 0., -1., 1.)
    argument = root.RooFormulaVar("spline_argument", "2+2*@0", root.RooArgList(theta))
    spline = root.RooSpline1D("composed_spline", "", argument, 4,
                             array("d", [0., 1., 3., 5.]), array("d", [1., 2., 0., 1.]), "CSPLINE")
    registry = default_registry()
    nodes = registry.export([spline])
    f = jax.jit(lambda t: Evaluation(nodes, registry, ("spline_theta",), [t]).value("composed_spline"))
    for t in np.linspace(-.9, .9, 15):
        theta.setVal(float(t))
        np.testing.assert_allclose(f(t), spline.getVal(), atol=2e-14, rtol=2e-14)
        def reference(point):
            theta.setVal(float(point[0])); return spline.getVal()
        fd = finite_difference(reference, np.array([t]), np.array([1e-5]))[0]
        np.testing.assert_allclose(jax.grad(f)(t), fd, atol=1e-9, rtol=2e-7)
