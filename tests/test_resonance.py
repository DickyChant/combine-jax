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
from combine_jax.resonance import root_erf


def make_doublecb(root, shifted=False):
    x = root.RooRealVar("rx", "", .2, -6., 7.)
    xp = root.RooRealVar("rxp", "", .3, -2., 3.)
    variables = [root.RooRealVar(n, "", val, lo, hi) for n, val, lo, hi in
                 [("rmean", .2, -.8, .8), ("rwidth", 1.1, .6, 1.7),
                  ("ra1", 1.3, .7, 2.), ("rn1", 2.1, 1.4, 4.),
                  ("ra2", 1.7, .6, 2.3), ("rn2", 2.5, 1.4, 5.)]]
    pdf = root.RooDoubleCB("resonance", "", *([x, xp] if shifted else [x]), *variables)
    return locals()


def make_erfexp(root):
    x = root.RooRealVar("rx", "", .4, -2., 6.)
    variables = [root.RooRealVar(n, "", val, lo, hi) for n, val, lo, hi in
                 [("rslope", -.2, -.6, -.03), ("roffset", .7, .2, 1.5), ("rwidth", 1.1, .5, 1.6)]]
    pdf = root.RooErfExpPdf("resonance", "", x, *variables)
    return locals()


@pytest.mark.parametrize("shifted,axis_name", [(False, "rx"), (True, "rx"), (True, "rxp")])
def test_doublecb_values_integrals_and_derivatives(root, shifted, axis_name):
    model = make_doublecb(root, shifted)
    pdf = model["pdf"]; axis = model["x"] if axis_name == "rx" else model["xp"]
    # Hold the optional second coordinate fixed for these class probes.
    model["x"].setConstant(); model["xp"].setConstant()
    registry = default_registry(); nodes = registry.export([pdf], [axis_name])
    names = tuple(v.GetName() for v in model["variables"])
    coordinates = np.linspace(axis.getMin(), axis.getMax(), 19)
    norm = root.RooArgSet(axis); integral = pdf.createIntegral(norm)

    def evaluate(theta):
        ctx = Evaluation(nodes, registry, names, theta, {axis_name: coordinates})
        return ctx.value("resonance"), ctx.integral("resonance", (axis_name,)), ctx.density("resonance", (axis_name,))
    evaluate, grad = jax.jit(evaluate), jax.jit(jax.jacrev(lambda t: evaluate(t)[2]))

    def reference(theta):
        for variable, value in zip(model["variables"], theta, strict=True): variable.setVal(float(value))
        raw, densities = [], []
        for coordinate in coordinates:
            axis.setVal(float(coordinate)); raw.append(pdf.getVal()); densities.append(pdf.getVal(norm))
        return np.array(raw), integral.getVal(), np.array(densities)

    points = np.random.default_rng(7511).uniform([v.getMin() for v in model["variables"]], [v.getMax() for v in model["variables"]], (20, 6))
    for point in points:
        for actual, expected in zip(evaluate(point), reference(point), strict=True):
            np.testing.assert_allclose(actual, expected, atol=3e-13, rtol=3e-13, equal_nan=False)
        fd = finite_difference(lambda t: reference(t)[2], point, np.full(6, 1e-4)).T
        np.testing.assert_allclose(grad(point), fd, atol=2e-8, rtol=3e-6, equal_nan=False)


@pytest.mark.parametrize("low,high", [(-6., -4.), (-.2, .7), (4., 7.), (-6., 7.)])
@pytest.mark.parametrize("power", [1., 1.+3e-6, 1.-3e-6, 1.+3e-5, 1.-3e-5])
def test_doublecb_tail_log_branch_and_partial_ranges(root, low, high, power):
    model = make_doublecb(root)
    x, pdf = model["x"], model["pdf"]
    x.setRange(low, high)
    for v in (model["variables"][3], model["variables"][5]):
        v.setRange(.2, 5.); v.setVal(power)
    registry = default_registry(); nodes = registry.export([pdf], ["rx"])
    # Normalisation does not require an event value for the integrated axis.
    ctx = Evaluation(nodes, registry, (), ())
    norm = root.RooArgSet(x); integral = pdf.createIntegral(norm)
    np.testing.assert_allclose(ctx.integral("resonance", ("rx",)), integral.getVal(), atol=2e-11, rtol=2e-10, equal_nan=False)


@pytest.mark.parametrize("slope,width", [(-.2, 1.1), (.3, .7), (0., 1.1), (-.2, .005), (-.2, -.1), (-.2, .01)])
def test_erfexp_value_integral_and_derivatives(root, slope, width):
    model = make_erfexp(root); x, pdf = model["x"], model["pdf"]
    vs = model["variables"]
    vs[0].setRange(-1., 1.); vs[0].setVal(slope)
    vs[2].setRange(-1., 2.); vs[2].setVal(width)
    registry = default_registry(); nodes = registry.export([pdf], ["rx"])
    names = tuple(v.GetName() for v in vs); theta = np.array([v.getVal() for v in vs])
    coordinates = np.linspace(-2., 6., 17)
    norm = root.RooArgSet(x); integral = pdf.createIntegral(norm)
    def evaluate(t):
        ctx = Evaluation(nodes, registry, names, t, {"rx": coordinates})
        return ctx.value("resonance"), ctx.integral("resonance", ("rx",)), ctx.density("resonance", ("rx",))
    def reference(t):
        for v, p in zip(vs, t, strict=True): v.setVal(float(p))
        raw, densities = [], []
        for coordinate in coordinates:
            x.setVal(float(coordinate)); raw.append(pdf.getVal()); densities.append(pdf.getVal(norm))
        return np.array(raw), integral.getVal(), np.array(densities)
    for actual, expected in zip(evaluate(theta), reference(theta), strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-8 if slope == 0 else 2e-13, rtol=2e-9 if slope == 0 else 2e-12, equal_nan=False)
    if slope and width != .01:
        grad = jax.jacrev(lambda t: evaluate(t)[2])(theta)
        fd = finite_difference(lambda t: reference(t)[2], theta, np.full(3, 1e-4)).T
        np.testing.assert_allclose(grad, fd, atol=2e-8, rtol=3e-6, equal_nan=False)
    if slope == 0:
        # The explicit v11 substitution is discontinuous. AD follows that
        # branch; a finite difference across zero is not its derivative.
        np.testing.assert_array_equal(jax.jacrev(lambda t: evaluate(t)[2])(theta)[:, 0], 0.)


@pytest.mark.parametrize("family", ["doublecb", "doublecb_signed", "erfexp"])
def test_resonance_extended_nll(root, family):
    root = load_caching_nll()
    model = make_doublecb(root) if family.startswith("doublecb") else make_erfexp(root)
    if family == "doublecb_signed":
        alpha, power = model["variables"][2:4]
        alpha.setRange(-1., -.8); alpha.setVal(-.9)
        power.setRange(3., 6.); power.setVal(5.)
    x, shape = model["x"], model["pdf"]
    rate = root.RooRealVar("rrate", "", 20., 10., 30.)
    pdf = root.RooAddPdf("resonance_sum", "", root.RooArgList(shape), root.RooArgList(rate))
    weight = root.RooRealVar("rweight", "", 1., 0., 100.)
    collection = root.RooArgSet(x, weight)
    data = root.RooDataSet("resonance_data", "", collection, root.RooFit.WeightVar(weight))
    for coordinate, w in zip(np.linspace(.1, 4., 5), [2., 0., 6.5, 3., 1.2]):
        x.setVal(float(coordinate)); data.add(collection, w)
    registry = default_registry(); nodes = registry.export([pdf], ["rx"])
    pars = parameters(pdf, ["rx"])
    likelihood = Likelihood(nodes, registry, pars,
                            [Channel("res", "resonance_sum", ("rx",), bind_data(data, ["rx"]))], [])
    reference = root.cacheutils.CachingAddNLL("res_ref", "", pdf, data, True)
    reference.clearConstantZeroPoint(); reference.clearZeroPoint()
    variables = {v.GetName(): v for v in [*model["variables"], rate]}
    def ref(theta):
        for name, val in zip(likelihood.parameter_names, theta, strict=True): variables[name].setVal(float(val))
        return reference.getVal()
    for index, theta in enumerate(np.random.default_rng(7512).uniform([p.minimum for p in pars], [p.maximum for p in pars], (25, len(pars)))):
        actual, grad = likelihood.value_and_grad(theta)
        np.testing.assert_allclose(actual, ref(theta), atol=1e-6, rtol=0, equal_nan=False)
        fd = finite_difference(ref, theta, np.full(len(pars), 1e-4))
        assert np.max(np.abs(grad-fd)/np.maximum(np.abs(fd), 1e-8)) < 1e-4
        if index == 0:
            fd_hessian = finite_difference(lambda p: likelihood.value_and_grad(p)[1], theta, np.full(len(pars), 1e-4)).T
            np.testing.assert_allclose(likelihood.hessian(theta), fd_hessian, atol=1e-6, rtol=2e-6, equal_nan=False)


@pytest.mark.parametrize("family", ["doublecb", "erfexp"])
def test_resonance_dependent_parameter_integral_rejected(root, family):
    x = root.RooRealVar("dependent_x", "", .2, -1., 1.)
    width = root.RooFormulaVar("dependent_width", "2+@0", root.RooArgList(x))
    one = root.RooConstVar("res_one", "", 1.)
    pdf = (root.RooDoubleCB("dependent_shape", "", x, one, width, one, one, one, one) if family == "doublecb"
           else root.RooErfExpPdf("dependent_shape", "", x, one, one, width))
    registry = default_registry(); nodes = registry.export([pdf])
    with pytest.raises(UnsupportedModelError, match="depend on integration variable"):
        Evaluation(nodes, registry, (), ()).integral("dependent_shape", ("dependent_x",))


def test_doublecb_nonpositive_shape_parameters_mark_invalid(root):
    model = make_doublecb(root)
    registry = default_registry(); nodes = registry.export([model["pdf"]])
    for index, variable in enumerate(model["variables"][1:], 1):
        context = Evaluation(nodes, registry, (variable.GetName(),), [0. if index in (2,4) else -.1])
        assert np.isnan(context.value("resonance"))
        assert np.isnan(context.integral("resonance", ("rx",)))
    model["variables"][1].setRange(-1., 2.); model["variables"][1].setVal(-.1)
    with pytest.raises(UnsupportedModelError, match="requires positive"):
        registry.export([model["pdf"]])


@pytest.mark.parametrize("alpha1,alpha2", [(-.953105072, .92994867), (1.2, -.7), (-.8, -.6)])
def test_doublecb_signed_alpha_value_and_advertised_integral(root, alpha1, alpha2):
    model = make_doublecb(root)
    for index, value in ((2,alpha1),(3,5.),(4,alpha2),(5,4.)):
        v=model["variables"][index]; v.setRange(-2., 7.); v.setVal(value)
    x,pdf=model["x"],model["pdf"]
    registry=default_registry(); nodes=registry.export([pdf],["rx"])
    coordinates=np.linspace(-6.,7.,31)
    norm=root.RooArgSet(x); integral=pdf.createIntegral(norm)
    ctx=Evaluation(nodes,registry,(),(),{"rx":coordinates})
    expected=[]
    for v in coordinates:
        x.setVal(float(v));expected.append(pdf.getVal(norm))
    np.testing.assert_allclose(ctx.density("resonance",("rx",)),expected,atol=1e-13,rtol=1e-12,equal_nan=False)
    np.testing.assert_allclose(ctx.integral("resonance",("rx",)),integral.getVal(),atol=1e-12,rtol=1e-12,equal_nan=False)


def test_root_error_function_tail_rounding(root):
    values = np.r_[np.linspace(-6., -1.001, 1001), np.linspace(1.001, 6., 1001)]
    expected = np.array([root.TMath.Erf(float(v)) for v in values])
    np.testing.assert_allclose(root_erf(values), expected, atol=2*np.finfo(float).eps, rtol=0)
    np.testing.assert_allclose(jax.jit(root_erf)(values), expected, atol=2*np.finfo(float).eps, rtol=0)
    # These negative-tail points expose ROOT's double subtraction, distinct
    # from a direct correctly rounded erf. Preserve it exactly under JIT.
    points = np.array([-3.689, -4.2, -4.6, -5.1])
    np.testing.assert_array_equal(jax.jit(root_erf)(points), [root.TMath.Erf(float(v)) for v in points])


def test_erfexp_remote_turn_on(root):
    x = root.RooRealVar("remote_x", "", 100., 30., 210.)
    vs = [root.RooConstVar(f"remote_{i}", "", v) for i,v in enumerate([-.08334598692980426, 393.65912228443176, 78.30276625993297])]
    pdf = root.RooErfExpPdf("remote_erfexp", "", x, *vs)
    registry = default_registry(); nodes = registry.export([pdf], ["remote_x"])
    coords = np.linspace(30., 210., 21)
    def evaluate():
        ctx = Evaluation(nodes, registry, (), (), {"remote_x": coords})
        return ctx.density("remote_erfexp", ("remote_x",)), ctx.integral("remote_erfexp", ("remote_x",))
    norm = root.RooArgSet(x); integral = pdf.createIntegral(norm)
    expected = []
    for value in coords:
        x.setVal(float(value)); expected.append(pdf.getVal(norm))
    for density, norm_value in (evaluate(), jax.jit(evaluate)()):
        np.testing.assert_allclose(norm_value, integral.getVal(), atol=1e-20, rtol=1e-8, equal_nan=False)
        np.testing.assert_allclose(density, expected, atol=1e-12, rtol=1e-8, equal_nan=False)
