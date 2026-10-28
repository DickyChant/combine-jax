import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.formula import parse_formula
from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize("expression", ["@0*(@1)*(@2)", "@0+2*@1-@2/(@0+3)",
    "TMath::Power(@0,2)+exp(@1)*sqrt(@2)", "(@0>0)*sin(@1)+(@0<=0)*cos(@1)",
    "TMath::Max(@0,@1)+TMath::Min(@0,@2)", "@0^2+log(@2)+abs(@1)",
    "TMath::Pi()*@0+TMath::Erf(@1)", "(@0>0 && @1<0) || @2>3"])
def test_formula_matches_root_values_and_gradients(root, expression):
    variables = [root.RooRealVar(n, "", v, -5., 5.) for n, v in zip(("a", "b", "c"), (1., -.4, 2.))]
    formula = root.RooFormulaVar("formula", "", expression, root.RooArgList(*variables))
    registry = default_registry(); nodes = registry.export([formula])
    value = jax.jit(lambda t: Evaluation(nodes, registry, ("a", "b", "c"), t).value("formula"))
    def ref(t):
        for v, p in zip(variables, t): v.setVal(float(p))
        return float(formula.getVal())
    for theta in np.random.default_rng(217).uniform([.2, -2., .3], [2., -.1, 2.7], (6, 3)):
        np.testing.assert_allclose(value(theta), ref(theta), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(jax.grad(value)(theta), finite_difference(ref, theta, np.full(3, 1e-4)),
                                   rtol=1e-7, atol=1e-10)


@pytest.mark.parametrize("expression", ["__import__('os').system('pwd')", "open('file')", "a.real", "x[@0]",
    "[a for a in x]", "lambda: 1", "@4", "unknown(@0)", "@0 ? @1", "@0 ? : @2"])
def test_unallowlisted_syntax_is_rejected(expression):
    with pytest.raises(UnsupportedModelError):
        parse_formula(expression, ("a", "b", "c"))


@pytest.mark.parametrize("expression", [
    "@0*(1+(@2>=0?0:@1*abs(@2)))",
    "@0>0 ? @1 : @2>0 ? @2 : -@1",
    "@0>0 ? (@1>0 ? @1 : @2) : -@2",
    "max(@0>0 ? @1 : @2, @2>0 ? @0 : -@1)",
    "(@0>0 ? @0 : -@0)+(@1>0 ? @1 : -@1)"])
def test_c_ternary_values_and_derivatives(root, expression):
    vs = [root.RooRealVar(n, "", .4, -4., 4.) for n in ("ta", "tb", "tc")]
    formula = root.RooFormulaVar("ternary", "", expression, root.RooArgList(*vs))
    registry = default_registry(); nodes = registry.export([formula])
    f = jax.jit(lambda t: Evaluation(nodes, registry, ("ta", "tb", "tc"), t).value("ternary"))
    def ref(t):
        for v, p in zip(vs, t, strict=True): v.setVal(float(p))
        return formula.getVal()
    for point in np.random.default_rng(7513).uniform(-2., 2., (25,3)):
        np.testing.assert_allclose(f(point), ref(point), atol=1e-13, rtol=1e-13, equal_nan=False)
        np.testing.assert_allclose(jax.grad(f)(point), finite_difference(ref, point, np.full(3,1e-5)), atol=1e-9, rtol=1e-7, equal_nan=False)


def test_ternary_inactive_branches_preserve_finite_derivatives(root):
    x = root.RooRealVar("branch_x", "", .4, -3., 3.)
    formula = root.RooFormulaVar("branch_formula", "@0>0 ? sqrt(@0) : sqrt(-@0)", root.RooArgList(x))
    registry = default_registry(); nodes = registry.export([formula])
    f = jax.jit(lambda t: Evaluation(nodes, registry, ("branch_x",), [t]).value("branch_formula"))
    point = np.array([-2., -.2, .3, 1.4])
    np.testing.assert_allclose(f(point), np.sqrt(np.abs(point)), atol=1e-14)
    expected = np.sign(point)/2/np.sqrt(np.abs(point))
    np.testing.assert_allclose(jax.jacrev(f)(point), np.diag(expected), atol=1e-14, equal_nan=False)
    np.testing.assert_allclose(jax.hessian(lambda t: f(t).sum())(point), np.diag(-.25/np.abs(point)**1.5), atol=1e-13, equal_nan=False)
    for p in point:
        x.setVal(float(p)); np.testing.assert_allclose(f(p), formula.getVal(), atol=1e-14)


def test_scalar_ternary_broadcasts_constant_branch(root):
    x = root.RooRealVar("branch_obs", "", .4, -3., 3.)
    t = root.RooRealVar("branch_theta", "", .4, -3., 3.)
    formula = root.RooFormulaVar("branch_broadcast", "@0>0 ? 0 : @1", root.RooArgList(t,x))
    registry = default_registry(); nodes = registry.export([formula], ["branch_obs"])
    data = {"branch_obs": np.array([-1., .4, 2.])}
    f = jax.jit(lambda p: Evaluation(nodes, registry, ("branch_theta",), [p], data).value("branch_broadcast"))
    np.testing.assert_array_equal(f(1.), np.zeros(3))
    np.testing.assert_array_equal(f(-1.), data["branch_obs"])


def test_numeric_prefix_names_and_scientific_literals(root):
    names = ("13TeV_lumi", "13TeV_lumi_2018", "MX")
    variables = [root.RooRealVar(n, "", v, .1, 3000.) for n, v in zip(names, (59.7, 41.5, 2000.), strict=True)]
    expression = "(1.30676e+06/MX^1-1.1315e+09/MX^2)*13TeV_lumi+13TeV_lumi_2018+@0"
    formula = root.RooFormulaVar("named_formula", "", expression, root.RooArgList(*variables))
    registry = default_registry(); nodes = registry.export([formula])
    f = jax.jit(lambda t: Evaluation(nodes, registry, names, t).value("named_formula"))
    def ref(t):
        for v, p in zip(variables, t, strict=True): v.setVal(float(p))
        return float(formula.getVal())
    for point in ([59.7, 41.5, 2000.], [60., 40., 1900.], [57., 42., 2100.]):
        point = np.asarray(point)
        np.testing.assert_allclose(f(point), ref(point), rtol=1e-14)
        np.testing.assert_allclose(jax.grad(f)(point), finite_difference(ref, point, np.array([.01,.01,.1])), rtol=1e-8)
