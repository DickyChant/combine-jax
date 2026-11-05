"""Keep scalar RooFit slopes and native optimized-cache leaf slopes distinct."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.likelihood import from_workspace as oracle_import
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.model import from_workspace, evaluate_model_numpy, make_model_function
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from combine_jax.graph import UnsupportedModelError


def fixture(root, kind):
    w = root.RooWorkspace('w')
    for expression in ('x[130,110,150]', 'p[-.0354,-.08,.08]', 'rate[94,30,160]', 'q[.2,.1,.3]'):
        w.factory(expression)
    two = kind in ('two_parameters', 'constant_parameter')
    formula = '-@0*@0*@1' if two else '-@0*@0'
    args = root.RooArgList(w.var('p'))
    if two: args.add(w.var('q'))
    if kind == 'constant_parameter': w.var('q').setConstant(True)
    coefficient = root.RooFormulaVar('coefficient', formula, args)
    shape = root.RooExponential('shape', '', w.var('x'), coefficient, kind == 'negated')
    direct_shape = shape
    if kind == 'nested':
        uniform = root.RooUniform('uniform', '', root.RooArgSet(w.var('x')))
        fraction = root.RooRealVar('fraction', '', .8); fraction.setConstant(True)
        direct_shape = root.RooAddPdf('nested', '', root.RooArgList(shape, uniform), root.RooArgList(fraction))
    wrapped = root.RooProdPdf('wrapped', '', root.RooArgList(direct_shape))
    if kind == 'multipdf':
        index = root.RooCategory('index', '')
        uniform = root.RooUniform('alternative', '', root.RooArgSet(w.var('x')))
        wrapped = root.RooMultiPdf('wrapped', '', index, root.RooArgList(shape, uniform))
    category = root.RooCategory('CMS_channel', '')
    category.defineType('direct'); category.defineType('generic')
    a = root.RooAddPdf('direct_sum', '', root.RooArgList(direct_shape), root.RooArgList(w.var('rate')))
    b = root.RooAddPdf('generic_sum', '', root.RooArgList(wrapped), root.RooArgList(w.var('rate')))
    pdf = root.RooSimultaneous('model', '', category); pdf.addPdf(a, 'direct'); pdf.addPdf(b, 'generic')
    getattr(w, 'import')(pdf, root.RooFit.RecycleConflictNodes(), root.RooFit.Silence())
    weight = root.RooRealVar('weight', '', 1., 0., 100.)
    columns = root.RooArgSet(w.var('x'), w.cat('CMS_channel'), weight)
    data = root.RooDataSet('toy_asimov', '', columns, root.RooFit.WeightVar(weight))
    grid = np.linspace(110.25, 149.75, 80)
    weights = 94*np.exp(-.0354*grid)/np.exp(-.0354*grid).sum()
    for label in ('direct', 'generic'):
        w.cat('CMS_channel').setLabel(label)
        for x, count in zip(grid, weights, strict=True):
            w.var('x').setVal(float(x)); data.add(columns, float(count))
    getattr(w, 'import')(data)
    mc = root.RooStats.ModelConfig('ModelConfig', w); mc.SetPdf(w.pdf('model'))
    mc.SetObservables(root.RooArgSet(w.var('x'), w.cat('CMS_channel')))
    mc.SetParametersOfInterest(root.RooArgSet(w.var('rate')))
    mc.SetNuisanceParameters(root.RooArgSet(w.var('p'), w.var('q')))
    getattr(w, 'import')(mc)
    return w


@pytest.mark.parametrize('kind', ['formula', 'nested', 'multipdf', 'two_parameters', 'constant_parameter', 'negated'])
def test_cached_leaf_and_generic_formula_have_distinct_native_semantics(root, kind):
    root = load_caching_nll(); w = fixture(root, kind)
    old = oracle_import(w, data='toy_asimov', analytic_barlow_beeston=False)
    if kind == 'negated':
        # Keep the existing tensor guard for the generic negative-sign route;
        # this correction only lowers the native optimized-cache route.
        assert len(old.cache_routes) == 1 and old.cache_routes[0]['coefficient_sign'] == -1.
        with pytest.raises(UnsupportedModelError, match='negated exponential'):
            from_workspace(w, data='toy_asimov', analytic_barlow_beeston=False)
        return
    model = from_workspace(w, data='toy_asimov', analytic_barlow_beeston=False)
    routes = model.metadata['provenance']['native_cache_routes']
    assert len(routes) == (0 if kind in ('two_parameters','constant_parameter') else 1)
    assert old.nodes['shape'].inputs[1] == 'coefficient'
    if routes:
        assert routes[0]['cached_parameter'] == 'p'
        assert old.nodes[routes[0]['alias']].inputs == ('x', 'p')
    native = root.cacheutils.CachingSimNLL(w.pdf('model'), w.data('toy_asimov'), root.RooArgSet())
    native.clearZeroPoint(); native.clearConstantZeroPoint()
    def reference(theta):
        for name, value in zip(old.parameter_names, theta, strict=True): w.var(name).setVal(float(value))
        return float(native.getVal())
    arrays = jax.device_put(dict(model.arrays))
    function = make_model_function(model)
    vg = jax.jit(jax.value_and_grad(function))
    for value in (-.0354, -.015, .02):
        point = old.initial.copy(); point[old.parameter_names.index('p')] = value
        expected = reference(point)
        actual, gradient = vg(point, arrays)
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0.)
        np.testing.assert_allclose(evaluate_model_numpy(model, point), expected, atol=1e-9, rtol=0.)
        np.testing.assert_allclose(old._value(point), expected, atol=1e-9, rtol=0.)
        derivative = finite_difference(reference, point, np.full(len(point), 1e-5))
        np.testing.assert_allclose(gradient, derivative, atol=1e-7, rtol=1e-5)
    # Direct registry evaluation retains RooFit's actual formula coefficient.
    registry = default_registry(); nodes = registry.export([w.pdf('shape')], ['x'])
    point = old.initial.copy(); reference(point); w.var('x').setVal(125.)
    context = Evaluation(nodes, registry, old.parameter_names, jnp.asarray(point), {'x':jnp.asarray(125.)})
    np.testing.assert_allclose(context.density('shape', ('x',)),
        w.pdf('shape').getVal(root.RooArgSet(w.var('x'))), atol=1e-13, rtol=1e-12)
