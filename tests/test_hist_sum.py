from array import array

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_sum(root, *, log=False, threshold=10., active=False, region=.7):
    edges = [0., .5, 2., 4., 7., 9.]
    x = root.RooRealVar('sum_x', '', 1., 0., 9.)
    x.setBinning(root.RooBinning(5, array('d', edges)))
    morphs = [root.RooRealVar('sum_m'+str(i), '', 0., -3., 3.) for i in range(3)]
    rates = [root.RooRealVar('sum_rate'+str(i), '', 1., .1, 3.) for i in range(2)]
    values = [[6., 3., 2., 1., 0.], [30., 3., .1, 8., 0.]]
    errors = [[.6, 0., .3, 0., 0.], [3., 2., 1., 0., 0.]]
    held = [x, *morphs, *rates]
    funcs = []
    for p in range(2):
        hist = root.TH1D('sum_hist'+str(p), '', 5, array('d', edges)); hist.SetDirectory(0)
        for b, (v, e) in enumerate(zip(values[p], errors[p], strict=True), 1):
            hist.SetBinContent(b, v); hist.SetBinError(b, e)
        func = root.CMSHistFunc('sum_func'+str(p), '', x, hist, True)
        local = [morphs[p], morphs[2]]
        func.setVerticalMorphs(root.RooArgList(*local))
        func.setVerticalSmoothRegion(region)
        if log and p == 1:
            func.setVerticalType(root.CMSHistFunc.LogQuadLinear)
        func.prepareStorage()
        for m in range(2):
            for direction in (0, 1):
                h = hist.Clone('sum_shape_%s_%s_%s' % (p, m, direction)); h.SetDirectory(0)
                for b, v in enumerate(values[p], 1):
                    factor = 1.+(-.14 if direction == 0 else .19)*(m+1)*(b%3+1)
                    h.SetBinContent(b, v*factor)
                func.setShape(0, 0, m+1, direction, h); held.append(h)
        if active:
            func.setActiveBins(4)
        funcs.append(func); held.extend([hist, func])
    summed = root.CMSHistSum('hist_sum', '', x, root.RooArgList(*funcs), root.RooArgList(*rates))
    stats = summed.setupBinPars(threshold) if threshold is not None else root.RooArgList()
    one = root.RooConstVar('sum_one', '', 1.)
    pdf = root.RooRealSumPdf('sum_pdf', '', root.RooArgList(summed), root.RooArgList(one), True)
    weight = root.RooRealVar('sum_weight', '', 1., 0., 100.)
    variables = root.RooArgSet(x, weight)
    data = root.RooDataSet('sum_asimov', '', variables, root.RooFit.WeightVar(weight))
    for b in range(5):
        x.setVal((edges[b]+edges[b+1])/2)
        data.add(variables, values[0][b]+values[1][b])
    held.extend([summed, stats, one, pdf, weight, variables, data])
    return dict(x=x, sum=summed, pdf=pdf, data=data, stats=stats, held=held)


def parameter_setup(fixture):
    params = parameters(fixture['pdf'], ['sum_x'])
    names = tuple(p.name for p in params)
    variables = fixture['pdf'].getVariables()
    objects = [variables.find(n) for n in names]
    initial = np.array([p.value for p in params])
    return params, names, objects, initial, variables


@pytest.mark.parametrize('log,threshold,active,region', [
    (False, None, False, .7), (True, None, True, .7),
    (False, 10., False, .7), (True, 10., False, .7),
    (False, 0., True, .7), (True, 0., False, 0.),
])
def test_hist_sum_values_integral_and_derivatives(root, log, threshold, active, region):
    fixture = make_sum(root, log=log, threshold=threshold, active=active, region=region)
    params, names, objects, initial, variables = parameter_setup(fixture)
    registry = default_registry(); graph = registry.export([fixture['pdf']], ['sum_x'])
    attrs = graph['hist_sum'].attrs
    if threshold == 10.:
        assert {t for row in attrs['bin_types'] for t in row} == {0, 1, 2, 3, 4}
    coords = np.array([0., .25, .5, .500001, 1.25, 2., 3., 4., 5.5, 7., 8., 9.])
    integral = fixture['sum'].createIntegral(root.RooArgSet(fixture['x']))
    def function(theta):
        context = Evaluation(graph, registry, names, theta, {'sum_x': coords})
        return jnp.concatenate((context.value('hist_sum'), jnp.atleast_1d(context.integral('hist_sum', ('sum_x',)))))
    function = jax.jit(function); gradient = jax.jit(jax.jacrev(function))
    def reference(theta):
        for p, v in zip(objects, theta, strict=True): p.setVal(float(v))
        out = []
        for coord in coords:
            fixture['x'].setVal(float(coord)); out.append(fixture['sum'].getVal())
        return np.array(out+[integral.getVal()])
    points = [initial]
    rng = np.random.default_rng(9021)
    for _ in range(25):
        point = initial.copy()
        for i, n in enumerate(names):
            if n.startswith('sum_m'): point[i] = rng.uniform(-1.7, 1.7)
            elif n.startswith('sum_rate'): point[i] = rng.uniform(.5, 2.)
            elif initial[i] == 0: point[i] = rng.uniform(-.1, .1)
            else: point[i] *= rng.uniform(.85, 1.15)
        points.append(point)
    for point in points:
        np.testing.assert_allclose(function(point), reference(point), rtol=3e-13, atol=3e-13)
        # Avoid region=0's genuine derivative discontinuity at theta=0.
        if region == 0 and np.array_equal(point, initial): continue
        columns = []
        for i in range(len(point)):
            step = np.zeros_like(point); step[i] = 1e-4
            columns.append((reference(point-2*step)-8*reference(point-step)+8*reference(point+step)-reference(point+2*step))/(12e-4))
        np.testing.assert_allclose(gradient(point), np.array(columns).T, rtol=2e-6, atol=2e-8)


@pytest.mark.parametrize('log', [False, True])
def test_hist_sum_serialized_native_nll(root, tmp_path, log):
    root = load_caching_nll()
    fixture = make_sum(root, log=log)
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(fixture['pdf'])
    getattr(workspace, 'import')(fixture['data'])
    path = tmp_path/'sum.root'; workspace.writeToFile(str(path))
    with open_workspace(path) as loaded:
        pdf, data = loaded.pdf('sum_pdf'), loaded.data('sum_asimov')
        registry = default_registry(); graph = registry.export([pdf], ['sum_x'])
        likelihood = Likelihood(graph, registry, parameters(pdf, ['sum_x']),
            [Channel('sum', 'sum_pdf', ('sum_x',), bind_data(data, ['sum_x']))], [])
        native = root.cacheutils.CachingAddNLL('sum_reference', '', pdf, data, True)
        native.clearConstantZeroPoint(); native.clearZeroPoint()
        def reference(theta):
            for name, value in zip(likelihood.parameter_names, theta, strict=True): loaded.var(name).setVal(float(value))
            return float(native.getVal())
        points = [likelihood.initial.copy()]
        for shift in (-.61, .43):
            point = likelihood.initial.copy()
            for i, n in enumerate(likelihood.parameter_names):
                if n.startswith('sum_m'): point[i] = shift
                elif n.startswith('sum_rate'): point[i] = 1.+shift/3
            points.append(point)
        for point in points:
            value, grad = likelihood.value_and_grad(point)
            np.testing.assert_allclose(value, reference(point), rtol=0., atol=1e-6)
            fd = finite_difference(reference, point, np.full(len(point), 1e-4))
            np.testing.assert_allclose(grad, fd, rtol=1e-4, atol=1e-8)
        del native
