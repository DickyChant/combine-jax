import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters, walk
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.templates import pack_templates, make_template_function, evaluate_templates
from test_hist_sum import make_sum
from test_tensor_likelihood_terms import scalar_export


@pytest.fixture(scope='module')
def hist_sum_workspace(tmp_path_factory):
    source = Path(__file__).resolve().parents[1]/'scripts/build_template_fixture.py'
    spec = importlib.util.spec_from_file_location('sum_fixture_builder', source)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module.build_template_fixture(tmp_path_factory.mktemp('hist_sum'), use_hist_sum=True)


@pytest.mark.parametrize('log,threshold,active,freeze', [
    (False, 10., False, False), (True, 10., False, False),
    (True, 0., True, True), (False, None, True, True),
])
def test_hist_sum_tensor_native_nll_ad_and_transport(root, tmp_path, log, threshold, active, freeze):
    root = load_caching_nll()
    fixture = make_sum(root, log=log, threshold=threshold, active=active)
    raw = walk(fixture['pdf'])
    if freeze:
        for name in ('sum_m0', 'sum_m1', 'sum_m2'):
            raw[name].setVal(.23); raw[name].setConstant(True)
    registry = default_registry(); graph = registry.export([fixture['pdf']], ['sum_x'])
    nodes = {n:(v.class_name, v.inputs, dict(v.attrs)) for n,v in graph.items()}
    info = parameters(fixture['pdf'], ['sum_x'])
    binding = bind_data(fixture['data'], ['sum_x'])
    old = Likelihood(graph, registry, info, [Channel('sum', 'sum_pdf', ('sum_x',), binding)], [])
    scalars = scalar_export(nodes, info, lambda n:float(raw[n].getVal()), observables=['sum_x'])
    rows = [dict(name='sum', pdf='sum_pdf', observables=('sum_x',),
                 coordinates=dict(binding.coordinates), weights=binding.weights)]
    packed = pack_templates(nodes, rows, scalars)
    packed.save(tmp_path/'templates'); packed = TensorBundle.load(tmp_path/'templates')
    sf = make_function(scalars); tf = make_template_function(packed)
    arrays = jax.device_put(dict(s=dict(scalars.arrays), t=dict(packed.arrays)))
    def fn(theta, buffers):
        return tf(sf(theta, buffers['s']), buffers['t'])[0][0]
    value = jax.jit(fn); gradient = jax.jit(jax.grad(fn)); hessian = jax.jit(jax.hessian(fn))
    native = root.cacheutils.CachingAddNLL('tensor_sum_reference', '', fixture['pdf'], fixture['data'], True)
    native.clearConstantZeroPoint(); native.clearZeroPoint()
    for shift in (0., -.63, .41):
        point = old.initial.copy()
        for i, name in enumerate(old.parameter_names):
            if name.startswith('sum_m'): point[i] = shift
            elif name.startswith('sum_rate'): point[i] = 1.+shift/3
            elif point[i] == 0: point[i] = shift/10
            else: point[i] *= 1.+shift/10
        for n, v in zip(old.parameter_names, point, strict=True): raw[n].setVal(float(v))
        np.testing.assert_allclose(value(point, arrays), native.getVal(), rtol=0., atol=1e-10)
        np.testing.assert_allclose(gradient(point, arrays), old.value_and_grad(point)[1], rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(hessian(point, arrays), old.hessian(point), rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(evaluate_templates(packed, np.asarray(sf(point, arrays['s'])))[0], [native.getVal()], atol=1e-10)
    assert jax.make_jaxpr(fn)(point, arrays).consts == []
    if freeze: assert not packed.metadata['schedule'][:-1]
    if threshold == 10.: assert 'poisson_histogram_indices' in packed.arrays
    else: assert 'poisson_histogram_indices' not in packed.arrays
    # A larger channel batch must not unroll extra graph operations.
    many = pack_templates(nodes, [{**rows[0], 'name':'replica'+str(i)} for i in range(9)], scalars)
    sv = np.asarray(sf(point, arrays['s']))
    traces = [jax.make_jaxpr(make_template_function(t))(sv, jax.device_put(dict(t.arrays)))
              for t in (packed, many)]
    assert len(traces[0].jaxpr.eqns) == len(traces[1].jaxpr.eqns)
    np.testing.assert_allclose(evaluate_templates(many, sv)[0], native.getVal(), rtol=0., atol=1e-10)
    del native


@pytest.mark.parametrize('profile,global_value,frozen', [
    (True, 0., False), (False, 0., False), (True, .8, False), (True, .8, True),
])
def test_hist_sum_complete_profiled_model(hist_sum_workspace, tmp_path, profile, global_value, frozen):
    # Reuse the native acceptance harness, including ROOT finite differences,
    # profiled Hessian and NumPy-only transport, with the CMSHistSum card option.
    from test_tensor_templates import test_complete_automcstats_native_nll_gradient_hessian as compare
    compare(hist_sum_workspace, tmp_path, profile, global_value, frozen)


@pytest.mark.parametrize('profile', [False, True])
def test_hist_sum_standard_26_point_native_gates(hist_sum_workspace, profile):
    from combine_jax.root_io import open_workspace
    from combine_jax.validation import validate_workspace
    with open_workspace(hist_sum_workspace) as workspace:
        assert workspace.function('prop_binch').ClassName() == 'CMSHistSum'
        report = validate_workspace(workspace, random_points=25, seed=1819, analytic_barlow_beeston=profile)
    assert report['max_nll_absolute_error'] < 1e-6
    assert report['max_channel_absolute_error'] < 1e-6
    assert report['max_gradient_relative_error'] < 1e-4
    assert report['max_hessian_absolute_error'] < 1e-6
    assert bool(report['profiled_parameters']) == profile
