"""Storage-only remapping preserves ROOT values and both AD implementations."""
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.binned_channels import evaluate_binned_channels, make_binned_function
from combine_jax.tensor.density_compaction import (
    compact_density_storage, compact_model_density_storage, density_index_buffers)
from combine_jax.tensor.model import from_workspace, make_model_function, evaluate_model_numpy
from test_discrete_masks import multipdf_path
from test_tensor_binned_channels import bind_model
from test_tensor_grid_expressions import conditional_fixture


def add_dead_slots(bundle):
    """Independently embed the arena at odd indices, with unreachable sentinels."""
    arrays = dict(bundle.arrays)
    original = arrays['initial_values']
    enlarged = np.full(2*len(original)+3, -999., dtype=np.float64)
    enlarged[1:2*len(original):2] = original
    arrays['initial_values'] = enlarged
    for name in density_index_buffers(bundle):
        arrays[name] = 2*arrays[name]+1
    return TensorBundle(bundle.metadata, arrays)


@pytest.mark.parametrize('fixed', [False, True])
def test_conditional_native_values_gradient_hessian_and_transport(root, fixed, tmp_path):
    root = load_caching_nll()
    fixture = conditional_fixture(root, 'gaussian', fixed=fixed)
    model = bind_model(root, fixture['pdf'], [fixture[n] for n in ('x','y','z')], fixture['coordinates'])
    original = model['binned']
    padded = add_dead_slots(original)
    compact = compact_density_storage(padded)
    assert compact.arrays['initial_values'].size <= original.arrays['initial_values'].size
    assert compact.arrays['initial_values'].size < padded.arrays['initial_values'].size
    assert compact_density_storage(compact) is compact
    # Non-arena index domains, input data, normalization and scalar buffers stay exact.
    changed = {'initial_values', *density_index_buffers(original)}
    for name in original.arrays.keys()-changed:
        assert compact.arrays[name] is padded.arrays[name]
        np.testing.assert_array_equal(compact.arrays[name], original.arrays[name])
    compact.save(tmp_path/'compact')
    loaded = TensorBundle.load(tmp_path/'compact')
    sf = model['sf']; scalar_arrays = model['arrays']['s']
    points = [model['old'].initial, model['old'].initial+0.013]
    for mode in ('ad', 'structural', 'structural_sequential'):
        bf = make_binned_function(loaded, gradient=mode)
        arrays = jax.device_put(dict(loaded.arrays))
        def f(theta, buffers): return jnp.sum(bf(sf(theta, scalar_arrays), buffers))
        vg = jax.jit(jax.value_and_grad(f)); hessian = jax.jit(jax.hessian(f))
        for theta in points:
            value, grad = vg(theta, arrays)
            np.testing.assert_allclose(value, model['native'](theta), rtol=0, atol=1e-9)
            np.testing.assert_allclose(grad, model['old'].value_and_grad(theta)[1], rtol=1e-10, atol=1e-9)
            np.testing.assert_allclose(hessian(theta, arrays), model['old'].hessian(theta), rtol=1e-10, atol=1e-9)
            scalar_values = np.asarray(sf(theta, scalar_arrays))
            np.testing.assert_array_equal(evaluate_binned_channels(padded, scalar_values),
                                          evaluate_binned_channels(loaded, scalar_values))
    np.testing.assert_array_equal(padded.arrays['initial_values'][1:2*len(original.arrays['initial_values']):2],
                                  original.arrays['initial_values'])


@pytest.mark.parametrize('freeze_shapes', [False, True])
def test_all_discrete_choices_and_full_model_metadata(multipdf_path, freeze_shapes):
    with open_workspace(multipdf_path) as workspace:
        if freeze_shapes:
            for name in ('multi_slope','multi_c1','multi_c2'):
                workspace.var(name).setConstant(True)
        original = from_workspace(workspace, discrete_policy='profile', compact_density_storage=False)
    compact = compact_model_density_storage(original)
    assert compact.arrays['binned__initial_values'].size < original.arrays['binned__initial_values'].size
    assert compact.metadata['parameter_info'] == original.metadata['parameter_info']
    assert compact.metadata['provenance'] == original.metadata['provenance']
    assert compact_model_density_storage(compact) is compact
    theta = original.arrays['parameter_info__initial']
    funcs = [jax.jit(jax.value_and_grad(make_model_function(b, binned_gradient='structural')))
             for b in (original, compact)]
    hess = [jax.jit(jax.hessian(make_model_function(b))) for b in (original, compact)]
    for choice in (0, 1):
        buffers = [dict(b.arrays, binned__discrete_indices=np.array([choice], dtype=np.int32))
                   for b in (original, compact)]
        for point in (theta, theta+0.017):
            result = [f(point, a) for f, a in zip(funcs, buffers)]
            for a, b in zip(*result): np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(hess[0](point,buffers[0]),hess[1](point,buffers[1]),rtol=1e-12,atol=1e-12)
            np.testing.assert_array_equal(
                evaluate_model_numpy(TensorBundle(original.metadata,buffers[0]),point),
                evaluate_model_numpy(TensorBundle(compact.metadata,buffers[1]),point))


def test_empty_arena_signed_zero_padding_and_corruption(root):
    root = load_caching_nll(); fixture = conditional_fixture(root, 'gaussian', fixed=True)
    model = bind_model(root, fixture['pdf'], [fixture[n] for n in ('x','y','z')], fixture['coordinates'])
    padded = add_dead_slots(model['binned'])
    arrays = dict(padded.arrays)
    key = padded.metadata['program'][-1]['buffers']['outputs']
    # Runtime output slots are retained even though their initial value is overwritten.
    arrays['initial_values'] = arrays['initial_values'].copy()
    arrays['initial_values'][arrays[key].ravel()[0]] = -0.
    negative_zero = compact_density_storage(TensorBundle(padded.metadata, arrays))
    assert np.signbit(negative_zero.arrays['initial_values'][negative_zero.arrays[key].ravel()[0]])
    bad = dict(padded.arrays); bad[key] = np.full_like(bad[key],len(bad['initial_values']))
    with pytest.raises(ValueError, match='Invalid binned index'):
        compact_density_storage(TensorBundle(padded.metadata,bad))
    # Empty run-time program: all channel values have already folded.
    metadata = deepcopy(padded.metadata); metadata['program'] = []; metadata['nll_program'] = []
    metadata['schedule'][2]['schedule'] = []; metadata['schedule'][3]['schedule'] = []
    arrays = {k:v for k,v in padded.arrays.items() if not k.startswith(('program_', 'nll_'))}
    arrays['initial_nll'] = np.array([3.]); arrays['fast_inputs'] = arrays['fast_outputs'] = np.empty(0,dtype=np.int32)
    empty = compact_density_storage(TensorBundle(metadata, arrays))
    assert empty.arrays['initial_values'].size == 0
    scalars = np.asarray(model['sf'](model['old'].initial,model['arrays']['s']))
    np.testing.assert_array_equal(evaluate_binned_channels(empty,scalars),[3.])
    np.testing.assert_array_equal(jax.jit(make_binned_function(empty))(scalars,dict(empty.arrays)),[3.])


@pytest.mark.parametrize('fit_policy', ['workspace', 'combine_v11_default'])
def test_workspace_import_compacts_by_default_and_preserves_policy(multipdf_path, fit_policy):
    with open_workspace(multipdf_path) as workspace:
        for name in ('multi_slope', 'multi_c1', 'multi_c2'):
            workspace.var(name).setConstant(True)
        kwargs=dict(discrete_policy='profile',fit_policy=fit_policy)
        original=from_workspace(workspace,compact_density_storage=False,**kwargs)
        default=from_workspace(workspace,**kwargs)
        with pytest.raises(ValueError,match='must be boolean'):
            from_workspace(workspace,compact_density_storage='true',**kwargs)
    compact=compact_model_density_storage(original)
    assert default.metadata==compact.metadata
    assert default.arrays.keys()==compact.arrays.keys()
    for name in default.arrays:
        np.testing.assert_array_equal(default.arrays[name],compact.arrays[name])
