"""Padding removal preserves coefficients, native equations and higher AD."""
import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.template_morphs import pack_sparse_group, validate_sparse_indices
from combine_jax.tensor.templates import morph_batch
from test_bin_statistics import template_path


def records(mode, repeat=1):
    rng = np.random.default_rng(7231)
    result = []
    for i, (f, m) in enumerate([(1, 1), (3, 2), (7, 9)]*repeat):
        base = rng.uniform(.8, 1.2, f)
        result.append(dict(source_active=f, output_size=8, indices=list(np.arange(m) % 5),
            base=base, diff=rng.normal(0, .02, (m, f)), sum=rng.normal(0, .01, (m, f)),
            region=0. if i % 3 == 0 else 1., nominal_sum=float(np.exp(base).sum()),
            projection=rng.uniform(0, 1, (8, f)), tail=rng.uniform(0, .01, 8)))
    return result


def dense_group(group):
    d = len(group); f = max(h['source_active'] for h in group); m = max(len(h['indices']) for h in group)
    output_bins = max(h['output_size'] for h in group)
    inputs = np.zeros((d, m), dtype=np.int32)
    b = dict(base=np.zeros((d, f)), diff=np.zeros((d, m, f)), sum=np.zeros((d, m, f)),
        morph_mask=np.zeros((d, m), dtype=bool), source_mask=np.zeros((d, f), dtype=bool),
        region=np.ones(d), nominal_sum=np.ones(d), projection=np.zeros((d, output_bins, f)), tail=np.zeros((d, output_bins)))
    for i, h in enumerate(group):
        f, m = h['source_active'], len(h['indices'])
        inputs[i, :m] = h['indices']; b['morph_mask'][i, :m] = True; b['source_mask'][i, :f] = True
        for field in ['base']:b[field][i, :f] = h[field]
        for field in ['diff', 'sum']:b[field][i, :m, :f] = h[field]
        for field in ['region', 'nominal_sum', 'tail']:b[field][i] = h[field]
        b['projection'][i, :, :f] = h['projection']
    return inputs, b


def objective(mode):
    def function(theta, inputs, buffers):
        result = morph_batch(mode, theta[inputs], buffers, jnp,
            segment_sum=lambda v, i, n:jax.ops.segment_sum(v, i, num_segments=n, indices_are_sorted=True))
        return jnp.sum(jnp.sin(result))
    return function


@pytest.mark.parametrize('mode', ['linear', 'log', 'raw'])
def test_sparse_coefficients_values_gradient_and_hessian(mode):
    group = records(mode)
    inputs, sparse = pack_sparse_group(group, 8)
    dense_inputs, dense = dense_group(group)
    validate_sparse_indices(sparse, len(group), len(inputs))
    assert len(sparse['diff']) == sum(h['source_active']*len(h['indices']) for h in group)
    for i, h in enumerate(group):
        first = sum(len(g['indices']) for g in group[:i])
        for j in range(len(h['indices'])):
            take = sparse['entry_morphs'] == first+j
            np.testing.assert_array_equal(sparse['diff'][take], h['diff'][j])
            np.testing.assert_array_equal(sparse['sum'][take], h['sum'][j])
    fn = objective(mode)
    v = jax.jit(fn); g = jax.jit(jax.grad(fn)); hess = jax.jit(jax.hessian(fn))
    for theta in [np.zeros(5), np.array([-.7, .4, 1.2, -1.8, .9])]:
        np.testing.assert_allclose(morph_batch(mode, theta[inputs], sparse),
                                   morph_batch(mode, theta[dense_inputs], dense), atol=1e-14, rtol=1e-14)
        for method in [v, g, hess]:
            np.testing.assert_allclose(method(theta, inputs, sparse), method(theta, dense_inputs, dense),
                                       atol=2e-13, rtol=2e-13)
    trace = jax.make_jaxpr(fn)(theta, inputs, sparse)
    assert trace.consts == []
    large_inputs, large = pack_sparse_group(records(mode, 4), 8)
    assert len(trace.jaxpr.eqns) == len(jax.make_jaxpr(fn)(theta, large_inputs, large).jaxpr.eqns)


@pytest.mark.parametrize('damage', ['wrong_histogram', 'duplicate', 'missing', 'out_of_range', 'dtype'])
def test_sparse_validation_rejects_missing_or_misbound_entries(damage):
    inputs, b = pack_sparse_group(records('linear'), 8)
    b = copy.deepcopy(b)
    if damage == 'wrong_histogram':b['morph_histograms'][0] = 2
    elif damage == 'duplicate':
        b['entry_morphs'][2] = b['entry_morphs'][1]; b['entry_bins'][2] = b['entry_bins'][1]
    elif damage == 'missing':
        for field in ['entry_morphs', 'entry_bins']:b[field] = b[field][:-1]
    elif damage == 'out_of_range':b['entry_bins'][0] = b['base'].size
    elif damage == 'dtype':b['entry_bins'] = b['entry_bins'].astype(np.float64)
    with pytest.raises(ValueError):validate_sparse_indices(b, 3, len(inputs))


def test_sparse_worker_buffers_reduce_padding_and_keep_zero_coefficients():
    group = records('linear', 20)
    for h in group:h['diff'][:] = 0.
    inputs, sparse = pack_sparse_group(group, 8)
    dense_inputs, dense = dense_group(group)
    assert sparse['diff'].size and not sparse['diff'].any()
    assert sum(a.nbytes for a in sparse.values())+inputs.nbytes < sum(a.nbytes for a in dense.values())+dense_inputs.nbytes
    fn = jax.jit(jax.value_and_grad(objective('linear')))
    sizes = []
    for idx, fields in [(inputs, sparse), (dense_inputs, dense)]:
        compiled = fn.lower(np.ones(5)*.2, idx, fields).compile()
        memory = compiled.memory_analysis()
        sizes.append(memory.argument_size_in_bytes+memory.output_size_in_bytes+memory.temp_size_in_bytes)
    assert sizes[0] < sizes[1]


def test_legacy_dense_bundle_transport_keeps_values_and_derivatives(template_path, tmp_path):
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import from_workspace, unpack_model
    from combine_jax.tensor.scalar import evaluate_numpy
    from combine_jax.tensor.templates import make_template_function, validate_templates
    with open_workspace(template_path) as w:model = from_workspace(w)
    parts = unpack_model(model); sparse = parts['templates']
    metadata = copy.deepcopy(sparse.metadata); arrays = dict(sparse.arrays)
    for batch in metadata['schedule'][:-1]:
        b = {key:arrays[name] for key, name in batch['buffers'].items()}
        sparse_inputs = arrays[batch['inputs']]; group = []
        for i in range(len(arrays[batch['outputs']])):
            f = int(b['source_mask'][i].sum()); active = b['morph_histograms'] == i
            take = b['entry_bins']//b['base'].shape[1] == i
            m = int(active.sum())
            group.append(dict(source_active=f, output_size=b['tail'].shape[1],
                indices=sparse_inputs[active].tolist(), base=b['base'][i,:f],
                diff=b['diff'][take].reshape(f,m).T, sum=b['sum'][take].reshape(f,m).T,
                region=b['region'][i], nominal_sum=b['nominal_sum'][i],
                projection=b['projection'][i,:,:f], tail=b['tail'][i]))
        inputs, fields = dense_group(group)
        for name in batch['buffers'].values():arrays.pop(name)
        prefix = batch['inputs']+'_legacy_'
        batch['buffers'] = {key:prefix+key for key in fields}
        arrays.update({batch['buffers'][key]:value for key,value in fields.items()})
        arrays[batch['inputs']] = inputs
        batch.pop('storage')  # Original dense artifacts have no storage selector.
    legacy = TensorBundle(metadata, arrays); validate_templates(legacy)
    legacy.save(tmp_path/'legacy'); legacy = TensorBundle.load(tmp_path/'legacy')
    scalars = evaluate_numpy(parts['scalars'], model.arrays['parameter_info__initial'])
    def run(bundle):
        evaluate = make_template_function(bundle)
        def loss(s, a):return jnp.sum(evaluate(s,a)[0])
        return jax.jit(jax.value_and_grad(loss))(scalars, dict(bundle.arrays))
    for actual, expected in zip(run(sparse), run(legacy)):
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
