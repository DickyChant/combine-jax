"""Native NLL and streamed structural AD for floating and morphed 1D bins."""
import subprocess
import sys

import jax
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.binned_channels import make_binned_function, pack_binned_channels, evaluate_binned_channels
from combine_jax.tensor.scalar import evaluate_numpy
from test_parametric_hist import make_hist
from test_tensor_binned_channels import bind_model, check_model


@pytest.mark.parametrize('region', [None, 0., .7, 'empty'])
def test_parametric_hist_tensor_nll_structural_gradient_hvp_and_transport(root, tmp_path, region):
    root = load_caching_nll()
    fixture = make_hist(root, region)
    model = bind_model(root, fixture['pdf'], [fixture['x']], fixture['coordinates'])
    check_model(model)
    # Duplicate channels exercise the sequential structural pullback too.
    rows = [{**model['rows'][0], 'name': 'replica'+str(i)} for i in range(3)]
    bundle = pack_binned_channels(model['nodes'], rows, model['scalar'])
    arrays = jax.device_put(dict(bundle.arrays))
    scalar = model['sf']
    def f(theta, policy):
        return make_binned_function(bundle, gradient=policy, row_chunk_size=3)(scalar(theta, model['arrays']['s']), arrays).sum()
    theta = model['old'].initial
    direction = np.random.default_rng(41).normal(size=len(theta))
    reference = jax.jit(jax.value_and_grad(lambda t: f(t, 'ad')))(theta)
    refhvp = jax.jit(lambda t: jax.jvp(jax.grad(lambda v: f(v, 'ad')), (t,), (direction,))[1])(theta)
    for policy in ('structural', 'structural_sequential'):
        function = lambda t: f(t, policy)
        actual = jax.jit(jax.value_and_grad(function))(theta)
        for a, e in zip(actual, reference, strict=True):
            np.testing.assert_allclose(a, e, rtol=1e-11, atol=1e-11)
        hvp = jax.jit(lambda t: jax.jvp(jax.grad(function), (t,), (direction,))[1])(theta)
        np.testing.assert_allclose(hvp, refhvp, rtol=1e-10, atol=1e-10)
    model['scalar'].save(tmp_path/'scalar')
    bundle.save(tmp_path/'binned')
    script = '''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalar');b=TensorBundle.load(p/'binned')
assert np.all(np.isfinite(evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable, '-c', script, str(tmp_path)], check=True, capture_output=True)


def test_parametric_hist_padded_morph_counts_share_a_batch(root):
    from test_tensor_likelihood_terms import scalar_export
    root = load_caching_nll()
    fixtures = [make_hist(root, region) for region in (None, .7)]
    models = []
    for i, fixture in enumerate(fixtures):
        fixture['pdf'].SetName(f'ph{i}')
        fixture['x'].SetName(f'phx{i}')
        for variable in fixture['variables']:
            variable.SetName(f'{variable.GetName()}_{i}')
        models.append(bind_model(root, fixture['pdf'], [fixture['x']],
                                 {f'phx{i}': fixture['coordinates']['phx']}))
    nodes = {n: v for m in models for n, v in m['nodes'].items() if n != 'test_sum'}
    for i, m in enumerate(models):
        nodes[f'sum{i}'] = m['nodes']['test_sum']
    info = {p.name: p for m in models for p in m['info']}
    raw = {n: v for m in models for n, v in m['raw'].items()}
    scalars = scalar_export(nodes, list(info.values()), lambda n: float(raw[n].getVal()), observables=['phx0','phx1'])
    rows = [{**m['rows'][0], 'name': f'channel{i}', 'pdf': f'sum{i}'} for i, m in enumerate(models)]
    bundle = pack_binned_channels(nodes, rows, scalars)
    sums = [b for b in bundle.metadata['program'] if b['operation'] == 'parametric1d_sum']
    assert len(sums) == 1 and bundle.arrays[sums[0]['buffers']['outputs']].shape[0] == 2
    values = evaluate_numpy(scalars, scalars.arrays['parameter_initial'])
    expected = [m['native'](m['old'].initial) for m in models]
    np.testing.assert_allclose(evaluate_binned_channels(bundle, values), expected, rtol=0., atol=1e-9)
    actual = jax.jit(make_binned_function(bundle))(values, jax.device_put(dict(bundle.arrays)))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=1e-9)
