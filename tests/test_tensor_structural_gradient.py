"""Streamed scores retain native gates, shared shape paths and exact HVPs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.binned_channels import make_binned_function, pack_binned_channels
from combine_jax.root_io import load_caching_nll
import test_tensor_binned_channels as cases
from test_tensor_analytic_shapes import shape_fixture


@pytest.fixture(params=['structural', 'structural_sequential'])
def structural(monkeypatch, request):
    original = cases.make_binned_function
    def factory(bundle):
        return original(bundle, gradient=request.param, row_chunk_size=3)
    monkeypatch.setattr(cases, 'make_binned_function', factory)


@pytest.mark.parametrize('family', ['step', 'hist2d', 'step_floor'])
def test_parametric_native(root, tmp_path, structural, family):
    cases.test_parametric_bin_channel_native_nll_ad_and_serialization(root, family, tmp_path)


@pytest.mark.parametrize('kind', ['nset', 'cset', 'nested'])
def test_conditional_native(root, structural, kind):
    cases.test_conditional_histogram_product_and_nested_fraction_mixture(root, kind)


@pytest.mark.parametrize('algorithm,region', [(0, 0.), (0, .7), (1, .7), (2, .7)])
def test_vertical_native(root, structural, algorithm, region):
    cases.test_vertical_raw_parametric_integrals_and_floors(root, algorithm, region)


@pytest.mark.parametrize('family', ['gaussian', 'exponential', 'power', 'doublecb', 'erfexp'])
def test_analytic_native(root, structural, family):
    root = load_caching_nll()
    fixture = shape_fixture(root, family)
    model = cases.bind_model(root, fixture['shape'], [fixture['x']], fixture['coordinates'])
    cases.check_model(model)


@pytest.mark.parametrize('selected', [0, 1])
def test_multipdf_native(root, structural, selected):
    cases.test_frozen_binned_multipdf_and_native_correction(root, selected)


@pytest.mark.parametrize('policy', ['structural', 'structural_sequential'])
def test_shared_rate_shape_and_repeated_child_accumulation(root, policy):
    root = load_caching_nll()
    fixture = shape_fixture(root, 'gaussian')
    x, shape, mean = fixture['x'], fixture['shape'], fixture['mean']
    rate1 = root.RooFormulaVar('rate1', '5+@0', root.RooArgList(mean))
    rate2 = root.RooFormulaVar('rate2', '7-@0', root.RooArgList(mean))
    mixture = root.RooAddPdf('inner_extended', '', root.RooArgList(shape, shape), root.RooArgList(rate1, rate2))
    model = cases.bind_model(root, mixture, [x], fixture['coordinates'])
    batch = pack_binned_channels(model['nodes'], [{**model['rows'][0], 'name': f'copy{i}'} for i in range(3)], model['scalar'])
    arrays = dict(s=model['arrays']['s'], b=jax.device_put(dict(batch.arrays)))
    def objective(policy, chunk):
        bf = make_binned_function(batch, gradient=policy, row_chunk_size=chunk)
        return lambda theta, a: jnp.sum(bf(model['sf'](theta, a['s']), a['b']))
    reference = objective('ad', 3)
    point = model['old'].initial
    for chunk in (1, 3, 32):
        function = objective(policy, chunk)
        actual = jax.jit(jax.value_and_grad(function))(point, arrays)
        expected = jax.jit(jax.value_and_grad(reference))(point, arrays)
        for a, b in zip(actual, expected, strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
        np.testing.assert_allclose(jax.jit(jax.hessian(function))(point, arrays),
                                  jax.jit(jax.hessian(reference))(point, arrays), rtol=1e-11, atol=1e-10)


def test_empty_and_masked_channels(root, structural):
    cases.test_shared_padding_empty_channels_masks_and_fixed_folding(root)


def test_empty_row_gradient_and_invalid_chunk(root):
    root = load_caching_nll()
    fixture = shape_fixture(root, 'gaussian')
    model = cases.bind_model(root, fixture['shape'], [fixture['x']], fixture['coordinates'])
    row = model['rows'][0]
    empty = {**row, 'weights': np.empty(0), 'coordinates': {n: np.empty(0) for n in row['observables']}}
    batch = pack_binned_channels(model['nodes'], [empty], model['scalar'])
    arrays = dict(s=model['arrays']['s'], b=jax.device_put(dict(batch.arrays)))
    outputs = []
    for policy in ('ad', 'structural'):
        bf = make_binned_function(batch, gradient=policy, row_chunk_size=3)
        def function(theta):
            return jnp.sum(bf(model['sf'](theta, arrays['s']), arrays['b']))
        outputs.append(jax.jit(jax.value_and_grad(function))(model['old'].initial))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        np.testing.assert_array_equal(actual, expected)
    for bad in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match='positive integer'):
            make_binned_function(batch, gradient='structural', row_chunk_size=bad)


def test_complete_import_uses_structural_pullback(tmp_path):
    import importlib.util
    from pathlib import Path
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.model import from_workspace, make_model_function
    from combine_jax.likelihood import from_workspace as old_import
    source = Path(__file__).parents[1] / 'scripts/build_binned_fixture.py'
    spec = importlib.util.spec_from_file_location('build_binned_fixture', source)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    path = builder.build_binned_fixture(tmp_path / 'fixture')
    with open_workspace(path) as workspace:
        old = old_import(workspace)
        model = from_workspace(workspace, path.parent / 'card.txt')
    function = make_model_function(model, binned_gradient='structural', row_chunk_size=3)
    arrays = jax.device_put(dict(model.arrays))
    actual = jax.jit(jax.value_and_grad(function))(old.initial, arrays)
    expected = old.value_and_grad(old.initial)
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_allclose(a, b, rtol=1e-11, atol=1e-10)
    np.testing.assert_allclose(jax.jit(jax.hessian(function))(old.initial, arrays), old.hessian(old.initial),
                              rtol=1e-11, atol=1e-10)
