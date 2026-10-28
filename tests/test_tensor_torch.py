"""Optional adapter: native fresh-card acceptance and functional runtime buffers."""
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip('torch')

from combine_jax.graph import UnsupportedModelError
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.torch_backend import _ArrayAPI, make_torch_function, to_torch_arrays


def test_counting_native_and_runtime_buffers(counting_path):
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, Path(__file__).parent/'cards/counting.txt')
        fn = make_torch_function(model)
        arrays = to_torch_arrays(model)
        saved = {key: value.clone() for key, value in arrays.items()}
        reference = root.cacheutils.CachingSimNLL(workspace.pdf('model_s'), workspace.data('data_obs'),
                                                workspace.obj('ModelConfig').GetNuisanceParameters())
        reference.clearConstantZeroPoint(); reference.clearZeroPoint()
        theta = torch.tensor(model.arrays['scalars__parameter_initial'], requires_grad=True)
        for delta in (0., .03, -.04):
            point = theta+delta
            for name, value in zip(model.metadata['parameter_names'], point.detach().numpy(), strict=True):
                workspace.var(name).setVal(float(value))
            actual = fn(point, arrays)
            assert abs(float(actual.detach())-float(reference.getVal())) < 1e-10
            changed = {**arrays, 'offset': arrays['offset']+2.}
            shifted = fn(point, changed)
            assert abs(float((shifted-actual).detach())-2.) < 1e-14
            g1 = torch.autograd.grad(actual, theta, retain_graph=True)[0]
            g2 = torch.autograd.grad(shifted, theta)[0]
            torch.testing.assert_close(g1, g2, rtol=0, atol=0)
        for key in arrays:
            torch.testing.assert_close(arrays[key], saved[key], rtol=0, atol=0)
        with pytest.raises(ValueError, match='float64'):
            fn(theta.float(), arrays)
        unsupported = TensorBundle({**model.metadata, 'kind': 'unknown_likelihood'}, model.arrays)
        with pytest.raises(UnsupportedModelError, match='Unsupported'):
            make_torch_function(unsupported)


@pytest.mark.parametrize('name', ['minimum','maximum'])
def test_minimum_maximum_keep_half_derivative_at_scalar_tie(name):
    point = torch.tensor([1.], dtype=torch.float64, requires_grad=True)
    result = getattr(_ArrayAPI(torch), name)(point, 1.)
    assert torch.autograd.grad(result.sum(), point)[0].item() == .5


@pytest.mark.parametrize('family', ['gaussian','gaussian_mean','exponential',
    'exponential_coefficient','power','doublecb','doublecb_xp','doublecb_signed','erfexp'])
def test_analytic_native_and_old_ad(root, family):
    from test_tensor_analytic_shapes import shape_fixture
    from test_tensor_binned_channels import bind_model
    from combine_jax.tensor.model import compose_binned
    from combine_jax.tensor.counting import pack_counting_channels
    from combine_jax.tensor.constraints import pack_constraints
    root = load_caching_nll()
    fixture = shape_fixture(root, family)
    reference = bind_model(root, fixture['shape'], [fixture['x']], fixture['coordinates'])
    scalars = reference['scalar']
    model = compose_binned(scalars, pack_constraints(reference['nodes'], [], {}, scalars),
        pack_counting_channels(reference['nodes'], [], scalars), reference['binned'])
    function = make_torch_function(model)
    arrays = to_torch_arrays(model)
    old = reference['old']
    rng = np.random.default_rng(2351)
    points = [old.initial, np.array([rng.uniform(p.minimum+.1*(p.maximum-p.minimum),
        p.maximum-.1*(p.maximum-p.minimum)) for p in old.parameters])]
    for point in points:
        theta = torch.tensor(point, dtype=torch.float64, requires_grad=True)
        value = function(theta, arrays)
        gradient = torch.autograd.grad(value, theta)[0].numpy()
        hessian = torch.autograd.functional.hessian(lambda x: function(x, arrays), theta).numpy()
        np.testing.assert_allclose(float(value.detach()), reference['native'](point), rtol=0, atol=1e-9)
        np.testing.assert_allclose(gradient, old.value_and_grad(point)[1], rtol=1e-10, atol=1e-9)
        np.testing.assert_allclose(hessian, old.hessian(point), rtol=1e-10, atol=1e-9)
