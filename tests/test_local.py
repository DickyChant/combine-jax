import numpy as np
import pytest

from combine_jax.likelihood import from_workspace
from combine_jax.root_io import open_workspace


def compare_local(likelihood, points, *, constraint_batch_size=1, hessian_chunk_size=None, prepared=False):
    local = likelihood.local_evaluator(constraint_batch_size=constraint_batch_size,
                                       hessian_chunk_size=hessian_chunk_size)
    assert sorted(n for t in local.terms for n in t.constraints) == sorted(likelihood.constraints)
    if prepared:
        for operation in ("value","value_and_grad","hessian"):
            local.prepare(operation,points[0],release_cache=True)
    for point in points:
        actual, gradient = local.value_and_grad(point)
        expected, full_gradient = likelihood.value_and_grad(point)
        np.testing.assert_allclose(actual, expected, atol=1e-11, rtol=1e-14)
        np.testing.assert_allclose(local.value(point), expected, atol=1e-11, rtol=1e-14)
        np.testing.assert_allclose(gradient, full_gradient, atol=1e-10, rtol=1e-11)
        np.testing.assert_allclose(local.hessian(point, retain_compiled=False), likelihood.hessian(point), atol=1e-10, rtol=1e-11)
    local.clear_cache()
    return local


@pytest.mark.parametrize("hessian_chunk_size", (None,2))
def test_prepared_kernels_survive_cache_release_and_can_be_rebuilt(counting_path,hessian_chunk_size):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    points = [likelihood.initial,likelihood.initial+np.array([.2,-.4,.3])]
    local = compare_local(likelihood,points,constraint_batch_size=32,
                          hessian_chunk_size=hessian_chunk_size,prepared=True)
    for operation in ("value","value_and_grad","hessian"):
        records = local.prepare(operation,points[1],release_cache=True)
        assert records and all(not r["reused"] for r in records)
        assert all(r["reused"] for r in local.prepare(operation,points[0],release_cache=True))
    np.testing.assert_allclose(local.hessian(points[0],retain_compiled=False,release_cache=True),
                               likelihood.hessian(points[0]),atol=1e-10,rtol=1e-11)
    local.clear_cache(release_cache=True)
    value,gradient = local.value_and_grad(points[1])
    expected,expected_gradient = likelihood.value_and_grad(points[1])
    np.testing.assert_allclose(value,expected,atol=1e-11)
    np.testing.assert_allclose(gradient,expected_gradient,atol=1e-10,rtol=1e-11)
    with pytest.raises(ValueError,match="retain_compiled=False"):
        local.hessian(points[0],release_cache=True)
    with pytest.raises(ValueError,match="Unknown local operation"):
        local.prepare("unknown")


@pytest.mark.parametrize("hessian_chunk_size", [None, 1, 2, 8])
def test_local_shared_parameters_constraints_and_offset(counting_path, hessian_chunk_size):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    points = [likelihood.initial, likelihood.initial+np.array([.3, -.7, .5])]
    local = compare_local(likelihood.at_zero(points[0]), points, hessian_chunk_size=hessian_chunk_size)
    assert any(len(t.indices)<len(local.parameter_names) for t in local.terms)
    assert sum("r" in t.parameter_names for t in local.terms) == 2
    with pytest.raises(ValueError, match="parameter shape"):
        local.value([1.])


@pytest.mark.parametrize("batch_size", [2, 32])
def test_batched_constraints_preserve_full_derivatives(counting_path, batch_size):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    points = [likelihood.initial, likelihood.initial+np.array([.3, -.7, .5])]
    local = compare_local(likelihood.at_zero(points[0]), points, constraint_batch_size=batch_size)
    assert len(local.terms) < len(likelihood.local_evaluator().terms)
    assert any(len(t.constraints) > 1 for t in local.terms)


@pytest.mark.parametrize("batch_size", [0, -1, 1.5])
def test_constraint_batch_size_requires_positive_integer(counting_path, batch_size):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    with pytest.raises((ValueError, TypeError)):
        likelihood.local_evaluator(constraint_batch_size=batch_size)


@pytest.mark.parametrize("chunk_size", [0, -1, 1.5])
def test_hessian_chunk_size_requires_positive_integer(counting_path, chunk_size):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    with pytest.raises((ValueError, TypeError)):
        likelihood.local_evaluator(hessian_chunk_size=chunk_size)
