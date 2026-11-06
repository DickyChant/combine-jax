"""Near-stationary EDM restarts still require measured positive curvature."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.optimize import make_bfgs


@pytest.mark.parametrize('block',[0,1])
def test_small_secant_restart_preserves_failure_and_requires_update(block):
    objective=lambda x,_: jnp.dot(x,x)
    args=(jnp.array([1e-12]),{},jnp.array([-5.]),jnp.array([5.]))
    settings=dict(stopping_criterion='edm',edm_tolerance=1e-4,max_iterations=4,
                  inverse_update_block_size=block)
    original=jax.device_get(jax.jit(make_bfgs(objective,**settings))(*args))
    assert original['status']==1
    assert original['skipped_updates']==original['iteration']==4
    corrected=jax.device_get(jax.jit(make_bfgs(objective,rescale_small_secants=True,**settings))(*args))
    assert corrected['status']==0
    assert corrected['iteration']>corrected['skipped_updates']
    np.testing.assert_allclose(corrected['inverse'],[[.5]],rtol=1e-14,atol=0)
    exact_edm=float(jnp.dot(corrected['x'],corrected['x']))
    assert exact_edm<1e-4
    np.testing.assert_allclose(corrected['estimated_edm'],exact_edm,rtol=1e-14,atol=0)


@pytest.mark.parametrize('block',[0,1])
def test_small_gradient_does_not_bypass_wrong_initial_metric(block):
    objective=lambda x,_: .5e-12*jnp.dot(x,x)
    args=(jnp.array([1e4]),{},jnp.array([-1e5]),jnp.array([1e5]))
    result=jax.device_get(jax.jit(make_bfgs(objective,rescale_small_secants=True,
        stopping_criterion='edm',edm_tolerance=1e-6,max_iterations=1,
        inverse_update_block_size=block))(*args))
    assert result['status']!=0
    assert result['corrected_estimated_edm']>1e-6


@pytest.mark.parametrize('block',[0,2])
def test_ordinary_updates_remain_identical(block):
    objective=lambda x,_: jnp.dot(x,jnp.array([2.,5.,9.])*x)
    args=(jnp.array([1.,-2.,3.]),{},jnp.full(3,-10.),jnp.full(3,10.))
    options=dict(stopping_criterion='edm',edm_tolerance=1e-4,inverse_update_block_size=block)
    plain=jax.device_get(jax.jit(make_bfgs(objective,**options))(*args))
    scaled=jax.device_get(jax.jit(make_bfgs(objective,rescale_small_secants=True,**options))(*args))
    for key in ('x','value','inverse','status','iteration','corrected_estimated_edm'):
        np.testing.assert_array_equal(scaled[key],plain[key])


def test_invalid_option():
    with pytest.raises(ValueError,match='boolean'):
        make_bfgs(lambda x,_:jnp.dot(x,x),rescale_small_secants=1)
