"""The blocked update must preserve the full dense BFGS formula."""
import numpy as np
import pytest

import jax
import jax.numpy as jnp

from combine_jax.tensor.bfgs_update import blocked_bfgs_update, deferred_identity


@pytest.mark.parametrize('n,block',[(1,128),(13,5),(137,128)])
@pytest.mark.parametrize('seed',[3.,np.nan,np.inf,-np.inf])
def test_deferred_identity_overwrites_every_element(n,block,seed):
    dependency=jnp.full(n,seed,dtype=jnp.float64)
    actual=jax.jit(lambda g:deferred_identity(g,block_size=block))(dependency)
    np.testing.assert_array_equal(actual,np.eye(n))


@pytest.mark.parametrize('block',[0,-1,1.5,True])
def test_deferred_identity_rejects_invalid_block(block):
    with pytest.raises(ValueError,match='positive integer'):
        deferred_identity(jnp.ones(2,dtype=jnp.float64),block_size=block)


@pytest.mark.parametrize('dependency',[np.empty(0),np.ones((2,2)),np.ones(2,dtype=np.float32)])
def test_deferred_identity_rejects_invalid_dependency(dependency):
    with pytest.raises(ValueError,match='nonempty float64'):
        deferred_identity(jnp.asarray(dependency))


def dense(base, step, change):
    hy = base@change
    ys = jnp.dot(step, change)
    result = base+(ys+jnp.dot(change,hy))/(ys*ys)*jnp.outer(step,step)
    result -= (jnp.outer(hy,step)+jnp.outer(step,hy))/ys
    result = .5*(result+result.T)
    relative = jnp.sum(jnp.abs(result-base))/jnp.maximum(jnp.sum(jnp.abs(result)),jnp.finfo(base.dtype).tiny)
    return result, relative, jnp.all(jnp.isfinite(result))


@pytest.mark.parametrize('n,block', [(1,128),(5,1),(5,3),(5,128),(137,16),(137,128)])
@pytest.mark.parametrize('kind', ['symmetric','asymmetric','bound_masked','reset'])
def test_dense_update_agreement(n,block,kind):
    rng=np.random.default_rng(527)
    base=rng.normal(size=(n,n))
    if kind!='asymmetric':base=base@base.T+np.eye(n)
    if kind=='bound_masked':
        blocked=np.arange(n)%3==0
        base=np.where(blocked[:,None]|blocked[None,:],np.eye(n),base)
    if kind=='reset':base=np.eye(n)
    step=rng.normal(size=n);change=step+.01*rng.normal(size=n)
    args=tuple(jnp.asarray(v,dtype=jnp.float64) for v in (base,step,change))
    reference=jax.jit(dense)(*args)
    actual=jax.jit(lambda *a:blocked_bfgs_update(*a,block_size=block))(*args)
    np.testing.assert_allclose(actual[0],reference[0],rtol=2e-13,atol=5e-13)
    np.testing.assert_allclose(actual[1],reference[1],rtol=2e-13,atol=5e-15)
    assert bool(actual[2])==bool(reference[2])
    np.testing.assert_allclose(actual[0],actual[0].T,rtol=0,atol=5e-13)


def test_repeated_updates_and_nonfinite_status():
    n=19;rng=np.random.default_rng(300)
    a=b=jnp.eye(n,dtype=jnp.float64)
    fn=jax.jit(lambda *args:blocked_bfgs_update(*args,block_size=7))
    for _ in range(8):
        step=jnp.asarray(rng.normal(size=n));change=step+.03*jnp.asarray(rng.normal(size=n))
        a,ra,fa=fn(a,step,change);b,rb,fb=dense(b,step,change)
        np.testing.assert_allclose(a,b,rtol=2e-12,atol=2e-13)
        np.testing.assert_allclose(ra,rb,rtol=2e-12,atol=2e-15)
        assert fa and fb
    bad=jnp.full_like(a,jnp.nan)
    assert not bool(fn(bad,step,change)[2])


@pytest.mark.parametrize('reset',[False,True])
@pytest.mark.parametrize('update',[False,True])
@pytest.mark.parametrize('kind',['symmetric','asymmetric'])
def test_implicit_bound_mask_reset_and_skipped_update(reset,update,kind):
    rng=np.random.default_rng(800);n=13
    matrix=rng.normal(size=(n,n))
    if kind=='symmetric':matrix=matrix@matrix.T+np.eye(n)
    matrix=jnp.asarray(matrix)
    blocked=jnp.arange(n)%3==0
    step=jnp.asarray(rng.normal(size=n))
    change=step+.02*jnp.asarray(rng.normal(size=n)) if update else jnp.zeros(n)
    base=jnp.where(reset|blocked[:,None]|blocked[None,:],jnp.eye(n),matrix)
    expected=dense(base,step,change) if update else (base,0.,True)
    actual=jax.jit(lambda m,s,c,b,r,u:blocked_bfgs_update(m,s,c,blocked=b,reset=r,update=u,block_size=5))(
        matrix,step,change,blocked,jnp.array(reset),jnp.array(update))
    np.testing.assert_allclose(actual[0],expected[0],rtol=2e-13,atol=5e-13)
    np.testing.assert_allclose(actual[1],expected[1],rtol=2e-13,atol=5e-15)
    assert actual[2]==expected[2]


@pytest.mark.parametrize('options',[{'blocked':jnp.zeros(2)}, {'blocked':jnp.zeros(3,dtype=bool)},
                                     {'reset':1}, {'update':jnp.array([True])}])
def test_invalid_masks_rejected(options):
    with pytest.raises(ValueError,match='bool'):
        blocked_bfgs_update(jnp.eye(2,dtype=jnp.float64),jnp.ones(2),jnp.ones(2),**options)


@pytest.mark.parametrize('seed',[-1e100,np.nan])
def test_every_output_element_overwrites_arbitrary_initial_contents(seed):
    base=jnp.arange(169,dtype=jnp.float64).reshape(13,13)
    change=jnp.zeros(13).at[0].set(seed)
    result,relative,finite=jax.jit(lambda b,c:blocked_bfgs_update(
        b,jnp.ones(13),c,block_size=5,update=False))(base,change)
    np.testing.assert_array_equal(result,base)
    assert float(relative)==0. and bool(finite)


@pytest.mark.parametrize('block',[0,-1,1.5,True])
def test_invalid_block_rejected(block):
    with pytest.raises(ValueError,match='positive integer'):
        blocked_bfgs_update(jnp.eye(2,dtype=jnp.float64),jnp.ones(2),jnp.ones(2),block_size=block)


@pytest.mark.parametrize('base,step,change',[
    (np.eye(2,dtype=np.float32),np.ones(2),np.ones(2)),
    (np.zeros((2,3)),np.ones(2),np.ones(2)),
    (np.empty((0,0)),np.empty(0),np.empty(0)),
    (np.eye(2),np.ones(3),np.ones(2)),
])
def test_invalid_shapes_rejected(base,step,change):
    with pytest.raises(ValueError):blocked_bfgs_update(jnp.asarray(base),jnp.asarray(step),jnp.asarray(change))
