"""Fold all refinements, preserving native adaptivity and local derivatives."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.romberg import RombergConfig, integrate_numpy
from combine_jax.romberg_table import build_table, integral


@pytest.mark.parametrize('config',[
    RombergConfig(max_steps=12),RombergConfig(max_steps=12,fixed_steps=6),
    RombergConfig(max_steps=12,extrapolate=False),RombergConfig(max_steps=12,eps_abs=1e-12,eps_rel=1e-12)])
def test_folded_linear_basis_native_refinement_gradient_hvp(config):
    def basis(x):return np.stack((np.exp(-.2*x*x),x*np.exp(-.2*x*x),x*x*np.exp(-.2*x*x)))
    table=build_table(basis,-2.,3.,config)
    def f(p,t):return integral(p,t,5.,config,jnp)[0]
    value=jax.jit(f);gradient=jax.jit(jax.grad(f));hessian=jax.jit(jax.hessian(f))
    for p in np.random.default_rng(457).uniform([.8,-.1,.01],[1.2,.1,.1],(26,3)):
        direct=integrate_numpy(lambda c,x:c@basis(x),p,-2.,3.,config)
        folded=integral(p,table,5.,config)
        np.testing.assert_allclose(folded[0],direct[0],atol=2e-13,rtol=2e-13)
        assert folded[1:]==direct[1:]
        np.testing.assert_allclose(value(p,table),direct[0],atol=2e-13,rtol=2e-13)
        # At the selected refinement the expression is linear in coefficients.
        expected=table[direct[1]-1]
        if direct[2]==1:
            from combine_jax.romberg import extrapolate
            expected=extrapolate(table[direct[1]-5:direct[1]])[0]
        if direct[2]==2:expected*=0
        np.testing.assert_allclose(gradient(p,table),expected,atol=2e-13,rtol=2e-13)
        np.testing.assert_array_equal(hessian(p,table),np.zeros((3,3)))
    assert not jax.make_jaxpr(f)(p,table).consts
