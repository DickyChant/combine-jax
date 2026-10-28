"""EDM estimates are checked against known objective gaps, not only statuses."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.optimize import make_bfgs
from combine_jax.tensor.fitting import make_fit_function


@pytest.mark.parametrize('offset', [0., 1.e7])
def test_edm_coupled_quadratic_matches_exact_gap_and_counts(offset):
    matrix = jnp.array([[17., 3., 0.], [3., 2., .1], [0., .1, .07]])
    center = jnp.array([.3, -1.7, 2.1])
    def objective(x, a):
        d = x-a['center']
        return a['offset']+.5*d@a['matrix']@d
    args = (jnp.zeros(3), dict(matrix=matrix, center=center, offset=jnp.array(offset)),
            jnp.full(3, -10.), jnp.full(3, 10.))
    fn = make_bfgs(objective, stopping_criterion='edm', edm_tolerance=1e-4)
    result = jax.device_get(jax.jit(fn)(*args))
    assert result['status'] == 0 and 0 <= result['estimated_edm'] < 1e-4
    d = result['x']-center
    exact_gap = float(.5*d@matrix@d)
    assert exact_gap < 1e-4
    assert result['function_evaluations'] == result['gradient_evaluations']+result['line_search_evaluations']
    assert result['gradient_evaluations'] == result['iteration']+1
    trace = jax.make_jaxpr(fn)(*args)
    assert not trace.consts and 'callback' not in str(trace)
    assert sum(e.primitive.name == 'while' for e in trace.jaxpr.eqns) == 1


def test_identity_metric_cannot_prematurely_accept_flat_quadratic():
    def objective(x, _): return .5*1e-6*(x[0]-100.)**2
    result = jax.device_get(jax.jit(make_bfgs(objective, stopping_criterion='edm'))(
        jnp.zeros(1), {}, jnp.full(1, -jnp.inf), jnp.full(1, jnp.inf)))
    assert result['iteration'] > 0
    assert result['status'] == 0 and result['value'] < 1e-4


def test_active_bound_uses_reduced_edm_and_keeps_covariance_separate():
    def objective(x, _): return .5*(x[0]+2.)**2+3.*(x[1]-.4)**2
    coordinates = dict(reference=jnp.zeros(2), free_indices=jnp.arange(2), scale=jnp.ones(2),
                       lower=jnp.array([0., -2.]), upper=jnp.array([1., 2.]),
                       physical_lower=jnp.array([0., -2.]), physical_upper=jnp.array([1., 2.]))
    result = jax.device_get(jax.jit(make_fit_function(objective, stopping_criterion='edm'))(coordinates, {}))
    assert result['status'] == result['covariance_status'] == 0
    assert result['estimated_edm'] < 1e-4 and result['free_gradient'][0] == 2.
    np.testing.assert_allclose(result['covariance'], np.diag([1., 1/6]), atol=1e-12)
    assert abs(result['values'][1]-.4) < .01


@pytest.mark.parametrize('threshold', [0., -1., np.nan, np.inf, True])
def test_invalid_edm_threshold(threshold):
    with pytest.raises(ValueError, match='edm_tolerance'):
        make_bfgs(lambda x, _: jnp.sum(x*x), edm_tolerance=threshold)


def test_invalid_criterion():
    with pytest.raises(ValueError, match='stopping_criterion'):
        make_bfgs(lambda x, _: jnp.sum(x*x), stopping_criterion='unknown')


def test_fit_refines_optimistic_metric_on_conditioned_quadratics():
    def objective(x, a):
        delta=x-a['center']
        return .5*delta@a['matrix']@delta
    coordinates=dict(reference=jnp.zeros(7),free_indices=jnp.arange(7),scale=jnp.ones(7),
                     lower=jnp.full(7,-jnp.inf),upper=jnp.full(7,jnp.inf),
                     physical_lower=jnp.full(7,-jnp.inf),physical_upper=jnp.full(7,jnp.inf))
    fit=jax.jit(make_fit_function(objective,stopping_criterion='edm'))
    no_refinement=jax.jit(make_fit_function(objective,stopping_criterion='edm',max_edm_refinements=0))
    estimated_only=jax.jit(make_fit_function(objective,stopping_criterion='edm',edm_validation='estimated'))
    rng=np.random.default_rng(358)
    refined=0
    for i in range(10):
        q,_=np.linalg.qr(rng.normal(size=(7,7)))
        a=dict(matrix=q@np.diag(np.geomspace(.01,100,7))@q.T,center=rng.normal(size=7))
        result=jax.device_get(fit(coordinates,a))
        assert result['status']==0 and result['edm_curvature_validated']
        assert 0<=result['hessian_edm']<1e-4 and result['value']<1e-4
        assert result['iteration']<=1000
        assert result['hessian_evaluations']==1+result['edm_refinements']
        refined+=int(result['edm_refinements']>0)
        if i==0:
            rejected=jax.device_get(no_refinement(coordinates,a))
            assert rejected['status']==5 and not rejected['converged']
            assert rejected['hessian_edm']>1e-4 and not rejected['valid_covariance']
            diagnostic=jax.device_get(estimated_only(coordinates,a))
            assert diagnostic['status']==0 and not diagnostic['edm_refinement_enabled']
            assert diagnostic['corrected_estimated_edm']<1e-4
            assert not diagnostic['edm_curvature_validated'] and diagnostic['hessian_edm']>1e-4
            np.testing.assert_array_equal(diagnostic['values'],rejected['values'])
    assert refined>0


def test_edm_bound_metric_holds_blocked_coordinates_fixed():
    matrix=jnp.array([[4.,1.],[1.,2.]])
    center=jnp.array([2.,-.5])
    def objective(x,_):return .5*(x-center)@matrix@(x-center)
    coordinates=dict(reference=jnp.zeros(2),free_indices=jnp.arange(2),scale=jnp.ones(2),
                     lower=jnp.array([-1.,-3.]),upper=jnp.array([1.,3.]),
                     physical_lower=jnp.array([-1.,-3.]),physical_upper=jnp.array([1.,3.]))
    result=jax.device_get(jax.jit(make_fit_function(objective,stopping_criterion='edm'))(coordinates,{}))
    assert result['status']==0 and result['values'][0]==1.
    gap=float(objective(result['values'],{})-objective(jnp.array([1.,0.]),{}))
    np.testing.assert_allclose(result['hessian_edm'],gap,atol=1e-14,rtol=1e-8)
    np.testing.assert_allclose(result['covariance'],np.linalg.inv(matrix),atol=1e-12)
