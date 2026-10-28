"""Analytic convergence, active bounds and explicit optimizer failure states."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.optimize import make_bfgs


def quadratic(x, a):
    d = x-a['center']
    return .5*d @ a['matrix'] @ d


@pytest.mark.parametrize('bounded', [False, True])
def test_coupled_quadratic_bounds_runtime_arrays_and_device_control_flow(bounded):
    fn = make_bfgs(quadratic, gradient_tolerance=1e-9)
    compiled = jax.jit(fn)
    initial = jnp.array([0., 0.])
    lower = jnp.array([-1., -3.]) if bounded else jnp.full(2, -jnp.inf)
    upper = jnp.array([1., 3.]) if bounded else jnp.full(2, jnp.inf)
    for center in ([2., -.5], [3., .7]):
        arrays = dict(matrix=jnp.array([[4., 1.], [1., 2.]]), center=jnp.array(center))
        result = jax.device_get(compiled(initial, arrays, lower, upper))
        expected = [1., center[1]+.5*(center[0]-1.)] if bounded else center
        assert result['status'] == 0
        np.testing.assert_allclose(result['x'], expected, atol=1e-8)
        assert np.max(np.abs(result['projected_gradient'])) < 1e-9
        if bounded:
            assert result['gradient'][0] < 0 and result['x'][0] == 1.
        assert result['gradient_evaluations'] == result['iteration']+1
        assert result['function_evaluations'] == result['gradient_evaluations']+result['line_search_evaluations']
    trace = jax.make_jaxpr(fn)(initial, arrays, lower, upper)
    assert not trace.consts
    assert sum(e.primitive.name == 'while' for e in trace.jaxpr.eqns) == 1
    body = next(e for e in trace.jaxpr.eqns if e.primitive.name == 'while').params['body_jaxpr']
    assert any(e.primitive.name == 'while' for e in body.jaxpr.eqns)
    assert 'callback' not in str(trace)


def test_rosenbrock_backtracking_and_nonlinear_curvature():
    def objective(x, a): return (a-x[0])**2+100*(x[1]-x[0]**2)**2
    fn = jax.jit(make_bfgs(objective, gradient_tolerance=1e-7))
    result = jax.device_get(fn(jnp.array([-1.2, 1.]), jnp.array(1.), jnp.full(2, -2.), jnp.full(2, 2.)))
    assert result['status'] == 0 and result['line_search_evaluations'] > result['iteration']
    np.testing.assert_allclose(result['x'], [1., 1.], atol=1e-7)
    assert result['value'] < 1e-15


def test_reduced_curvature_on_predefined_coupled_bound_quadratics():
    # Full-space secant changes contain nonzero blocked gradient components.
    # Feeding those into a projected search stalls these known convex minima.
    def objective(x, arrays):
        delta = x-arrays['minimum']
        return arrays['offset']+.5*delta@arrays['matrix']@delta+arrays['linear']@delta

    solve = jax.jit(make_bfgs(objective, max_iterations=1000, gradient_tolerance=1e-7))
    rng = np.random.default_rng(236)
    lower = jnp.array([-10., -10., -10., -np.inf, -np.inf, -np.inf])
    upper = jnp.full(6, jnp.inf)
    for _ in range(100):
        factor = rng.normal(size=(6, 6))
        matrix = factor.T@factor+.1*np.eye(6)
        minimum = np.r_[np.full(3, -10.), rng.uniform(-2., 2., size=3)]
        linear = np.r_[rng.uniform(.05, 2., size=3), np.zeros(3)]
        for offset in (0., 1.e7):
            arrays = dict(matrix=matrix, minimum=minimum, linear=linear, offset=np.array(offset))
            result = jax.device_get(solve(jnp.zeros(6), arrays, lower, upper))
            assert result['status'] == 0
            assert np.max(abs(result['projected_gradient'])) <= 1e-7
            np.testing.assert_allclose(result['x'], minimum, atol=1e-6, rtol=0.)
            np.testing.assert_allclose(result['gradient'], matrix@(result['x']-minimum)+linear,
                                       atol=1e-12, rtol=1e-12)
            assert np.all(result['gradient'][:3] > 0.)


@pytest.mark.parametrize('sign', [-1., 1.])
def test_blocked_coordinate_can_become_free_when_other_coordinates_move(sign):
    matrix = jnp.array([[4., 3.], [3., 4.]])
    center = sign*jnp.array([.5, 1.])
    initial = sign*jnp.array([0., 3.])
    lower = jnp.array([0., -5.]) if sign > 0 else jnp.array([-1., -5.])
    upper = jnp.array([1., 5.]) if sign > 0 else jnp.array([0., 5.])
    # x starts blocked at its bound, but must be released after y changes.
    assert sign*float((matrix@(initial-center))[0]) > 0
    result = jax.device_get(jax.jit(make_bfgs(quadratic, gradient_tolerance=1e-9))(
        initial, dict(matrix=matrix, center=center), lower, upper))
    assert result['status'] == 0
    np.testing.assert_allclose(result['x'], center, atol=1e-9, rtol=0.)
    assert np.max(abs(result['gradient'])) <= 1e-9


@pytest.mark.parametrize('case', ['limit', 'line', 'nonfinite', 'bounds', 'equal_bounds', 'bad_start'])
def test_failures_never_report_convergence(case):
    objective = lambda x, a: jnp.sum(x*x)*a
    config = {'max_iterations': 0} if case == 'limit' else {'max_linesearch': 1} if case == 'line' else {}
    initial = jnp.array([1.])
    lower, upper = jnp.array([-2.]), jnp.array([2.])
    a = jnp.array(jnp.nan if case == 'nonfinite' else 1.)
    if case == 'bounds': upper = jnp.array([0.])
    if case == 'equal_bounds': lower = upper = initial
    if case == 'bad_start': initial = jnp.array([jnp.nan])
    result = jax.device_get(jax.jit(make_bfgs(objective, **config))(initial, a, lower, upper))
    assert result['status'] == dict(limit=1, line=2, nonfinite=3, bounds=4, equal_bounds=4, bad_start=4)[case]
    if case == 'line':
        np.testing.assert_array_equal(result['x'], initial)
        assert result['gradient_evaluations'] == 1 and result['function_evaluations'] == 2
    else: assert result['iteration'] == 0


def test_empty_and_configuration_contracts():
    objective = lambda x, a: jnp.sum(x*x)+a
    fn = jax.jit(make_bfgs(objective))
    empty = jnp.empty(0, dtype=jnp.float64)
    result = jax.device_get(fn(empty, jnp.array(7.), empty, empty))
    assert result['status'] == 0 and result['value'] == 7. and result['iteration'] == 0
    assert result['inverse'].shape == (0, 0)
    for kw in ({'max_iterations': -1}, {'max_iterations': True}, {'max_linesearch': 0},
               {'max_linesearch': 2.5}, {'gradient_tolerance': 0.}, {'gradient_tolerance': np.inf}):
        with pytest.raises(ValueError): make_bfgs(objective, **kw)
    for value in (-1.,np.nan,np.inf,True):
        with pytest.raises(ValueError):make_bfgs(objective,armijo_roundoff_multiplier=value)
    for x, lo, hi in [(jnp.ones((1, 1)), empty, empty), (jnp.ones(1, dtype=jnp.float32), empty, empty),
                      (jnp.ones(1), empty, empty)]:
        with pytest.raises(ValueError): fn(x, jnp.array(0.), lo, hi)
