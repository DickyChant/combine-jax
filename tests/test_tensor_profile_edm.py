"""Profile stopping diagnostics stay visible without returning dense matrices."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import Parameter
from combine_jax.tensor.parameter_info import pack_parameter_info, fit_coordinates, COORDINATE_FIELDS
from combine_jax.tensor.profiling import make_profile_scan_function


def coordinates(points, *, all_fixed=False):
    parameters = [Parameter('x', 0., -4., 4., 1., 'poi')]
    if not all_fixed:
        parameters.append(Parameter('y', 0., -4., 4., 1., 'nuisance'))
    info = pack_parameter_info(parameters)
    rows = [fit_coordinates(info, fixed={'x': point}) for point in points]
    return {key: jnp.asarray(np.stack([row[key] for row in rows])) for key in COORDINATE_FIELDS}


@pytest.mark.parametrize('validation', ['exact_hessian', 'estimated'])
def test_profile_reports_edm_and_curvature_scope(validation):
    def objective(theta, arrays):
        residual = theta[1] - theta[0]
        return residual**4 + .002 * residual**2 + arrays['offset'] * theta[0]**2
    points = [-1., .5, 1.5]
    function = jax.jit(make_profile_scan_function(objective, stopping_criterion='edm',
        edm_validation=validation, edm_tolerance=1e-4, gradient_tolerance=1e-12))
    result = jax.device_get(function(coordinates(points), {'offset': jnp.float64(.3)}))
    np.testing.assert_array_equal(result['status'], 0)
    np.testing.assert_array_equal(result['values'][:, 0], points)
    assert np.all((result['corrected_estimated_edm'] >= 0) & (result['corrected_estimated_edm'] < 1e-4))
    assert np.max(np.abs(result['projected_gradient'])) > 1e-7  # EDM, not a strict-gradient fit
    assert not any(name in result for name in ('hessian', 'covariance', 'inverse'))
    if validation == 'exact_hessian':
        assert np.all(result['edm_curvature_validated'])
        assert np.all((result['hessian_edm'] >= 0) & (result['hessian_edm'] < 1e-4))
        assert np.all(result['hessian_evaluations'] >= 1)
        assert np.all(result['edm_refinement_enabled'])
    else:
        assert not np.any(result['edm_curvature_validated'])
        assert np.isnan(result['hessian_edm']).all()
        np.testing.assert_array_equal(result['hessian_evaluations'], 0)
        np.testing.assert_array_equal(result['edm_refinements'], 0)
        assert not np.any(result['edm_refinement_enabled'])


def test_all_fixed_edm_profile_has_zero_work_and_zero_edm():
    fn = jax.jit(make_profile_scan_function(lambda theta, data: theta[0]**2,
        stopping_criterion='edm', edm_validation='estimated'))
    result = jax.device_get(fn(coordinates([-1., 0., 2.], all_fixed=True), {}))
    np.testing.assert_array_equal(result['status'], 0)
    np.testing.assert_array_equal(result['value'], [1., 0., 4.])
    np.testing.assert_array_equal(result['corrected_estimated_edm'], 0)
    np.testing.assert_array_equal(result['hessian_edm'], 0)
    np.testing.assert_array_equal(result['hessian_evaluations'], 0)
    assert np.all(result['edm_curvature_validated'])


def test_failed_profile_retains_edm_failure_without_curvature():
    fn = jax.jit(make_profile_scan_function(lambda theta, data: (theta[1] - theta[0])**2,
        stopping_criterion='edm', max_iterations=0))
    result = jax.device_get(fn(coordinates([-1., 1.]), {}))
    np.testing.assert_array_equal(result['status'], 1)
    assert np.all(result['corrected_estimated_edm'] > 1e-4)
    assert not np.any(result['edm_curvature_validated'])
    np.testing.assert_array_equal(result['hessian_evaluations'], 0)
