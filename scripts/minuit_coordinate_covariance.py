"""Diagnostic covariance in Minuit coordinates, given the actual internal point.

This is separate from fit acceptance. The caller must supply derivatives at the
same external point, with the same parameter order and errordef. In particular,
Minuit's clipped Ext2int result cannot replace the saved internal fit state.
No curvature repair, projection, or pseudoinverse is performed.
"""

from importlib.metadata import version

import numpy as np


def capture_internal_point(minuit):
    """Read actual coordinates using the explicitly pinned private state layout."""
    if version("iminuit") != "2.33.0":
        raise ValueError("Internal state capture is validated only for iminuit 2.33.0")
    if minuit.fmin is None:
        raise ValueError("A completed minimization is required")
    state = minuit._fmin._src.__getstate__()
    if len(state) != 6 or not state[2]:
        raise ValueError("Unsupported Minuit serialized state")
    internal = np.asarray(state[2][-1].vec, dtype=np.float64).copy()
    trafo = minuit._last_state.trafo
    free = np.asarray([trafo.ext_of_int(i) for i in range(trafo.variable_parameters)])
    expected = np.flatnonzero(~np.asarray(minuit.fixed))
    if not np.array_equal(free, expected) or internal.shape != free.shape:
        raise ValueError("Minuit internal parameter mapping differs from the free parameters")
    external = np.asarray(minuit.values)
    reconstructed = np.asarray([trafo.int2ext(i, float(u)) for i, u in enumerate(internal)])
    if not np.array_equal(reconstructed, external[free]):
        raise ValueError("Minuit internal point does not reproduce the external point")
    return free.astype(int), internal


def _validate_bounds(bounds, *, fixed=False):
    low, high = bounds
    if any(v is not None and not np.isfinite(v) for v in bounds):
        raise ValueError("Use None for an absent bound, finite numbers otherwise")
    if low is not None and high is not None and (low > high or (low == high and not fixed)):
        raise ValueError("Lower bound must be below upper bound for free coordinates")


def transform_coordinate(u, bounds):
    """Return external value, first derivative and second derivative."""
    _validate_bounds(bounds)
    low, high = bounds
    if low is not None and high is not None:
        half = .5*(high-low)
        return low+half*(np.sin(u)+1.), half*np.cos(u), -half*np.sin(u)
    if low is not None or high is not None:
        root = np.hypot(1., u)
        sign = 1. if low is not None else -1.
        origin = low if low is not None else high
        return origin+sign*(root-1.), sign*u/root, sign/root**3
    return float(u), 1., 0.


def coordinate_covariance(point, gradient, hessian, bounds, free, internal, *, errordef=.5):
    """Return the full chain-rule covariance and separate Minuit error fields.

    Fixed rows, columns and error entries are zero. Positive internal curvature
    is required, but a bound may make the external covariance singular. Such a
    result is diagnostic and does not pass an external positive-definite gate
    implicitly. Arrays are returned without JSON conversion for later auditing.
    """
    point, gradient, hessian, internal = (
        np.asarray(v, dtype=np.float64) for v in (point, gradient, hessian, internal))
    free = np.asarray(free)
    size = len(point)
    if (point.shape != (size,) or gradient.shape != (size,)
            or hessian.shape != (size, size) or len(bounds) != size
            or free.ndim != 1 or not np.issubdtype(free.dtype, np.integer)
            or not np.array_equal(free, np.unique(free))
            or np.any((free < 0) | (free >= size)) or internal.shape != free.shape):
        raise ValueError("Inconsistent point, derivative, bound or free-coordinate shapes")
    if not all(np.isfinite(a).all() for a in (point, gradient, hessian, internal)):
        raise ValueError("Nonfinite input")
    if not np.isfinite(errordef) or errordef <= 0:
        raise ValueError("errordef must be finite and positive")
    free_set = set(free)
    for i, (x, pair) in enumerate(zip(point, bounds)):
        _validate_bounds(pair, fixed=i not in free_set)
        low, high = pair
        if (low is not None and x < low) or (high is not None and x > high):
            raise ValueError("External point is outside its bounds")
    transformed = np.asarray([transform_coordinate(u, bounds[i]) for i, u in zip(free, internal)])
    if len(free):
        external, jacobian, second = transformed.T
        if not np.allclose(external, point[free], atol=1e-12, rtol=1e-14):
            raise ValueError("Actual internal coordinates do not reproduce the external point")
    else:
        external = jacobian = second = np.empty(0, dtype=np.float64)
    correction = gradient[free]*second
    curvature = hessian[np.ix_(free, free)]*np.outer(jacobian, jacobian)+np.diag(correction)
    if not np.isfinite(curvature).all() or np.any(np.diag(curvature) <= 0):
        raise ValueError("Internal curvature has nonpositive diagonal or is nonfinite")
    scale = np.sqrt(np.diag(curvature))
    normalized = curvature/np.outer(scale, scale)
    asymmetry = float(np.max(abs(normalized-normalized.T), initial=0.))
    if asymmetry >= 1e-6:
        raise ValueError("Internal curvature is asymmetric")
    # Symmetrization is diagnostic only; invert all original entries unchanged.
    np.linalg.cholesky(.5*(normalized+normalized.T))
    inverse = np.linalg.inv(normalized)/np.outer(scale, scale)
    identity_error = float(np.max(abs(curvature@inverse-np.eye(len(free))), initial=0.))
    internal_covariance = 2.*errordef*inverse
    if not np.isfinite(internal_covariance).all() or np.any(np.diag(internal_covariance) <= 0):
        raise ValueError("Unstable internal covariance")
    sigma = np.sqrt(np.diag(internal_covariance))
    correlation = internal_covariance/np.outer(sigma, sigma)
    covariance_asymmetry = float(np.max(abs(correlation-correlation.T), initial=0.))
    np.linalg.cholesky(.5*(correlation+correlation.T))
    if identity_error >= 1e-6 or covariance_asymmetry >= 1e-6:
        raise ValueError("Unstable internal covariance inverse")
    covariance = np.zeros((size, size), dtype=np.float64)
    covariance[np.ix_(free, free)] = internal_covariance*np.outer(jacobian, jacobian)
    errors = np.zeros(size, dtype=np.float64)
    for i, u, x, step in zip(free, internal, external, sigma):
        low, high = bounds[i]
        if low is None and high is None:
            errors[i] = step
            continue
        up = transform_coordinate(u+step, bounds[i])[0]-x
        down = transform_coordinate(u-step, bounds[i])[0]-x
        if low is not None and high is not None and step > 1.:
            up = high-low
        errors[i] = .5*(abs(up)+abs(down))
    if not np.isfinite(covariance).all() or not np.isfinite(errors).all():
        raise ValueError("Nonfinite external covariance or errors")
    return dict(internal_hessian=curvature, internal_covariance=internal_covariance,
                external_covariance=covariance, error_fields=errors,
                jacobian=jacobian, second_derivative=second,
                gradient_chain_rule_diagonal=correction,
                internal_point_roundtrip_max_absolute_error=float(
                    np.max(abs(external-point[free]), initial=0.)),
                inverse_identity_max_absolute_error=identity_error,
                internal_curvature_normalized_asymmetry=asymmetry,
                internal_covariance_normalized_asymmetry=covariance_asymmetry,
                errordef=float(errordef))
