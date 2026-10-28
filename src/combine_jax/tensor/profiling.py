"""Fixed-parameter profile fits with the complete scan loop on the device."""


PROFILE_FIELDS = ('status', 'value', 'values', 'iteration', 'free_gradient',
    'projected_gradient', 'active_bounds', 'function_evaluations',
    'gradient_evaluations', 'line_search_evaluations', 'roundoff_accepts',
    'maximum_roundoff_allowance', 'estimated_edm', 'corrected_estimated_edm',
    'hessian_edm', 'edm_curvature_validated', 'edm_refinements',
    'edm_refinement_enabled', 'hessian_evaluations')


def make_profile_scan_function(function, **fit_options):
    """Return ``scan(stacked_coordinates, arrays)`` for an outer JIT.

    All points share the same free-coordinate layout and each fit starts from
    its supplied reference. The sequential device map keeps the live BFGS
    workspace bounded to one fit. It returns numerical status for every point;
    unsuccessful fits must not be used to construct a profile envelope.
    Matrices are omitted from the scan output. Projected-gradient fits and
    explicit estimated-EDM fits skip Hessian evaluation; exact-Hessian EDM
    validation computes curvature internally and can refine a smooth fit.
    EDM diagnostics and Hessian counts are retained for every point. Estimated
    EDM alone does not certify curvature or a minimum at a clipping surface.
    No parameter vectors or intermediate fit states are copied to the host.
    """
    import jax
    from .fitting import make_fit_function
    if 'covariance' in fit_options:
        raise ValueError('Profile scans omit final covariance calculations')
    fit = make_fit_function(function, covariance=False, **fit_options)

    def scan(coordinates, arrays):
        def point(c):
            result = fit(c, arrays)
            return {name: result[name] for name in PROFILE_FIELDS}
        return jax.lax.map(point, coordinates)
    return scan


def prepare_profile_scan(model, parameter, values, *, initial=None, fixed=None, **fit_options):
    """Validate a physical scan grid and prepare one compiled device scan.

    Return ``(compiled_scan, stacked_coordinates, model_arrays)``. Dispatch as
    ``compiled_scan(stacked_coordinates, model_arrays)``; results stay resident.
    Reuse the executable for new runtime grids with the same length/layout.
    The reference vector is the imported initial point unless ``initial`` is
    supplied, for example a previously accepted minimum. Every point uses that
    same nuisance start; this is not a guarantee of finding a global minimum.
    Discrete choices are frozen in each model and scanned as separate branches.
    """
    import jax
    import numpy as np
    from .model import make_model_function
    from .parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS
    info = model_parameter_info(model)
    names = info.metadata['names']
    if parameter not in names:
        raise ValueError(f'Unknown profile parameter: {parameter}')
    index = names.index(parameter)
    if info.arrays['lower'][index] == info.arrays['upper'][index]:
        raise ValueError(f'Cannot profile an imported fixed parameter: {parameter}')
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all():
        raise ValueError('Profile grid must be a finite nonempty one-dimensional array')
    if np.any(grid < info.arrays['lower'][index]) or np.any(grid > info.arrays['upper'][index]):
        raise ValueError('Profile grid exceeds the native physical bounds')
    fixed = dict(fixed or {})
    if parameter in fixed:
        raise ValueError('Profile parameter also appears in the additional fixed mapping')
    rows = [fit_coordinates(info, initial=initial, fixed={**fixed, parameter: float(value)})
            for value in grid]
    coordinates = {name: np.stack([row[name] for row in rows]) for name in COORDINATE_FIELDS}
    arrays = {name: value for name, value in model.arrays.items() if not name.startswith('parameter_info__')}
    function = make_profile_scan_function(make_model_function(model), **fit_options)
    return jax.jit(function), jax.device_put(coordinates), jax.device_put(arrays)
