"""Minuit gradient fits and physical-coordinate covariance from the AD Hessian."""

from dataclasses import dataclass
import time

from iminuit import Minuit
import numpy as np


@dataclass(frozen=True)
class FitResult:
    parameter_names: tuple[str, ...]
    values: np.ndarray
    nll: float
    gradient: np.ndarray
    hessian: np.ndarray
    covariance: np.ndarray | None
    covariance_status: str
    fixed: dict
    at_bounds: tuple[str, ...]
    valid_minimum: bool
    edm: float
    function_calls: int
    gradient_calls: int
    timings: dict

    @property
    def errors(self):
        return None if self.covariance is None else np.sqrt(np.diag(self.covariance))


def fit(likelihood, *, initial=None, fixed=None, evaluator=None, max_calls=10000,
        tolerance=1e-4, strategy=1, covariance=True, memory_policy="retain"):
    """Fit active continuous parameters with optional fixed physical values.

    Reuse an evaluator across profile fits to reuse its compiled kernels.
    Nonpositive free-parameter curvature is reported without constructing a
    pseudoinverse. Fixed coordinates have zero covariance. At-bound coordinates
    are reported separately because inverse curvature is a local approximation.
    The opt-in "release" memory policy retains explicitly prepared gradient
    kernels while clearing compiler caches, then drops them before streaming the
    complete Hessian. It clears process-wide JAX caches and recompiles gradients
    for later profile fits. No curvature entries are omitted.
    """
    if memory_policy not in ("retain","release"):
        raise ValueError("memory_policy must be 'retain' or 'release'")
    started = time.perf_counter()
    evaluator = likelihood.local_evaluator() if evaluator is None else evaluator
    if evaluator.parameter_names != likelihood.parameter_names:
        raise ValueError("Evaluator parameter order does not match the likelihood")
    names = likelihood.parameter_names
    point = np.array(likelihood.initial if initial is None else initial, dtype=np.float64, copy=True)
    if point.shape != likelihood.initial.shape or not np.all(np.isfinite(point)):
        raise ValueError("Initial point must be finite and match the parameter vector")
    fixed = dict(fixed or {})
    unknown = fixed.keys()-set(names)
    if unknown:
        raise ValueError(f"Unknown fixed parameters: {sorted(unknown)}")
    for parameter in likelihood.parameters:
        if parameter.minimum == parameter.maximum:
            fixed.setdefault(parameter.name, parameter.minimum)
    for name, value in fixed.items():
        point[names.index(name)] = float(value)
    for parameter, value in zip(likelihood.parameters, point, strict=True):
        if not np.isfinite(value) or not parameter.minimum <= value <= parameter.maximum:
            raise ValueError(f"Initial/fixed value outside the range of {parameter.name}")
    free = np.array([i for i, n in enumerate(names) if n not in fixed], dtype=np.intp)
    cached_point, cached_result = None, None

    def evaluate(values):
        nonlocal cached_point, cached_result
        array = np.asarray(values, dtype=np.float64)
        if cached_point is None or not np.array_equal(array, cached_point):
            cached_result = evaluator.value_and_grad(array)
            cached_point = array.copy()
        return cached_result

    preparation = []
    if memory_policy == "release":
        preparation = evaluator.prepare("value_and_grad",point,release_cache=True)
    first_value, first_gradient = evaluate(point)
    if not np.isfinite(first_value) or not np.all(np.isfinite(first_gradient)):
        raise ValueError("Initial point has a nonfinite NLL or gradient")
    timings = {"setup_and_first_value_gradient_seconds": time.perf_counter()-started}
    if preparation:
        timings.update(value_gradient_lowering_seconds=sum(r["lowering_seconds"] for r in preparation),
                       value_gradient_compile_seconds=sum(r["compile_seconds"] for r in preparation),
                       value_gradient_cache_release_seconds=sum(r["release_seconds"] for r in preparation))
    started = time.perf_counter()
    if len(free):
        minuit = Minuit(lambda *values: float(evaluate(values)[0]), *point, name=names,
                        grad=lambda *values: np.asarray(evaluate(values)[1]))
        minuit.errordef = .5
        minuit.tol = tolerance
        minuit.strategy = strategy
        for parameter in likelihood.parameters:
            minuit.limits[parameter.name] = (parameter.minimum, parameter.maximum)
            step = parameter.error if parameter.error > 0 else .1*max(abs(parameter.value), 1.)
            if parameter.minimum != parameter.maximum:
                minuit.errors[parameter.name] = min(step, .1*(parameter.maximum-parameter.minimum))
            minuit.fixed[parameter.name] = parameter.name in fixed
        minuit.migrad(ncall=max_calls)
        point = np.array(minuit.values)
        valid = bool(minuit.valid)
        edm = float(minuit.fmin.edm)
        calls, gradient_calls = minuit.nfcn, minuit.ngrad
    else:
        valid, edm, calls, gradient_calls = True, 0., 0, 0
    timings["minimization_seconds"] = time.perf_counter()-started
    value, gradient = evaluator.value_and_grad(point)
    if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
        valid = False
    hessian = np.empty((0, 0))
    covariance_matrix = None
    covariance_status = "not_requested"
    started = time.perf_counter()
    if covariance:
        if memory_policy == "release":
            evaluator.clear_cache(release_cache=True)
            hessian = evaluator.hessian(point,retain_compiled=False,release_cache=True)
        else:
            hessian = evaluator.hessian(point, retain_compiled=False)
        free_hessian = hessian[np.ix_(free, free)]
        if not np.all(np.isfinite(free_hessian)):
            covariance_status = "nonfinite_curvature"
        elif len(free) and np.linalg.eigvalsh(free_hessian).min() <= 0:
            covariance_status = "nonpositive_curvature"
        else:
            covariance_matrix = np.zeros_like(hessian)
            covariance_matrix[np.ix_(free, free)] = np.linalg.inv(free_hessian)
            covariance_status = "positive_definite" if len(free) else "all_fixed"
    timings["hessian_and_covariance_seconds"] = time.perf_counter()-started
    at_bounds = []
    for p, coordinate in zip(likelihood.parameters, point, strict=True):
        scale = max(abs(coordinate), p.error, 1.)
        if min(coordinate-p.minimum, p.maximum-coordinate) <= 1e-7*scale:
            at_bounds.append(p.name)
    return FitResult(names, point, float(value), np.asarray(gradient), np.asarray(hessian),
                     covariance_matrix, covariance_status, fixed, tuple(at_bounds),
                     valid, edm, calls, gradient_calls, timings)


def profile(likelihood, parameter, values, *, best_fit=None, evaluator=None, **fit_options):
    """Reoptimise other parameters at each requested fixed physical value."""
    if parameter not in likelihood.parameter_names:
        raise ValueError(f"Unknown profile parameter {parameter}")
    held = dict(fit_options.pop("fixed", {}))
    if "initial" in fit_options:
        raise ValueError("Pass an initial point to fit(), then provide its best_fit to profile()")
    evaluator = likelihood.local_evaluator() if evaluator is None else evaluator
    if best_fit is None:
        best_fit = fit(likelihood, fixed=held, evaluator=evaluator, **fit_options)
    elif held and held != best_fit.fixed:
        raise ValueError("Fixed settings differ from the profile baseline")
    if best_fit.parameter_names != likelihood.parameter_names:
        raise ValueError("Profile baseline parameter order does not match")
    if parameter in best_fit.fixed:
        raise ValueError("Profile parameter is fixed in the baseline")
    if not best_fit.valid_minimum:
        raise ValueError("Profile baseline is not a valid minimum")
    point = best_fit.values
    result = []
    for value in values:
        fitted = fit(likelihood, initial=point, fixed={**best_fit.fixed, parameter: float(value)}, evaluator=evaluator, **fit_options)
        result.append((float(value), fitted.nll-best_fit.nll, fitted))
        if fitted.valid_minimum:
            point = fitted.values
    return best_fit, result
