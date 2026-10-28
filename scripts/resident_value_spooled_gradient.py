"""Bounded CPU fit evaluation with resident values and disk-backed gradients."""

from pathlib import Path
import time

import jax
import numpy as np

from combine_jax.local import release_compilation_cache
from scoped_executable import LoadedExecutable
from spooled_executable import SpooledExecutable


class ResidentValueSpooledGradient:
    """Keep every NLL term resident; load one gradient executable at a time."""

    def __init__(self, likelihood, directory, *, gradient_compiler_options=None, on_progress=None):
        if jax.default_backend() != "cpu" or not jax.config.jax_enable_x64:
            raise ValueError("This evaluator requires CPU float64")
        if jax.config.jax_use_simplified_jaxpr_constants:
            raise ValueError("Captured-array arguments are unsupported by the installed serializer")
        self.directory = Path(directory)
        if self.directory.exists():
            raise FileExistsError("Preserve earlier process-created executables")
        self.directory.mkdir(parents=True)
        self.likelihood = likelihood
        self.parameter_names = likelihood.parameter_names
        self.initial = likelihood.initial
        self.local = likelihood.local_evaluator(constraint_batch_size=32)
        self.gradient_compiler_options = dict(gradient_compiler_options or {})
        self.on_progress = on_progress
        self.value_preparation, self.gradient_preparation = [], []
        self.prepared = self.closed = False
        self.value_calls = self.gradient_calls = 0
        self.value_seconds = self.gradient_seconds = 0.

    def notify(self, operation, index, term):
        if self.on_progress:
            self.on_progress(dict(operation=operation, index=index, name=term.name,
                                  terms=len(self.local.terms)))

    def prepare(self, points):
        if self.prepared or self.closed:
            raise RuntimeError("Prepare a fresh live evaluator once")
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or not len(points) or points.shape[1:] != self.initial.shape:
            raise ValueError("Use complete parameter vectors")
        if not np.isfinite(points).all():
            raise ValueError("Reference points must be finite")
        for index, term in enumerate(self.local.terms):
            self.notify("prepare_value", index, term)
            begin = time.perf_counter()
            lowered = term.value.lower(points[0, term.indices])
            lowering = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile()
            compilation = time.perf_counter()-begin
            term.value = compiled
            del compiled, lowered
            begin = time.perf_counter()
            release_compilation_cache()
            self.value_preparation.append(dict(index=index, name=term.name,
                lowering_seconds=lowering, compile_seconds=compilation,
                cleanup_seconds=time.perf_counter()-begin))
        values = np.full(len(points), self.likelihood.offset, dtype=np.float64)
        gradients = np.zeros_like(points)
        for index, term in enumerate(self.local.terms):
            self.notify("prepare_gradient", index, term)
            begin = time.perf_counter()
            lowered = term.value_and_grad.lower(points[0, term.indices])
            lowering = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile(compiler_options=self.gradient_compiler_options)
            compilation = time.perf_counter()-begin
            raw_value, raw_gradient = compiled(points[0, term.indices])
            nominal_value, nominal_gradient = float(raw_value), np.array(raw_gradient, copy=True)
            del raw_value, raw_gradient
            stored, serialization = SpooledExecutable.write(
                compiled, self.directory/f"{index}.bin", backend="cpu")
            term.value_and_grad = stored
            del compiled, lowered
            release_compilation_cache()
            self.notify("validate_loaded_gradient_points", index, term)
            begin = time.perf_counter()
            with LoadedExecutable(stored) as kernel:
                for point_index, point in enumerate(points):
                    value, gradient = kernel(point[term.indices])
                    if point_index == 0:
                        np.testing.assert_equal(float(value), nominal_value)
                        np.testing.assert_array_equal(gradient, nominal_gradient)
                    values[point_index] += float(value)
                    gradients[point_index, term.indices] += gradient
            self.gradient_preparation.append(dict(index=index, name=term.name,
                compiler_options=self.gradient_compiler_options,
                lowering_seconds=lowering, compile_seconds=compilation,
                validate_loaded_points_seconds=time.perf_counter()-begin,
                validation_points=len(points), nominal_reload_bitwise_equal=True,
                load_scope=stored.scoped_loads[0], **serialization))
        nll_values = np.asarray([self.local.value(point) for point in points])
        if not all(np.isfinite(array).all() for array in (values, gradients, nll_values)):
            raise ValueError("Nonfinite validation output")
        self.prepared = True
        return dict(validation_values=nll_values, validation_vg_values=values,
                    validation_gradients=gradients)

    def value(self, point):
        if not self.prepared or self.closed:
            raise RuntimeError("Prepare a live evaluator before fitting")
        begin = time.perf_counter()
        result = self.local.value(point)
        self.value_seconds += time.perf_counter()-begin
        self.value_calls += 1
        return result

    def value_and_grad(self, point):
        if not self.prepared or self.closed:
            raise RuntimeError("Prepare a live evaluator before fitting")
        begin = time.perf_counter()
        result = self.local.value_and_grad(point)
        self.gradient_seconds += time.perf_counter()-begin
        self.gradient_calls += 1
        return result

    def snapshot(self):
        return dict(value_calls=self.value_calls, gradient_calls=self.gradient_calls,
                    value_seconds=self.value_seconds, gradient_seconds=self.gradient_seconds,
                    prepared=self.prepared, closed=self.closed)

    def close(self):
        if not self.closed:
            self.closed = True
            self.local.clear_cache(release_cache=True)

    def __enter__(self):
        return self

    def __exit__(self, kind, exception, tb):
        self.close()


def fit_separate_calls(likelihood, evaluator, *, initial=None, max_calls=50000,
                       strategy=0, tolerance=1e-4, expected_initial_nll=None, on_evaluation=None,
                       migrad_iterate=5, use_simplex=True):
    """Request gradients only through Minuit's gradient callback; no covariance."""
    from iminuit import Minuit

    if (max_calls < 1 or strategy not in (0, 1, 2) or not 0 < tolerance < np.inf
            or not isinstance(migrad_iterate, int) or migrad_iterate < 1):
        raise ValueError("Use positive finite optimizer limits, positive integer iterations and strategy 0, 1 or 2")
    if evaluator.parameter_names != likelihood.parameter_names:
        raise ValueError("Evaluator order differs from the likelihood")
    point = np.asarray(likelihood.initial if initial is None else initial, dtype=np.float64)
    if point.shape != likelihood.initial.shape or not np.isfinite(point).all():
        raise ValueError("Invalid initial point")
    for value, parameter in zip(point, likelihood.parameters, strict=True):
        if not parameter.minimum <= value <= parameter.maximum:
            raise ValueError(f"Initial value outside bounds of {parameter.name}")
    if all(p.minimum == p.maximum for p in likelihood.parameters):
        raise NotImplementedError("This experimental minimum driver requires at least one free parameter")
    value_point = gradient_point = None
    cached_value = cached_gradient = None
    maximum_nll_discrepancy = 0.
    initial_checked = False
    nonfinite_values = nonfinite_gradients = 0

    def value(*coordinates):
        nonlocal value_point, cached_value, initial_checked, nonfinite_values
        theta = np.asarray(coordinates, dtype=np.float64)
        if value_point is None or not np.array_equal(theta, value_point):
            cached_value = float(evaluator.value(theta))
            value_point = theta.copy()
            if not np.isfinite(cached_value):
                nonfinite_values += 1
            if not initial_checked:
                if expected_initial_nll is not None:
                    assert np.isfinite(cached_value) and abs(cached_value-expected_initial_nll) < 1e-6
                initial_checked = True
            if on_evaluation:
                on_evaluation("value", theta, cached_value)
        return cached_value

    def gradient(*coordinates):
        nonlocal gradient_point, cached_gradient, maximum_nll_discrepancy, nonfinite_gradients
        theta = np.asarray(coordinates, dtype=np.float64)
        if gradient_point is None or not np.array_equal(theta, gradient_point):
            if on_evaluation:
                on_evaluation("gradient_start", theta, None)
            nll, derivative = evaluator.value_and_grad(theta)
            ordinary = value(*coordinates)
            if np.isfinite(nll) and np.isfinite(ordinary):
                error = abs(float(nll)-ordinary)
                maximum_nll_discrepancy = max(maximum_nll_discrepancy, error)
                assert error < 1e-6, "Resident value and streamed value-gradient disagree"
            cached_gradient = np.asarray(derivative)
            if not np.isfinite(cached_gradient).all() or not np.isfinite(nll):
                nonfinite_gradients += 1
            gradient_point = theta.copy()
            if on_evaluation:
                on_evaluation("gradient", theta, float(nll))
        return cached_gradient

    begin = time.perf_counter()
    initial_value = value(*point)
    initial_gradient = gradient(*point)
    if not np.isfinite(initial_value) or not np.isfinite(initial_gradient).all():
        raise ValueError("Initial NLL/gradient must be finite")
    initial_seconds = time.perf_counter()-begin
    objective_offset = initial_value

    def centered_value(*coordinates):
        return value(*coordinates)-objective_offset

    minuit = Minuit(centered_value, *point, name=likelihood.parameter_names, grad=gradient)
    minuit.errordef, minuit.tol, minuit.strategy = .5, tolerance, strategy
    fixed = {}
    for parameter in likelihood.parameters:
        minuit.limits[parameter.name] = (parameter.minimum, parameter.maximum)
        step = parameter.error if parameter.error > 0 else .1*max(abs(parameter.value), 1.)
        if parameter.minimum != parameter.maximum:
            minuit.errors[parameter.name] = min(step, .1*(parameter.maximum-parameter.minimum))
        else:
            fixed[parameter.name] = parameter.minimum
            minuit.fixed[parameter.name] = True
    begin = time.perf_counter()
    minuit.migrad(ncall=max_calls, iterate=migrad_iterate, use_simplex=use_simplex)
    minimization_seconds = time.perf_counter()-begin
    begin = time.perf_counter()
    fitted = np.asarray(minuit.values)
    final_value = value(*fitted)
    final_gradient = np.array(gradient(*fitted), copy=True)
    shifted_nll = float(minuit.fval)
    restoration_error = abs(final_value-(objective_offset+shifted_nll))
    if np.isfinite(final_value) and np.isfinite(shifted_nll):
        assert restoration_error < 1e-6, "Minuit objective does not restore the canonical NLL"
    valid = (bool(minuit.valid) and np.isfinite(final_value) and
             np.isfinite(shifted_nll) and np.isfinite(final_gradient).all())
    at_bounds = [p.name for p, x in zip(likelihood.parameters, fitted, strict=True)
                 if min(x-p.minimum, p.maximum-x) <= 1e-7*max(abs(x), p.error, 1.)]
    return dict(values=fitted, gradient=final_gradient, nll=final_value,
                valid_minimum=bool(valid), edm=float(minuit.fmin.edm),
                fixed=fixed, at_bounds=at_bounds, covariance_status="not_requested",
                function_calls=minuit.nfcn, gradient_calls=minuit.ngrad,
                initial_nll=initial_value,
                objective_offset=objective_offset, shifted_nll=shifted_nll,
                canonical_restoration_absolute_error=restoration_error,
                initial_reference_absolute_error=None if expected_initial_nll is None else
                    abs(initial_value-expected_initial_nll),
                maximum_resident_streamed_nll_difference=maximum_nll_discrepancy,
                nonfinite_value_calls=nonfinite_values, nonfinite_gradient_calls=nonfinite_gradients,
                timings=dict(initial_value_gradient_seconds=initial_seconds,
                             minimization_seconds=minimization_seconds,
                             final_value_gradient_seconds=time.perf_counter()-begin))
