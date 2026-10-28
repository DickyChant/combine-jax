"""Reuse one trusted executable within a bounded block or reference-point batch.

This changes loading lifetime only. Each result is synchronized into owned host
arrays, and all columns/points are still evaluated. Benchmark callers retain
responsibility for process resource guards and complete-array validation.
"""

import hashlib
import time

import jax
from jax.experimental import serialize_executable
import numpy as np

from combine_jax.local import release_compilation_cache
from spooled_executable import SpooledExecutable


class LoadedExecutable:
    """Load a process-created SpooledExecutable once, then release it on exit."""

    def __init__(self, stored):
        if not isinstance(stored, SpooledExecutable):
            raise TypeError("Scoped loading requires a process-created SpooledExecutable")
        self.stored = stored
        self.compiled = None
        self.records = []
        self.entered = False

    def __enter__(self):
        if self.entered or getattr(self.stored, "_scoped_active", False):
            raise RuntimeError("An executable load scope cannot be reused or nested")
        self.entered = True
        started = time.perf_counter()
        payload = self.stored.path.read_bytes()
        if len(payload) != self.stored.size or hashlib.sha256(payload).hexdigest() != self.stored.sha256:
            raise ValueError("Serialized executable integrity check failed")
        self.read_seconds = time.perf_counter()-started
        started = time.perf_counter()
        self.compiled = serialize_executable.deserialize_and_load(
            payload, self.stored.in_tree, self.stored.out_tree, backend=self.stored.backend)
        self.load_seconds = time.perf_counter()-started
        self.stored._scoped_active = True
        return self

    def __call__(self, *args, **kwargs):
        if self.compiled is None:
            raise RuntimeError("Executable scope is not open")
        started = time.perf_counter()
        result = self.compiled(*args, **kwargs)
        host = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), result)
        execution = time.perf_counter()-started
        del result
        self.records.append(dict(read_and_verify_seconds=0., deserialize_seconds=0.,
                                 execute_and_copy_seconds=execution, cleanup_seconds=0.))
        return host

    def __exit__(self, exception_type, exception, tb):
        self.compiled = None
        started = time.perf_counter()
        release_compilation_cache()
        cleanup = time.perf_counter()-started
        self.stored._scoped_active = False
        if self.records:
            self.records[0].update(read_and_verify_seconds=self.read_seconds,
                                   deserialize_seconds=self.load_seconds)
            self.records[-1]["cleanup_seconds"] = cleanup
            self.stored.calls.extend(self.records)
        if not hasattr(self.stored, "scoped_loads"):
            self.stored.scoped_loads = []
        self.stored.scoped_loads.append(dict(
            calls=len(self.records), read_and_verify_seconds=self.read_seconds,
            deserialize_seconds=self.load_seconds, cleanup_seconds=cleanup,
            exception_type=exception_type.__name__ if exception_type else None))
        return False


def hessian_with_scoped_loads(evaluator, theta):
    """Assemble every Hessian entry, loading once per nonempty local term."""
    theta = evaluator._point(theta)
    result = np.zeros((len(theta), len(theta)), dtype=np.float64)
    for term in evaluator.terms:
        size = len(term.indices)
        if not size:
            continue
        name = "hessian_columns" if evaluator.hessian_chunk_size is not None else "hessian"
        with LoadedExecutable(getattr(term, name)) as kernel:
            if evaluator.hessian_chunk_size is None:
                result[np.ix_(term.indices, term.indices)] += kernel(theta[term.indices])
            else:
                chunk = min(evaluator.hessian_chunk_size, size)
                for first in range(0, size, chunk):
                    width = min(chunk, size-first)
                    directions = np.zeros((chunk, size), dtype=np.float64)
                    directions[np.arange(width), first+np.arange(width)] = 1.
                    columns = kernel(theta[term.indices], directions)[:width].T
                    result[np.ix_(term.indices, term.indices[first:first+width])] += columns
    return result


def value_gradient_points_with_scoped_loads(evaluator, points):
    """Evaluate a reference batch in global term order, loading each term once.

    These batched validation calls must not be reported as independent full-call
    latency. The benchmark's separately timed calls retain their stated policy.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or not len(points):
        raise ValueError("Reference points must be a nonempty matrix")
    for point in points:
        evaluator._point(point)
    values = np.full(len(points), evaluator.likelihood.offset, dtype=np.float64)
    gradients = np.zeros_like(points)
    for term in evaluator.terms:
        with LoadedExecutable(term.value_and_grad) as kernel:
            for index, point in enumerate(points):
                value, gradient = kernel(point[term.indices])
                values[index] += float(value)
                gradients[index, term.indices] += np.asarray(gradient)
    return values, gradients
