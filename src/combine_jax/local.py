"""Compile independent likelihood terms without bins-by-global-parameter arrays."""

from dataclasses import dataclass
import ctypes
import gc
import operator
import time

import jax
import numpy as np

from .discrete import active_dependencies


def release_compilation_cache():
    """Opt-in process-wide JAX cache release; explicit executables stay usable."""
    jax.clear_caches()
    gc.collect()
    trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
    if trim is not None:
        trim(0)


@dataclass
class LocalTerm:
    name: str
    parameter_names: tuple[str, ...]
    indices: np.ndarray
    constraints: tuple[str, ...]
    profiles: tuple
    function: object
    hessian_chunk_size: int | None = None

    def __post_init__(self):
        for name in ("value","value_and_grad","hessian","hessian_columns"):
            self.reset_kernel(name)

    def reset_kernel(self, name):
        previous = getattr(self,name,None)
        clear = getattr(previous,"clear_cache",None)
        if clear is not None:
            clear()
        if name == "value":
            kernel = jax.jit(self.function)
        elif name == "value_and_grad":
            kernel = jax.jit(jax.value_and_grad(self.function))
        elif name == "hessian":
            kernel = jax.jit(jax.hessian(self.function))
        elif name == "hessian_columns" and self.hessian_chunk_size is not None:
            gradient = jax.grad(self.function)

            def columns(theta, directions):
                return jax.vmap(lambda direction: jax.jvp(gradient, (theta,), (direction,))[1])(directions)

            kernel = jax.jit(columns)
        elif name == "hessian_columns":
            kernel = None
        else:
            raise ValueError(f"Unknown local kernel {name}")
        setattr(self,name,kernel)


class LocalEvaluator:
    """Host API returning NumPy derivatives in the likelihood's global order.

    Each JAX kernel sees only the parameters on which that term depends.
    Profiled-bin constraints are grouped with their owning channel, so their
    derivatives include the same analytic solve without counting it twice.
    The original Likelihood retains its pure JAX API for direct transformations.
    """

    def __init__(self, likelihood, *, constraint_batch_size=1, hessian_chunk_size=None):
        self.constraint_batch_size = operator.index(constraint_batch_size)
        if self.constraint_batch_size < 1:
            raise ValueError("constraint_batch_size must be positive")
        self.hessian_chunk_size = None if hessian_chunk_size is None else operator.index(hessian_chunk_size)
        if self.hessian_chunk_size is not None and self.hessian_chunk_size < 1:
            raise ValueError("hessian_chunk_size must be positive")
        self.likelihood = likelihood
        self.parameter_names = likelihood.parameter_names
        self.initial = likelihood.initial.copy()
        self.terms = self._terms()

    def _terms(self):
        likelihood = self.likelihood
        nodes = likelihood.nodes
        channels = [c for c in likelihood.channels if not c.masked]
        dependencies = [active_dependencies(nodes, [c.pdf]) for c in channels]
        profile_by_parameter = {n: i for i, p in enumerate(likelihood.profiles) for n in p.parameters}
        owners = {}
        for i, profile in enumerate(likelihood.profiles):
            owners[i] = next((j for j, deps in enumerate(dependencies) if profile.propagator in deps), None)
        grouped = [[] for _ in channels]
        separate = []
        for name in likelihood.constraints:
            deps = active_dependencies(nodes, [name])
            constraint_owners = {owners[profile_by_parameter[n]] for n in deps if n in profile_by_parameter}
            if len(constraint_owners) == 1 and None not in constraint_owners:
                grouped[next(iter(constraint_owners))].append(name)
            else:
                separate.append(name)
        specifications = [(c.name, c, tuple(cs)) for c, cs in zip(channels, grouped, strict=True)]
        for first in range(0, len(separate), self.constraint_batch_size):
            batch = tuple(separate[first:first+self.constraint_batch_size])
            label = "constraint:"+batch[0] if len(batch) == 1 else f"constraints:{first}:{first+len(batch)}"
            specifications.append((label, None, batch))
        terms = []
        for label, channel, constraints in specifications:
            roots = ([channel.pdf] if channel is not None else [])+list(constraints)
            deps = active_dependencies(nodes, roots)
            profile_indices = set()
            while True:
                added = {profile_by_parameter[n] for n in deps if n in profile_by_parameter}-profile_indices
                if not added:
                    break
                profile_indices.update(added)
                for i in added:
                    deps.update(active_dependencies(nodes, [likelihood.profiles[i].propagator]))
            indices = np.array([i for i, n in enumerate(self.parameter_names) if n in deps], dtype=np.intp)
            indices.flags.writeable = False
            names = tuple(self.parameter_names[i] for i in indices)
            profiles = tuple(likelihood.profiles[i] for i in sorted(profile_indices))
            function = self._function(channel, constraints, names, profiles)
            terms.append(LocalTerm(label, names, indices, constraints, profiles, function, self.hessian_chunk_size))
        return tuple(terms)

    def _function(self, channel, constraints, names, profiles):
        likelihood = self.likelihood

        def evaluate(theta):
            overrides = likelihood.profile_values(theta, parameter_names=names, profiles=profiles)
            value = likelihood._constraint_value(theta, overrides, names, constraints)
            if channel is not None:
                value += likelihood._channel_value(channel, theta, overrides, names)
            return value

        return evaluate

    def _point(self, theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != self.initial.shape:
            raise ValueError(f"Expected parameter shape {self.initial.shape}, got {theta.shape}")
        return theta

    def value(self, theta):
        theta = self._point(theta)
        return self.likelihood.offset+sum(float(t.value(theta[t.indices])) for t in self.terms)

    def value_and_grad(self, theta):
        theta = self._point(theta)
        value = self.likelihood.offset
        gradient = np.zeros_like(theta)
        for term in self.terms:
            local_value, local_gradient = term.value_and_grad(theta[term.indices])
            value += float(local_value)
            gradient[term.indices] += np.asarray(local_gradient)
        return value, gradient

    def prepare(self, operation="value_and_grad", theta=None, *, release_cache=False):
        """Retain explicit executables, optionally releasing compiler caches per term.

        Clearing caches affects this Python process's JAX caches. Use this opt-in
        mode when bounding compilation memory matters more than retaining other
        JIT traces. Returned records separate lowering, compilation and cleanup.
        """
        if operation not in ("value","value_and_grad","hessian"):
            raise ValueError(f"Unknown local operation {operation}")
        theta = self._point(self.initial if theta is None else theta)
        records = []
        for term in self.terms:
            name = operation
            arguments = (theta[term.indices],)
            if operation == "hessian" and self.hessian_chunk_size is not None:
                size = len(term.indices)
                if not size:
                    continue
                name = "hessian_columns"
                arguments += (np.zeros((min(self.hessian_chunk_size,size),size)),)
            kernel = getattr(term,name)
            if not hasattr(kernel,"lower"):
                records.append(dict(name=term.name,operation=operation,reused=True,
                                    lowering_seconds=0.,compile_seconds=0.,release_seconds=0.))
                continue
            started = time.perf_counter()
            lowered = kernel.lower(*arguments)
            lowering = time.perf_counter()-started
            started = time.perf_counter()
            compiled = lowered.compile()
            compilation = time.perf_counter()-started
            setattr(term,name,compiled)
            del kernel,lowered,compiled
            started = time.perf_counter()
            if release_cache:
                release_compilation_cache()
            records.append(dict(name=term.name,operation=operation,reused=False,
                                lowering_seconds=lowering,compile_seconds=compilation,
                                release_seconds=time.perf_counter()-started if release_cache else 0.))
        return records

    def hessian(self, theta, *, retain_compiled=True, release_cache=False):
        if release_cache and retain_compiled:
            raise ValueError("release_cache requires retain_compiled=False")
        theta = self._point(theta)
        result = np.zeros((len(theta), len(theta)), dtype=np.float64)
        for term in self.terms:
            if self.hessian_chunk_size is None:
                name = "hessian"
                kernel = term.hessian
                result[np.ix_(term.indices, term.indices)] += np.asarray(kernel(theta[term.indices]))
            else:
                size = len(term.indices)
                if not size:
                    continue
                chunk = min(self.hessian_chunk_size, size)
                name = "hessian_columns"
                kernel = term.hessian_columns
                for first in range(0, size, chunk):
                    width = min(chunk, size-first)
                    directions = np.zeros((chunk, size), dtype=np.float64)
                    directions[np.arange(width), first+np.arange(width)] = 1.
                    columns = np.asarray(kernel(theta[term.indices], directions))[:width].T
                    result[np.ix_(term.indices, term.indices[first:first+width])] += columns
            if not retain_compiled:
                term.reset_kernel(name)
                del kernel
                if release_cache:
                    release_compilation_cache()
        return result

    def clear_cache(self, *, release_cache=False):
        """Release JIT or explicit kernels and restore callable lazy kernels."""
        for term in self.terms:
            for name in ("value","value_and_grad","hessian","hessian_columns"):
                term.reset_kernel(name)
        if release_cache:
            release_compilation_cache()
