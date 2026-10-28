"""Retain exact local gradient kernels across bounded, separately spawned CPUs.

This experimental host evaluator changes executable ownership, not likelihood
terms. Workers return individual term results for original-order accumulation.
It deliberately has no partial-Hessian or covariance implementation.
"""

import hashlib
import json
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np


NO_FUSION_OPTIONS = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def core_digest():
    return hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()


def rss(pid):
    try:
        lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        return float(next(line.split()[1] for line in lines if line.startswith("VmRSS:")))/1024
    except (FileNotFoundError, ProcessLookupError, StopIteration):
        return 0.


def descendants(pid):
    found, pending = set(), [pid]
    while pending:
        parent = pending.pop()
        try:
            children = [int(p) for p in Path(f"/proc/{parent}/task/{parent}/children").read_text().split()]
        except (FileNotFoundError, ProcessLookupError):
            children = []
        for child in children:
            if child not in found:
                found.add(child)
                pending.append(child)
    return found


def _worker(index, count, core, specification, compiler_options, connection):
    try:
        os.sched_setaffinity(0, [core])
        os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                          OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        import jax
        import jaxlib
        from combine_jax.local import release_compilation_cache
        from hhvv_export_cache import load_export

        jax.config.update("jax_enable_x64", True)
        jax.config.update("jax_enable_compilation_cache", False)
        assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
        assert core_digest() == specification["core_sha256"]
        assert digest(specification["export"]) == specification["export_sha256"]
        likelihood = load_export(specification["export"], specification["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32)
        selected = [(i, term) for i, term in enumerate(evaluator.terms) if i % count == index]
        limit = int(Path("/proc/sys/vm/max_map_count").read_text())
        metadata = dict(worker=index, pid=os.getpid(), core=core, jax_version=jax.__version__,
            jaxlib_version=jaxlib.__version__, float64=True, devices=[str(d) for d in jax.devices()],
            root_imported="ROOT" in sys.modules,
            gradient_compiler_options=compiler_options,
            core_sha256=core_digest(), helper_sha256=digest(__file__), mapping_limit=limit,
            mapping_guard=limit-4096, offset=likelihood.offset,
            discrete_choices=likelihood.discrete_choices,
            mask_configuration=likelihood.mask_configuration,
            parameter_names=list(likelihood.parameter_names), initial=likelihood.initial.tolist(),
            parameters=[dict(name=p.name, value=p.value, error=p.error, minimum=p.minimum, maximum=p.maximum)
                        for p in likelihood.parameters],
            terms_total=len(evaluator.terms),
            terms=[dict(index=i, name=term.name, indices=term.indices.tolist()) for i, term in selected])
        connection.send(dict(kind="result", operation="initialize", worker=index, payload=metadata))
        assert not metadata["root_imported"], "Numerical workers must not import ROOT"
        prepared = False
        while True:
            request = connection.recv()
            if request["operation"] == "close":
                return
            if request["operation"] == "prepare":
                assert not prepared
                point = np.asarray(request["point"], dtype=np.float64)
                records = []
                for global_index, term in selected:
                    mappings = len(Path("/proc/self/maps").read_text().splitlines())
                    if mappings >= metadata["mapping_guard"]:
                        raise MemoryError("Worker mapping guard reached before compilation")
                    started = time.perf_counter()
                    lowered = term.value_and_grad.lower(point[term.indices])
                    lower_seconds = time.perf_counter()-started
                    started = time.perf_counter()
                    compiled = lowered.compile(compiler_options=compiler_options)
                    compile_seconds = time.perf_counter()-started
                    term.value_and_grad = compiled
                    del compiled, lowered
                    started = time.perf_counter()
                    release_compilation_cache()
                    record = dict(index=global_index, name=term.name, operation="value_and_grad", reused=False,
                        lowering_seconds=lower_seconds, compile_seconds=compile_seconds,
                        release_seconds=time.perf_counter()-started, rss_mib=rss(os.getpid()),
                        mappings=len(Path("/proc/self/maps").read_text().splitlines()))
                    records.append(record)
                    connection.send(dict(kind="progress", operation="prepare", worker=index, payload=record))
                    if record["mappings"] >= metadata["mapping_guard"]:
                        raise MemoryError("Worker mapping guard reached after compilation")
                prepared = True
                connection.send(dict(kind="result", operation="prepare", worker=index, payload=records))
            elif request["operation"] == "value_and_grad":
                assert prepared
                point = np.asarray(request["point"], dtype=np.float64)
                results = []
                for global_index, term in selected:
                    value, gradient = term.value_and_grad(point[term.indices])
                    results.append((global_index, float(value), np.array(gradient, copy=True)))
                connection.send(dict(kind="result", operation="value_and_grad", worker=index,
                                     call=request["call"], payload=results))
            else:
                raise ValueError(f"Unsupported worker operation: {request['operation']}")
    except EOFError:
        pass
    except BaseException as error:
        try:
            connection.send(dict(kind="error", worker=index, exception_type=type(error).__name__,
                                 exception=traceback.format_exc()))
        finally:
            raise
    finally:
        connection.close()


class ShardedCpuEvaluator:
    def __init__(self, specification, *, workers=3, rss_limit_mib=10500., wall_seconds=21600.,
                 on_progress=None, cpu_no_fusion=False):
        if (not 1 <= workers <= 7 or not 0 < rss_limit_mib <= 11000 or wall_seconds <= 0
                or not np.isfinite([rss_limit_mib, wall_seconds]).all()):
            raise ValueError("Use 1–7 workers and positive limits with aggregate RSS <=11000 MiB")
        if Path.cwd() == Path("/tmp/sqian/combine-jax"):
            for policy in ("spooled", "scoped"):
                for phase in ("smoke", "hhvv"):
                    p = Path(f"reports/gpu_{policy}_{phase}_driver.json")
                    if p.exists() and json.loads(p.read_bytes())["status"] == "running":
                        raise RuntimeError("Finish the active benchmark before other numerical jobs")
        self.specification = specification
        self.compiler_options = dict(NO_FUSION_OPTIONS) if cpu_no_fusion else {}
        self.original_affinity = sorted(os.sched_getaffinity(0))
        if len(self.original_affinity) < workers+1:
            raise ValueError("Reserve a separate core for the parent, within the first eight cores")
        self.cores = self.original_affinity[:workers+1]
        self.rss_limit_mib, self.wall_seconds = rss_limit_mib, wall_seconds
        self.on_progress = on_progress
        self.started = time.monotonic()
        self.processes, self.connections, self.worker_metadata = [], [], []
        self.progress, self.preparation = {}, []
        self.aggregate_rss_peak_mib = 0.
        self.worker_mapping_peaks = {}
        self.last_mapping_sample = -np.inf
        self.calls, self.prepared, self.closed = 0, False, False
        self.cleanup = []
        os.sched_setaffinity(0, [self.cores[-1]])
        context = mp.get_context("spawn")
        try:
            for index in range(workers):
                parent, child = context.Pipe()
                process = context.Process(target=_worker,
                    args=(index, workers, self.cores[index], specification, self.compiler_options, child),
                    name=f"combine-jax-cpu-{index}")
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            results = self._receive("initialize")
            self.worker_metadata = [results[i] for i in range(workers)]
            first = self.worker_metadata[0]
            for worker in self.worker_metadata:
                for key in ("parameter_names", "initial", "parameters", "offset", "terms_total",
                            "core_sha256", "helper_sha256", "jax_version", "jaxlib_version", "float64",
                            "root_imported", "discrete_choices", "mask_configuration", "gradient_compiler_options"):
                    assert worker[key] == first[key], key
            self.term_metadata = sorted([term for worker in self.worker_metadata for term in worker["terms"]],
                                        key=lambda term: term["index"])
            assert [t["index"] for t in self.term_metadata] == list(range(first["terms_total"]))
            assert len({t["name"] for t in self.term_metadata}) == first["terms_total"]
            self.parameter_names = tuple(first["parameter_names"])
            self.initial = np.asarray(first["initial"], dtype=np.float64)
            self.offset = first["offset"]
            self.likelihood = SimpleNamespace(parameter_names=self.parameter_names, initial=self.initial,
                parameters=[SimpleNamespace(**p) for p in first["parameters"]],
                discrete_choices=first["discrete_choices"],
                mask_configuration=first["mask_configuration"])
        except BaseException:
            self.close(force=True)
            raise

    def snapshot(self):
        return dict(aggregate_rss_peak_mib=self.aggregate_rss_peak_mib,
                    worker_mapping_peaks=dict(self.worker_mapping_peaks), calls=self.calls,
                    progress=dict(self.progress), prepared=self.prepared,
                    worker_pids=[p.pid for p in self.processes],
                    cpu_affinity_parent=[self.cores[-1]], cpu_affinity_workers=self.cores[:-1])

    def _check_resources(self):
        elapsed = time.monotonic()-self.started
        processes = {os.getpid(), *descendants(os.getpid())}
        total = sum(rss(pid) for pid in processes)
        self.aggregate_rss_peak_mib = max(self.aggregate_rss_peak_mib, total)
        if elapsed > self.wall_seconds:
            raise TimeoutError("Sharded evaluator wall-time guard reached")
        if total > self.rss_limit_mib:
            raise MemoryError(f"Aggregate parent/descendant RSS guard reached: {total:.1f} MiB")
        if time.monotonic()-self.last_mapping_sample > 1.:
            self.last_mapping_sample = time.monotonic()
            for process in self.processes:
                try:
                    mappings = len(Path(f"/proc/{process.pid}/maps").read_text().splitlines())
                except (FileNotFoundError, ProcessLookupError):
                    continue
                self.worker_mapping_peaks[str(process.pid)] = max(
                    self.worker_mapping_peaks.get(str(process.pid), 0), mappings)
                if mappings >= int(Path("/proc/sys/vm/max_map_count").read_text())-4096:
                    raise MemoryError(f"Worker {process.pid} mapping guard reached")

    def _receive(self, operation, call=None):
        remaining = dict(enumerate(self.connections))
        results = {}
        while remaining:
            self._check_resources()
            ready = wait(list(remaining.values()), timeout=.1)
            for connection in ready:
                index = next(i for i, c in remaining.items() if c is connection)
                message = connection.recv()
                assert message["worker"] == index
                if message["kind"] == "error":
                    error_type = MemoryError if message["exception_type"] == "MemoryError" else RuntimeError
                    raise error_type(f"Worker {index} failed:\n{message['exception']}")
                assert message["operation"] == operation
                if message["kind"] == "progress":
                    self.progress[str(index)] = message["payload"]
                    if self.on_progress:
                        self.on_progress(self.snapshot())
                    continue
                assert message["kind"] == "result"
                if call is not None:
                    assert message["call"] == call
                results[index] = message["payload"]
                del remaining[index]
            for index, connection in remaining.items():
                if self.processes[index].exitcode is not None and not connection.poll():
                    raise RuntimeError(f"Worker {index} exited {self.processes[index].exitcode}")
        self._check_resources()
        return results

    def _point(self, theta):
        point = np.asarray(theta, dtype=np.float64)
        if point.shape != self.initial.shape:
            raise ValueError("Point shape differs from the original parameter vector")
        return point

    def prepare(self, operation="value_and_grad", theta=None, *, release_cache=True):
        if operation != "value_and_grad" or not release_cache:
            raise ValueError("This evaluator supports retained gradients with compiler-cache release only")
        if self.closed:
            raise RuntimeError("Evaluator is closed")
        if self.prepared:
            return [dict(name=r["name"], reused=True, lowering_seconds=0., compile_seconds=0.,
                         release_seconds=0.) for r in self.preparation]
        point = self._point(self.initial if theta is None else theta)
        for connection in self.connections:
            connection.send(dict(operation="prepare", point=point))
        results = self._receive("prepare")
        self.preparation = sorted([record for records in results.values() for record in records],
                                  key=lambda r: r["index"])
        assert [r["index"] for r in self.preparation] == list(range(len(self.term_metadata)))
        self.prepared = True
        return self.preparation

    def value_and_grad(self, theta):
        if self.closed or not self.prepared:
            raise RuntimeError("Prepare live workers before evaluating gradients")
        point = self._point(theta)
        self.calls += 1
        for connection in self.connections:
            connection.send(dict(operation="value_and_grad", call=self.calls, point=point))
        responses = self._receive("value_and_grad", self.calls)
        terms = sorted([record for records in responses.values() for record in records], key=lambda r: r[0])
        assert [r[0] for r in terms] == list(range(len(self.term_metadata)))
        value, gradient = self.offset, np.zeros_like(point)
        for (index, local_value, local_gradient), metadata in zip(terms, self.term_metadata, strict=True):
            indices = np.asarray(metadata["indices"], dtype=np.intp)
            assert local_gradient.shape == indices.shape and local_gradient.dtype == np.float64
            value += local_value
            gradient[indices] += local_gradient
        return value, gradient

    def hessian(self, *args, **kwargs):
        raise NotImplementedError("This gradient evaluator has no Hessian; use a separately validated full-matrix step")

    def close(self, *, force=False):
        if self.closed:
            return
        self.closed = True
        actions = {p.pid: [] for p in self.processes}
        for process, connection in zip(self.processes, self.connections, strict=True):
            if process.is_alive():
                if force:
                    process.terminate()
                    actions[process.pid].append("terminate")
                else:
                    try:
                        connection.send(dict(operation="close"))
                    except (BrokenPipeError, EOFError, OSError):
                        pass
        for process, connection in zip(self.processes, self.connections, strict=True):
            process.join(timeout=2.)
            forced = process.is_alive()
            if forced:
                process.kill()
                actions[process.pid].append("kill")
                process.join(timeout=2.)
            connection.close()
            self.cleanup.append(dict(pid=process.pid, exit_code=process.exitcode,
                                     signals=actions[process.pid], force_cleanup_requested=force))
        os.sched_setaffinity(0, self.original_affinity)

    def __enter__(self):
        return self

    def __exit__(self, kind, exception, tb):
        self.close(force=kind is not None)
