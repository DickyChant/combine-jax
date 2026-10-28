"""Explicit recovery of our hash-bound LCG executables on the original node.

This is an experimental, separately validated path. It does not extend the
same-process contract of spooled_executable.py or assume cache portability.
Only artifacts from an explicitly hash-selected, trusted producer are accepted.
"""

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import jax
from jax.experimental import serialize_executable
import numpy as np

from combine_jax.local import release_compilation_cache


LCG = "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def checked_path(path):
    path = Path(path)
    if path.is_absolute() or ".." in path.parts or path.is_symlink():
        raise ValueError("Expected a regular relative artifact inside the project")
    if not path.resolve().is_relative_to(Path.cwd().resolve()) or not path.is_file():
        raise ValueError("Artifact must exist inside the authorized project")
    return path


def core_hash():
    return hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()


def checked_runtime(producer):
    devices = jax.devices()
    assert devices and all(d.platform == "gpu" for d in devices)
    assert jax.config.jax_enable_x64 and "ROOT" not in sys.modules
    assert platform.node() == producer["host"] == "lxplus905.cern.ch"
    assert str(Path.cwd()) == producer["workdir"] == "/tmp/sqian/combine-jax"
    assert os.environ.get("COMBINE_JAX_LCG_VIEW") == producer["lcg_view"] == LCG
    assert jax.__version__ == producer["jax_version"] == "0.10.1"
    assert __import__("jaxlib").__version__ == producer["jaxlib_version"] == "0.10.1"
    assert sys.version == producer["python"]
    assert [str(d) for d in devices] == producer["devices"] == ["cuda:0"]
    assert producer["backend"] == "gpu" and producer["float64"]
    assert os.environ.get("XLA_FLAGS") == producer["xla_flags"]
    assert not jax.config.jax_use_simplified_jaxpr_constants
    assert os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE", "").lower() == "false"
    assert os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default") == producer["gpu_allocator"]["allocator"]
    metadata = subprocess.check_output(["nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,uuid",
        "--format=csv,noheader"], text=True).strip()
    assert metadata == producer["gpu_metadata"]
    assert core_hash() == producer["source_sha256"]
    for key, name in [("benchmark_sha256", "gpu_benchmark.py"),
                      ("scoped_executable_sha256", "scoped_executable.py"),
                      ("spooled_executable_sha256", "spooled_executable.py")]:
        assert digest(Path("scripts")/name) == producer[key]
    assert producer["constraint_batch_size"] == 32 and producer["hessian_chunk_size"] == 4
    assert producer["hessian_compiler_options"] == {} and producer["scoped_loads"]
    return dict(host=platform.node(), workdir=str(Path.cwd()), python=sys.version,
        lcg_view=LCG, jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__,
        cuda_plugin_version=importlib.metadata.version("jax-cuda12-plugin"),
        devices=[str(d) for d in devices], float64=True, gpu_metadata=metadata,
        xla_flags=os.environ.get("XLA_FLAGS"), source_sha256=core_hash(),
        cpu_affinity=sorted(os.sched_getaffinity(0)))


def records(producer):
    if "hessian" in producer.get("completed_operations", []):
        result = producer["timings"]["hessian"]["term_timings"]
        assert len(result) == producer["terms"]
    else:
        progress = producer["compile_progress"]
        assert progress["operation"] == "hessian"
        result = progress["term_timings"]
        assert len(result) == progress["completed_terms"]
    return result


def checked_record(record, term, index, producer):
    assert record["name"] == term.name and record["parameters"] == len(term.indices)
    assert record["kernel"] == "hessian_columns" and record["compiler_options"] == {}
    if not len(term.indices):
        assert record["skipped_empty_hessian"]
        return
    expected = Path("build/gpu_executables")/(
        producer["model"]+"_gpu_"+producer["report_tag"])/"hessian"/f"{index}.bin"
    path = checked_path(record["executable"])
    assert path == expected
    assert path.stat().st_size == record["serialized_bytes"]
    assert digest(path) == record["executable_sha256"]


class RecoveredExecutable:
    """Reconstruct PyTrees by lowering the same function, without recompiling."""

    def __init__(self, record, term, point):
        start = time.perf_counter()
        size = len(term.indices)
        lowered = term.hessian_columns.lower(point[term.indices], np.zeros((min(4, size), size)))
        self.in_tree, self.out_tree = lowered.in_tree, lowered.out_tree
        self.lowering_seconds = time.perf_counter()-start
        self.record = record
        self.loads = []
        self.compiled = None
        del lowered
        term.reset_kernel("hessian_columns")
        release_compilation_cache()

    def __enter__(self):
        assert self.compiled is None
        start = time.perf_counter()
        payload = checked_path(self.record["executable"]).read_bytes()
        assert len(payload) == self.record["serialized_bytes"]
        assert hashlib.sha256(payload).hexdigest() == self.record["executable_sha256"]
        read_seconds = time.perf_counter()-start
        start = time.perf_counter()
        self.compiled = serialize_executable.deserialize_and_load(
            payload, self.in_tree, self.out_tree, backend="gpu")
        self.current = dict(read_and_verify_seconds=read_seconds,
            deserialize_seconds=time.perf_counter()-start, calls=0, execute_and_copy_seconds=0.)
        return self

    def __call__(self, *args):
        assert self.compiled is not None
        start = time.perf_counter()
        result = np.array(self.compiled(*args), copy=True)
        self.current["execute_and_copy_seconds"] += time.perf_counter()-start
        self.current["calls"] += 1
        return result

    def __exit__(self, exception_type, exception, tb):
        self.compiled = None
        start = time.perf_counter()
        release_compilation_cache()
        self.current.update(cleanup_seconds=time.perf_counter()-start,
            exception_type=None if exception_type is None else exception_type.__name__)
        self.loads.append(self.current)
        return False


def all_columns(kernel, point):
    size = len(point)
    result = np.zeros((size, size), dtype=np.float64)
    chunk = min(4, size)
    for first in range(0, size, chunk):
        width = min(chunk, size-first)
        directions = np.zeros((chunk, size), dtype=np.float64)
        directions[np.arange(width), first+np.arange(width)] = 1.
        result[:, first:first+width] = kernel(point, directions)[:width].T
    assert np.isfinite(result).all()
    return result
