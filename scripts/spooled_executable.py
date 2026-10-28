"""Keep a process-created JAX executable on disk between benchmark calls.

Only load artifacts produced by this process through ``write``. JAX executable
deserialization is not an interface for untrusted files. PyTree definitions stay
in memory; no cross-process or cross-version cache compatibility is assumed.
"""

import hashlib
from pathlib import Path
import time

import jax
from jax.experimental import serialize_executable
import numpy as np

from combine_jax.local import release_compilation_cache


class SpooledExecutable:
    def __init__(self, path, digest, size, in_tree, out_tree, backend):
        self.path = Path(path)
        self.sha256 = digest
        self.size = size
        self.in_tree = in_tree
        self.out_tree = out_tree
        self.backend = backend
        self.calls = []

    @classmethod
    def write(cls, compiled, path, *, backend):
        """Serialize an actual executable; fail explicitly if JAX cannot do so."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        payload, in_tree, out_tree = serialize_executable.serialize(compiled)
        serialization_seconds = time.perf_counter()-start
        start = time.perf_counter()
        # Do not overwrite previous executable evidence or accept a symlink.
        with path.open("xb") as stream:
            stream.write(payload)
        write_seconds = time.perf_counter()-start
        instance = cls(path, hashlib.sha256(payload).hexdigest(), len(payload),
                       in_tree, out_tree, backend)
        return instance, dict(serialization_seconds=serialization_seconds,
                              write_seconds=write_seconds, serialized_bytes=len(payload),
                              executable_sha256=instance.sha256)

    def __call__(self, *args, **kwargs):
        start = time.perf_counter()
        payload = self.path.read_bytes()
        if len(payload) != self.size or hashlib.sha256(payload).hexdigest() != self.sha256:
            raise ValueError(f"Serialized executable integrity check failed: {self.path}")
        read_seconds = time.perf_counter()-start
        start = time.perf_counter()
        compiled = serialize_executable.deserialize_and_load(
            payload, self.in_tree, self.out_tree, backend=self.backend)
        del payload
        load_seconds = time.perf_counter()-start
        try:
            start = time.perf_counter()
            result = compiled(*args, **kwargs)
            # Own synchronized host copies so no returned device array keeps
            # executable/device state alive after this call.
            host = jax.tree_util.tree_map(lambda x: np.array(x, copy=True), result)
            execution_seconds = time.perf_counter()-start
            del result
        finally:
            del compiled
            start = time.perf_counter()
            release_compilation_cache()
            cleanup_seconds = time.perf_counter()-start
        self.calls.append(dict(read_and_verify_seconds=read_seconds,
                               deserialize_seconds=load_seconds,
                               execute_and_copy_seconds=execution_seconds,
                               cleanup_seconds=cleanup_seconds))
        return host
