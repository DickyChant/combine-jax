import os
from pathlib import Path
import subprocess

import pytest

# Set before importing ROOT/JAX and creating any thread pools.
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "true"
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])


@pytest.fixture(scope="session")
def root():
    from combine_jax.root_io import load_root
    return load_root()


@pytest.fixture(scope="session")
def counting_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("counting") / "workspace.root"
    card = Path(__file__).parent / "cards" / "counting.txt"
    subprocess.run(["text2workspace.py", str(card), "-o", str(path)], check=True, capture_output=True)
    return path
