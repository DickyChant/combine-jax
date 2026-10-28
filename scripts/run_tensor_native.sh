#!/usr/bin/env bash
# A separate process/runtime: never use the CUDA view's ROOT for native checks.
set -euo pipefail
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
export PATH=/usr/bin:/bin
export PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src:$PWD/scripts"
export TMPDIR="$PWD/build/native_tensor_tmp" XDG_CACHE_HOME="$PWD/build/native_tensor_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# Invoke directly so this venv can also be relocated into a Condor scratch dir.
exec "$PWD/.venv-combine-v11/bin/python" "$@"
