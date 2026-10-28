#!/usr/bin/env bash
# Run only on the pinned node; all caches and outputs stay in its local /tmp.
set -e
cd /tmp/sqian/combine-jax
test "$(hostname -f)" = lxplus905.cern.ch
mkdir -p .cache/cuda tmp reports build/gpu_results
export CUDA_CACHE_PATH="$PWD/.cache/cuda"
export XDG_CACHE_HOME="$PWD/.cache"
export TMPDIR="$PWD/tmp"
export COMBINE_JAX_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh
source "$COMBINE_JAX_LCG_VIEW"
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
exec python scripts/gpu_benchmark.py "$@"
