#!/usr/bin/env bash
set -eo pipefail
export COMBINE_JAX_LCG_VIEW=/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh
source "$COMBINE_JAX_LCG_VIEW"
set -u
# LCG packages are exposed through PYTHONPATH; retain the view's entries.
export PYTHONPATH="$PWD/src:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
export XDG_CACHE_HOME="$PWD/build/tensor_lcg_cache"
export TMPDIR="$PWD/build/tensor_tmp"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false
if [[ ${COMBINE_JAX_GPU_SERIAL_COMPILE:-0} == 1 ]]; then
    export XLA_FLAGS="$XLA_FLAGS --xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_force_compilation_parallelism=1"
fi
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_X64=true
mkdir -p "$XDG_CACHE_HOME" "$TMPDIR"
exec python "$@"
