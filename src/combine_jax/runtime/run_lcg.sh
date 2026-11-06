#!/usr/bin/env bash
set -eo pipefail
runtime_prefix="$1"; shift
runtime_view="$1"; shift
runtime_omp="${OMP_NUM_THREADS:-1}"
runtime_blas="${OPENBLAS_NUM_THREADS:-1}"
runtime_mkl="${MKL_NUM_THREADS:-1}"
runtime_xla="${XLA_FLAGS---xla_cpu_multi_thread_eigen=false --xla_gpu_enable_llvm_module_compilation_parallelism=false --xla_gpu_force_compilation_parallelism=1}"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH VIRTUAL_ENV
export PATH=/usr/bin:/bin PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1
source "$runtime_view"
export COMBINE_JAX_LCG_VIEW="$runtime_view"
export PYTHONPATH="$runtime_prefix/code/src${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="$runtime_omp" OPENBLAS_NUM_THREADS="$runtime_blas" MKL_NUM_THREADS="$runtime_mkl"
export JAX_ENABLE_X64=true XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_FLAGS="$runtime_xla"
exec python "$@"
