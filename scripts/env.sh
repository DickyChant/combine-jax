# Source after activating .venv. Keep numerical jobs within the project CPU limit.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=true
# Bare tokens stop XLA's option parser and silently hide later appended flags.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
# XLA creates additional thread pools according to CPU affinity.
taskset -pc "$(python -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:8])))')" $$ >/dev/null
export TMPDIR="$PWD/build/tmp"
mkdir -p "$TMPDIR"
