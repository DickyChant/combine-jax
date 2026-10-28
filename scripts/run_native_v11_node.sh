#!/usr/bin/env bash
# Separate native reference runtime; do not source the CUDA/ROOT LCG view.
set -euo pipefail
cd /tmp/sqian/combine-jax
test "$(hostname -f)" = lxplus905.cern.ch
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH
export PATH=/usr/bin:/bin
source .venv-combine-v11/bin/activate
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD/src:$PWD/scripts"
export TMPDIR="$PWD/tmp/native_v11"
export XDG_CACHE_HOME="$PWD/build/native_v11_cache"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=true
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
if [[ "${1:-}" != scripts/check_native_v11_environment.py ]]; then
    python - <<'PY'
import json
from pathlib import Path
for policy in ("spooled", "scoped"):
    for phase in ("smoke", "hhvv"):
        path = Path(f"reports/gpu_{policy}_{phase}_driver.json")
        if path.exists() and json.loads(path.read_text())["status"] == "running":
            raise RuntimeError(f"Finish the active benchmark before native reference work: {path}")
PY
fi
exec python "$@"
