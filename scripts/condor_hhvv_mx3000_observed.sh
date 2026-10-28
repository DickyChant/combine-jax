#!/bin/bash
set -euo pipefail
observed_scratch="$PWD"
mkdir results work
finish() {
  observed_status=$?
  /usr/bin/python3 - "$observed_scratch" "$observed_status" <<'PY'
import json, platform, shutil, sys, time
from pathlib import Path
base = Path(sys.argv[1])
for pattern in ('gpu_hhvv_MX3000_*.json', 'gpu_hhvv_MX3000_*.log'):
    for p in (base/'work/reports').glob(pattern):
        shutil.copyfile(p, base/'results'/p.name)
(base/'results/worker_exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),
    host=platform.node(), finished_unix=time.time()), indent=2)+'\n')
PY
  exit "$observed_status"
}
trap finish EXIT
tar xzf payload.tar.gz -C work
cd work
export JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
# Verify the actual assigned GPU and every frozen payload file before import/fit.
bash scripts/run_tensor_lcg.sh scripts/run_hhvv_mx3000_observed_gpu.py --phase verify
tar xzf build/hhvv_observed/import_environment.tar.gz
/usr/bin/python3 scripts/orchestrate_hhvv_mx3000_observed.py
