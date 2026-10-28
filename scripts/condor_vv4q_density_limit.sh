#!/bin/bash
set -euo pipefail
audit_scratch="$PWD"
mkdir results work
finish() {
  audit_status=$?
  /usr/bin/python3 - "$audit_scratch" "$audit_status" <<'PY'
import json, os, platform, shutil, sys, time
from pathlib import Path
base=Path(sys.argv[1])
for p in (base/'work/reports').glob('gpu_vv4q_*'):
    if p.is_file(): shutil.copyfile(p, base/'results'/p.name)
(base/'results/worker_exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),
    host=platform.node(), finished_unix=time.time(),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')), indent=2)+'\n')
PY
  exit "$audit_status"
}
trap finish EXIT
tar xzf payload.tar.gz -C work
cd work
export JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
bash scripts/run_tensor_lcg.sh scripts/run_vv4q_density_gpu_limit.py
