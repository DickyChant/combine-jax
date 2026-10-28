#!/bin/bash
set -euo pipefail
campaign_scratch="$PWD"
mkdir results work
finish() {
  campaign_status=$?
  /usr/bin/python3 - "$campaign_scratch" "$campaign_status" <<'PY'
import json,os,platform,shutil,sys,time
from pathlib import Path
base=Path(sys.argv[1])
for p in (base/'work/reports').glob('gpu_hhvv_v10_a100_*'):
    if p.is_file():shutil.copyfile(p,base/'results'/p.name)
native=base/'work/build/hhvv_v10/native_fit'
for p in native.glob('*'):
    if p.is_file() and (p.suffix=='.jsonl' or p.name in ('combine.log','combine_logger.out','counter_build.log')):
        shutil.copyfile(p,base/'results'/('native_'+p.name))
(base/'results/worker_exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),
    host=platform.node(),finished_unix=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2)+'\n')
PY
  exit "$campaign_status"
}
trap finish EXIT
tar xzf payload.tar.gz -C work
cd work
export JAX_PLATFORMS=cuda PYTHONDONTWRITEBYTECODE=1
bash scripts/run_tensor_lcg.sh scripts/run_hhvv_v10_a100_campaign.py
