#!/usr/bin/env bash
set -euo pipefail
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name 'gpu_tensor_cache_key_*' -exec cp -p {} "$task_scratch/results/" \;
    fi
    nvidia-smi --query-gpu=name,driver_version,memory.total,uuid --format=csv,noheader > "$task_scratch/results/gpu_tensor_cache_key_device_metadata.log" || true
    /usr/bin/python3 - "$task_status" > "$task_scratch/results/worker_exit.json" <<'PY'
import json,os,platform,sys,time
print(json.dumps(dict(exit_code=int(sys.argv[1]),host=platform.node(),finished_unix=time.time(),
                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2))
PY
    exit "$task_status"
}
trap finish EXIT
mkdir work
tar xzf payload.tar.gz -C work
tar xf diagnostic_patch.tar -C work
cd work
bash scripts/run_tensor_lcg.sh scripts/run_tensor_cache_key_diagnostics.py
