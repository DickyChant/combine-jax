#!/usr/bin/env bash
set -euo pipefail
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name 'gpu_tensor_primary_phase2_*' -exec cp -p {} "$task_scratch/results/" \;
    fi
    if [[ -d "$task_scratch/work/build/tensor_phase2_cache" ]]; then
        tar czf "$task_scratch/cache.tar.gz" -C "$task_scratch/work" build/tensor_phase2_cache
    else
        tar czf "$task_scratch/cache.tar.gz" --files-from /dev/null
    fi
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
tar xf runtime_patch.tar -C work
cd work
/usr/bin/python3 scripts/run_tensor_phase2.py
