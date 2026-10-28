#!/usr/bin/env bash
set -euo pipefail
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name 'gpu_tensor_reduced_fit_*' -exec cp -p {} "$task_scratch/results/" \;
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
bash scripts/run_tensor_lcg.sh scripts/run_tensor_reduced_fit.py
