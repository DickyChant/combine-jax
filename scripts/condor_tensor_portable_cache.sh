#!/usr/bin/env bash
set -euo pipefail
task_stage="${1:?producer or consumer required}"
[[ "$task_stage" == producer || "$task_stage" == consumer ]]
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name "gpu_tensor_portable_cache_${task_stage}_*" -exec cp -p {} "$task_scratch/results/" \;
    fi
    if [[ -d "$task_scratch/work/build/portable_consumer_cache" ]]; then
        tar czf "$task_scratch/portable_cache.tar.gz" -C "$task_scratch/work" build/portable_consumer_cache
    else
        tar czf "$task_scratch/portable_cache.tar.gz" --files-from /dev/null
    fi
    /usr/bin/python3 - "$task_status" "$task_stage" > "$task_scratch/results/worker_exit.json" <<'PY'
import json,os,platform,sys,time
print(json.dumps(dict(exit_code=int(sys.argv[1]),stage=sys.argv[2],host=platform.node(),
    finished_unix=time.time(),cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES')),indent=2))
PY
    exit "$task_status"
}
trap finish EXIT
mkdir work
tar xzf payload.tar.gz -C work
tar xf runtime_patch.tar -C work
if [[ "$task_stage" == consumer ]]; then
    tar xzf incoming_cache.tar.gz -C work
fi
cd work
bash scripts/run_tensor_lcg.sh scripts/run_tensor_portable_cache.py --stage "$task_stage"
