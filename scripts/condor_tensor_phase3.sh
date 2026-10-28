#!/usr/bin/env bash
set -euo pipefail
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name 'gpu_tensor_primary_phase3_*' -exec cp -p {} "$task_scratch/results/" \;
    fi
    if [[ -d "$task_scratch/work/build/phase3_native_fit" ]]; then
        for task_file in "$task_scratch/work/build/phase3_native_fit"/multidimfit*.root; do
            [[ ! -f "$task_file" ]] || cp -p "$task_file" "$task_scratch/results/gpu_tensor_primary_phase3_native_fit.root"
        done
        [[ ! -f "$task_scratch/work/build/phase3_native_fit/combine.log" ]] || cp -p "$task_scratch/work/build/phase3_native_fit/combine.log" "$task_scratch/results/gpu_tensor_primary_phase3_native_fit_combine.log"
    fi
    if [[ -d "$task_scratch/work/build/tensor_phase2_cache" ]]; then
        tar czf "$task_scratch/cache_after.tar.gz" -C "$task_scratch/work" build/tensor_phase2_cache
    else
        tar czf "$task_scratch/cache_after.tar.gz" --files-from /dev/null
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
tar xzf cache_input.tar.gz -C work
cd work
# Orchestration uses the view's Python for the shared hash helpers. Numerical
# children use either that CUDA view or the separate native v11/ROOT6.40 venv.
bash scripts/run_tensor_lcg.sh scripts/run_tensor_phase3.py
