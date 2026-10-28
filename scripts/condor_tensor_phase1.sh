#!/usr/bin/env bash
# Submitted with -spool from lxplus905; all payloads and results stay off AFS.
set -euo pipefail
task_scratch="$PWD"
task_tag="${1:-primary}"
[[ "$task_tag" =~ ^primary[_a-z0-9]*$ ]]
mkdir results
finish() {
    task_status=$?
    if [[ -d "$task_scratch/work/reports" ]]; then
        find "$task_scratch/work/reports" -maxdepth 1 -type f -name 'gpu_tensor_primary_*' -exec cp -p {} "$task_scratch/results/" \;
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
if [[ -f runtime_patch.tar ]]; then
    tar xf runtime_patch.tar -C work
fi
cd work
# The driver verifies jax.devices() before its first numerical operation.
/usr/bin/python3 scripts/run_tensor_node_matrix.py --tag "${task_tag}_smoke" \
    --cases counting automcstats --backends cpu gpu --modes jit \
    --operations nll value_gradient hessian fit --repeat 5 --native
/usr/bin/python3 scripts/run_tensor_node_matrix.py --tag "${task_tag}_hhvv" \
    --cases hhvv --backends cpu gpu --modes jit \
    --operations nll value_gradient --repeat 26 --native
cd "$task_scratch"
