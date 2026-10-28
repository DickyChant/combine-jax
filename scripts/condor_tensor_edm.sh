#!/usr/bin/env bash
set -euo pipefail
task_scratch="$PWD"
mkdir results
finish() {
    task_status=$?
    /usr/bin/python3 - "$task_scratch" "$task_status" <<'PY'
import hashlib,json,os,platform,shutil,sys,time
from pathlib import Path
base=Path(sys.argv[1]);work=base/'work';output=base/'results';attachments={}
for p in (work/'reports').glob('gpu_tensor_edm_*'):
    if p.is_file():shutil.copyfile(p,output/p.name)
for case in ('counting','automcstats','hhvv'):
    directory=work/'build'/('edm_native_'+case)
    for p in directory.glob('*'):
        if p.is_file() and (p.name in ('combine.log','combine_logger.out','counter_build.log') or p.suffix=='.jsonl'
                or (p.name.startswith('higgsCombine') and p.suffix=='.root')):
            target=output/('gpu_tensor_edm_native_'+case+'_'+p.name)
            shutil.copyfile(p,target)
            attachments[str(p.relative_to(work))]=dict(file=target.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(output/'worker_exit.json').write_text(json.dumps(dict(exit_code=int(sys.argv[2]),host=platform.node(),finished_unix=time.time(),
    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),attachments=attachments),indent=2)+'\n')
PY
    exit "$task_status"
}
trap finish EXIT
mkdir work
tar xzf payload.tar.gz -C work
tar xf runtime_patch.tar -C work
cd work
bash scripts/run_tensor_lcg.sh scripts/run_tensor_edm_campaign.py
