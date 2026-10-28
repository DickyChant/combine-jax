"""Measure one full nominal-Asimov Hessian on a primary GPU, without a fit."""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from tensor_benchmark_common import save_json,sha


def memory_guard(gpu):
    total=gpu['device_total_mib'];used=gpu['device_baseline_mib']
    if gpu.get('error') is not None or not all(math.isfinite(x) for x in (total,used)) or not 0<=used<total:
        raise ValueError('A valid GPU memory measurement is required')
    return min(68.,.70*total/1024,.75*(total-used)/1024)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wall-limit',type=float,default=3600.)
    args=parser.parse_args()
    if not 0<args.wall_limit<=3600:parser.error('Wall limit must be positive and at most one hour')
    prefix='reports/gpu_tensor_edm_hessian_pilot_'
    output=Path(prefix+'driver.json');assert not output.exists()
    runtime_path=Path('build/edm_runtime_manifest.json');runtime=json.loads(runtime_path.read_text())
    assert all(sha(p)==h for p,h in runtime['sources'].items())
    report=dict(status='running',script_sha256=sha(__file__),source_revision=runtime['source_revision'],
        runtime_manifest_sha256=sha(runtime_path),started_unix=time.time(),commands=[],
        scope='Full 2656-coordinate nominal pre-fit Asimov Hessian pilot, no fit. Numerical Hessian agreement requires a separate audit. Compile and execution are separate measurements.')
    child=None
    def run(script,*arguments):
        nonlocal child
        command=['bash','scripts/run_tensor_lcg.sh',script,*map(str,arguments)]
        row=dict(command=command,started_unix=time.time(),status='running');report['commands'].append(row)
        save_json(output,report)
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,start_new_session=True)
        row['pid']=child.pid;save_json(output,report)
        row['returncode']=child.wait(timeout=args.wall_limit)
        row.update(status='completed' if child.returncode==0 else 'failed',finished_unix=time.time())
        save_json(output,report)
        if child.returncode:raise RuntimeError('Pilot subprocess failed; retain its benchmark report')
    def interrupted(signum,frame):raise InterruptedError(f'Pilot received signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        run('scripts/verify_tensor_gpu.py','--output',prefix+'environment.json')
        environment=json.loads(Path(prefix+'environment.json').read_text())
        assert len(environment['devices'])==1 and 'H100' in environment['devices'][0]['kind']
        manifest=json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text());case=manifest['hhvv']
        metadata=json.loads((Path(case['model'])/'model.json').read_text())['metadata']
        from tensor_benchmark_dataset import benchmark_dataset
        assert benchmark_dataset(case,SimpleNamespace(metadata=metadata),label='hhvv')=='asimov_prefit'
        assert len(metadata['parameter_names'])==2656
        gpu=environment['resources']['gpu'];limit=memory_guard(gpu)
        report.update(dataset='asimov_prefit',dataset_identity=metadata['provenance']['dataset']['identity'],
            environment_sha256=sha(prefix+'environment.json'),compiler_memory_limit_gib=limit,
            guard_policy='min(68 GiB, 70% total VRAM, 75% initially free VRAM); benchmark independently enforces 80% of its measured free memory.',
            batch_size=1,nominal_start_only=True)
        save_json(output,report)
        run('scripts/benchmark_tensor.py','--model',case['model'],'--backend','gpu','--operation','hessian',
            '--batch-size','1','--repeat','1','--compiler-memory-limit-gib',limit,
            '--cache','build/hessian_pilot_cache','--output',prefix+'measurement.json')
        assert all(sha(p)==h for p,h in runtime['sources'].items())
        report['status']='measured_pending_hessian_agreement'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            report['interrupted_child_returncode']=child.returncode
        report['finished_unix']=time.time();save_json(output,report)


if __name__=='__main__':main()
