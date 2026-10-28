"""Validate PyTorch eager/CUDA-graph GPU execution on portable benchmark models."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--concurrent-report',type=Path)
    parser.add_argument('--cases',nargs='+',
        choices=('counting','automcstats','parametric','multipdf_exp','multipdf_bern','hhvv'),
        default=['counting','automcstats','parametric','multipdf_exp','multipdf_bern'])
    args=parser.parse_args()
    assert args.tag.replace('_','').isalnum()
    prefix='gpu_tensor_'+args.tag+'_'
    output=Path('reports')/(prefix+'driver.json');assert not output.exists()
    manifest=json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    report=dict(status='running',script_sha256=sha(__file__),commands=[],started_unix=time.time(),
        scope='GPU numerical/capture smoke checks only. A separately running CPU benchmark may share the node; timings are not accepted as isolated performance measurements.')
    def other_progress():
        if args.concurrent_report is None:return None
        row=json.loads(args.concurrent_report.read_text())
        return dict(report=str(args.concurrent_report),status=row['status'],stage=row['stage'],
            completed_warm_calls=len(row['timings'].get('warm_seconds',[])),elapsed_seconds=row['elapsed_seconds'])
    def save():save_json(output,report)
    def run(command,label):
        log=Path('reports')/(prefix+label+'.log')
        row=dict(command=command,log=str(log),status='running',started_unix=time.time(),concurrent_before=other_progress())
        report['commands'].append(row);save();print('run',label,flush=True)
        with log.open('x') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,status='passed' if result.returncode==0 else 'failed',
                   finished_unix=time.time(),concurrent_after=other_progress());save()
        print(label,row['status'],flush=True)
        return result.returncode
    def lcg(script,*args):return ['bash','scripts/run_tensor_lcg.sh',script,*map(str,args)]
    save()
    try:
        assert run(lcg('scripts/verify_tensor_gpu.py','--output','reports/'+prefix+'environment.json'),'environment')==0
        for case in args.cases:
            reports=[]
            for mode in ('eager','cuda_graph'):
                for operation in ('nll','value_gradient'):
                    label=case+'_'+mode+'_'+operation;path='reports/'+prefix+label+'.json'
                    command=lcg('scripts/benchmark_tensor_torch.py','--model',manifest[case]['model'],
                        '--points',manifest[case]['points'],'--backend','gpu','--mode',mode,
                        '--operation',operation,'--repeat','5','--output',path)
                    if run(command,label)==0:reports.append(path)
            if reports:
                run(lcg('scripts/audit_tensor_benchmarks.py','--case',case,'--reports',*reports,
                    '--output','reports/'+prefix+case+'_audit.json'),case+'_audit')
        report['status']='passed' if all(r['returncode']==0 for r in report['commands']) else 'partial_failures'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        report['finished_unix']=time.time();save()


if __name__=='__main__':
    main()
