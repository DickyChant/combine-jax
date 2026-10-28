"""Sequential complete-HHVV CPU/T4 mode timings and memory-limited Hessians."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    prefix = 'gpu_tensor_t4_complete_'
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    manifest = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    model = manifest['hhvv']
    report = dict(status='running', script_sha256=sha(__file__), started_unix=time.time(), commands=[],
        scope='Sequential project workloads on shared lxplus905: JAX JIT and Torch eager/CUDA graphs use 26 changing warm points, then native NLL timing and CPU/T4 Hessians. No concurrent project benchmark on this node. Independent native/matrix audits determine acceptance.',
        hessian_policy='One AD column per block on both CPU and T4 to reduce device memory; exact full 2656-by-2656 nominal Hessian, three warm calls, 8 GiB static allocation guard. Different block size from prior four-column A100/CPU measurements is explicit.')

    def path(label):
        return 'reports/'+prefix+label+'.json'

    def save():
        save_json(output, report)

    def run(command, label, *, required=False):
        log = Path('reports')/(prefix+label+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save()
        print('run', label, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors='replace')
            for line in child.stdout:
                stream.write(line); stream.flush()
                if line.startswith(('heartbeat','verified','measured_','passed','failed')):
                    print(label, line.rstrip(), flush=True)
            code = child.wait()
        row.update(returncode=code, status='passed' if code == 0 else 'failed', finished_unix=time.time()); save()
        if code and required:
            raise RuntimeError(f'Required prerequisite failed: {label}, exit {code}')
        return code

    def lcg(script, *args):
        return ['bash','scripts/run_tensor_lcg.sh',script,*map(str,args)]

    save()
    try:
        run(lcg('scripts/verify_tensor_gpu.py','--output',path('environment')), 'environment', required=True)
        for backend in ('cpu','gpu'):
            for operation in ('nll','value_gradient'):
                label=f'jax_{backend}_{operation}'
                run(lcg('scripts/benchmark_tensor.py','--model',model['model'],'--points',model['points'],
                    '--backend',backend,'--operation',operation,'--repeat','26','--output',path(label)),label)
        for backend, mode in (('cpu','eager'),('gpu','eager'),('gpu','cuda_graph')):
            reports=[]
            for operation in ('nll','value_gradient'):
                label=f'torch_{backend}_{mode}_{operation}'
                if run(lcg('scripts/benchmark_tensor_torch.py','--model',model['model'],'--points',model['points'],
                    '--backend',backend,'--mode',mode,'--operation',operation,'--repeat','26','--output',path(label)),label)==0:
                    reports.append(path(label))
            if reports:
                label=f'torch_{backend}_{mode}_audit'
                run(lcg('scripts/audit_tensor_benchmarks.py','--case','hhvv','--reports',*reports,'--output',path(label)),label)
        run(['bash','scripts/run_tensor_native.sh','scripts/benchmark_native_tensor.py','--case','hhvv',
            '--repeat','26','--output',path('native_nll')],'native_nll')
        for backend in ('gpu','cpu'):
            label=f'jax_{backend}_hessian'
            run(lcg('scripts/benchmark_tensor.py','--model',model['model'],'--points',model['curvature_points'],
                '--backend',backend,'--operation','hessian','--batch-size','1','--repeat','3',
                '--compiler-memory-limit-gib','8','--output',path(label)),label)
        report['status']='completed_pending_audit' if all(c['returncode']==0 for c in report['commands']) else 'partial_failures'
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        report['finished_unix']=time.time();save()


if __name__=='__main__':
    main()
