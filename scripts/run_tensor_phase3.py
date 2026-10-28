"""Cross-job cache reuse, corrected device fits, Torch modes and native fit timing."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='gpu_tensor_primary_phase3_')
    parser.add_argument('--without-cache-reuse', action='store_true',
                        help='Measure cold GPU NLL/gradient on another architecture without a transferred cache')
    args = parser.parse_args()
    prefix = args.prefix
    if not prefix.startswith('gpu_tensor_') or not prefix.endswith('_') or not prefix.replace('_', '').isalnum():
        parser.error('prefix must start with gpu_tensor_, end with _, and contain only letters, digits and underscores')
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    manifest = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    cache = Path('build/tensor_phase2_cache')
    if not args.without_cache_reuse:
        assert cache.is_dir()
    def cache_files():
        if args.without_cache_reuse:
            return {}
        return {str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*-cache'))}
    report = dict(status='running', script_sha256=sha(__file__), started_unix=time.time(),
        commands=[], cache_before=cache_files(), cache_snapshots=[],
        cache_reuse_requested=not args.without_cache_reuse,
        scope='One assigned full GPU: exact-bound regression, corrected HHVV fit attempt, JAX/PyTorch eager/CUDA-graph measurements and same-worker native executable fit. Cross-job cache reuse is tested only when requested. Independent numerical audits determine acceptance.')
    if not args.without_cache_reuse:
        assert len(report['cache_before']) == 2
    def save():
        save_json(output, report)
    def run(command, label, *, required=False, env=None):
        log = Path('reports')/(prefix+label+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save()
        print('run', command, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, errors='replace', env=env)
            for line in child.stdout:
                stream.write(line); stream.flush()
                if line.startswith(('heartbeat','combine heartbeat','verified','measured_','passed','failed')):
                    print(label, line.rstrip(), flush=True)
            code = child.wait()
        row.update(status='passed' if code == 0 else 'failed', returncode=code, finished_unix=time.time()); save()
        if required and code:
            raise RuntimeError(f'Required phase failed: {label}, exit {code}')
        return code
    def lcg(script, *args):
        return ['bash','scripts/run_tensor_lcg.sh',script,*map(str,args)]
    def path(label):
        return 'reports/'+prefix+label+'.json'
    def jax_benchmark(case, backend, operation, label, *, repeat=26, extra=()):
        model = manifest[case]
        points = model['curvature_points'] if operation == 'fit' else model['points']
        command = lcg('scripts/benchmark_tensor.py','--model',model['model'],'--points',points,
            '--backend',backend,'--operation',operation,'--repeat',repeat,'--output',path(label),*extra)
        return run(command, label, env={**os.environ,'JAX_LOG_COMPILES':'1'} if '--cache' in extra else None)
    def torch_benchmark(case, backend, mode, operation, label, *, repeat=26):
        model = manifest[case]
        return run(lcg('scripts/benchmark_tensor_torch.py','--model',model['model'],'--points',model['points'],
            '--backend',backend,'--mode',mode,'--operation',operation,'--repeat',repeat,'--output',path(label)),label)
    def audit(case, reports, label, *, required=False):
        assert reports
        return run(lcg('scripts/audit_tensor_benchmarks.py','--case',case,'--reports',*reports,
                       '--output',path(label)),label,required=required)
    save()
    try:
        run(lcg('scripts/verify_tensor_gpu.py','--output',path('environment')), 'environment', required=True)
        run(lcg('scripts/check_tensor_bound_mapping.py','--backend','gpu','--output',path('bound_regression')),
            'bound_regression', required=True)
        for case in ('counting','automcstats'):
            label='jax_fit_'+case
            assert jax_benchmark(case,'gpu','fit',label,repeat=1) == 0
            audit(case,[path(label)],label+'_audit',required=True)
        for operation in ('nll','value_gradient'):
            if args.without_cache_reuse:
                jax_benchmark('hhvv','gpu',operation,'jax_hhvv_gpu_'+operation)
            else:
                label='cross_cache_'+operation
                jax_benchmark('hhvv','gpu',operation,label,extra=('--cache',str(cache)))
                report['cache_snapshots'].append(dict(operation=operation, files=cache_files())); save()
        # The only changed fit policies are corrected physical coordinate
        # roundoff and a recorded larger iteration allowance; tolerance stays 1e-7.
        jax_benchmark('hhvv','gpu','fit','hhvv_fit',repeat=1,
            extra=('--fit-max-iterations','10000','--compiler-memory-limit-gib','28'))
        for case in ('counting','automcstats','parametric','multipdf_exp','multipdf_bern'):
            reports = []
            for backend,mode in (('cpu','eager'),('gpu','eager'),('gpu','cuda_graph')):
                for operation in ('nll','value_gradient'):
                    label=f'torch_{case}_{backend}_{mode}_{operation}'
                    if torch_benchmark(case,backend,mode,operation,label,repeat=5) == 0:
                        reports.append(path(label))
            if reports:
                audit(case,reports,'torch_'+case+'_audit')
        for operation in ('nll','value_gradient'):
            jax_benchmark('hhvv','cpu',operation,'jax_hhvv_cpu_'+operation)
        for backend,mode in (('cpu','eager'),('gpu','eager'),('gpu','cuda_graph')):
            reports = []
            for operation in ('nll','value_gradient'):
                label=f'torch_hhvv_{backend}_{mode}_{operation}'
                if torch_benchmark('hhvv',backend,mode,operation,label) == 0:
                    reports.append(path(label))
            if reports:
                audit('hhvv',reports,f'torch_hhvv_{backend}_{mode}_audit')
        run(['bash','scripts/run_tensor_native.sh','scripts/benchmark_native_tensor.py','--case','hhvv',
            '--repeat','26','--output',path('native_nll')],'native_nll')
        run(['bash','scripts/run_tensor_native.sh','scripts/benchmark_combine_fit.py','--case','hhvv',
            '--directory','build/phase3_native_fit','--wall-limit','10800','--output',path('native_fit')],'native_fit')
        report['status'] = 'completed_pending_audit' if all(c['returncode'] == 0 for c in report['commands']) else 'partial_failures'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report.update(finished_unix=time.time(), cache_after=cache_files()); save()


if __name__ == '__main__':
    main()
