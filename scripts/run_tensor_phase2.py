"""Primary GPU curvature/fit attempt and fresh-process executable-cache measurements."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    output = Path('reports/gpu_tensor_primary_phase2_driver.json')
    assert not output.exists()
    manifest = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    model = manifest['hhvv']
    cache = Path('build/tensor_phase2_cache')
    assert not cache.exists()
    report = dict(status='running', started_unix=time.time(), script_sha256=sha(__file__), commands=[],
        scope='A100/H100 full HHVV GPU Hessian and fit attempts, plus cold/fresh-process-warm persistent NLL/gradient caching. Numerical acceptance is a separate audit.',
        cache_snapshots=[])
    def save():
        temp = output.with_suffix('.json.tmp')
        temp.write_text(json.dumps(report, indent=2)+'\n'); temp.replace(output)
    def run(command, stem, *, required=True, env=None):
        log = Path('reports')/(stem+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save()
        print('run', command, flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
        row.update(status='passed' if result.returncode == 0 else 'failed', returncode=result.returncode,
                   finished_unix=time.time()); save()
        if required:
            result.check_returncode()
        return result.returncode
    save()
    try:
        stem = 'gpu_tensor_primary_phase2_environment'
        run(['bash','scripts/run_tensor_lcg.sh','scripts/verify_tensor_gpu.py','--output',f'reports/{stem}.json'], stem)
        for operation in ('nll','value_gradient'):
            for state in ('cold','warm'):
                stem = f'gpu_tensor_primary_phase2_cache_{operation}_{state}'
                run(['bash','scripts/run_tensor_lcg.sh','scripts/benchmark_tensor.py','--model',model['model'],
                    '--points',model['points'],'--backend','gpu','--operation',operation,'--repeat','5',
                    '--cache',str(cache),'--output',f'reports/{stem}.json'], stem,
                    env={**os.environ,'JAX_LOG_COMPILES':'1'})
                report['cache_snapshots'].append(dict(operation=operation, state=state,
                    files={str(p.relative_to(cache)):sha(p) for p in sorted(cache.rglob('*-cache'))}))
                save()
        # Preserve independent cache results even if a full Hessian or fit exceeds
        # its static memory budget. A nonconvergent fit is recorded as such.
        for operation in ('hessian','fit'):
            stem = 'gpu_tensor_primary_phase2_'+operation
            run(['bash','scripts/run_tensor_lcg.sh','scripts/benchmark_tensor.py','--model',model['model'],
                '--points',model['curvature_points'],'--backend','gpu','--operation',operation,
                '--repeat','3' if operation == 'hessian' else '1','--compiler-memory-limit-gib','28',
                '--output',f'reports/{stem}.json'], stem, required=False)
        report['status'] = 'completed_pending_audit' if all(r['returncode'] == 0 for r in report['commands']) else 'partial_failures'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished_unix'] = time.time(); save()


if __name__ == '__main__':
    main()
