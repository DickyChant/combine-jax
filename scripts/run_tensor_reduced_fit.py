"""GPU gates and a full HHVV fit with reduced-space BFGS curvature."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    prefix = 'gpu_tensor_reduced_fit_'
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    manifest = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())
    report = dict(status='running', script_sha256=sha(__file__), started_unix=time.time(), commands=[],
                  scope='Primary A100/H100 GPU retry after generic reduced-space BFGS validation. Nominal starts, 10000 HHVV iterations, unchanged 1e-7 gradient target, raw exact-AD final covariance. Numerical acceptance is a separate audit; a command exit is not fit convergence.')

    def path(label): return 'reports/'+prefix+label+'.json'

    def run(command, label, *, required=True):
        log = Path('reports')/(prefix+label+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save_json(output, report)
        print('run', label, flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode, status='passed' if result.returncode == 0 else 'failed', finished_unix=time.time())
        save_json(output, report)
        if required and result.returncode:
            raise RuntimeError('Required prerequisite failed: '+label)

    def lcg(script, *args): return ['bash', 'scripts/run_tensor_lcg.sh', script, *map(str,args)]

    save_json(output, report)
    try:
        run(lcg('scripts/verify_tensor_gpu.py', '--output', path('environment')), 'environment')
        for name in ('counting', 'automcstats'):
            run(lcg('scripts/benchmark_tensor.py', '--model', manifest[name]['model'], '--backend', 'gpu',
                    '--operation', 'fit', '--fit-max-iterations', '2000', '--repeat', '1',
                    '--output', path(name)), name)
            run(lcg('scripts/audit_tensor_benchmarks.py', '--case', name, '--reports', path(name),
                    '--output', path(name+'_audit')), name+'_audit')
        model = manifest['hhvv']
        for operation in ('nll', 'value_gradient'):
            run(lcg('scripts/benchmark_tensor.py', '--model', model['model'], '--points', model['points'],
                    '--backend', 'gpu', '--operation', operation, '--repeat', '26',
                    '--output', path(operation)), operation)
        run(lcg('scripts/audit_tensor_benchmarks.py', '--case', 'hhvv', '--reports', path('nll'), path('value_gradient'),
                '--output', path('values_audit')), 'values_audit')
        run(lcg('scripts/benchmark_tensor.py', '--model', model['model'], '--backend', 'gpu', '--operation', 'fit',
                '--fit-max-iterations', '10000', '--batch-size', '4', '--compiler-memory-limit-gib', '28',
                '--repeat', '1', '--output', path('hhvv')), 'hhvv')
        run(['bash', 'scripts/run_tensor_native.sh', 'scripts/check_tensor_fit_candidate.py',
             '--fit', path('hhvv'), '--output', path('native_point')], 'native_point', required=False)
        report['status'] = 'completed_pending_audit' if all(c['returncode'] == 0 for c in report['commands']) else 'partial_failures'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished_unix'] = time.time(); save_json(output, report)


if __name__ == '__main__':
    main()
