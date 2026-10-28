"""Produce or consume a location-portable full-HHVV GPU executable cache."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('producer', 'consumer'), required=True)
    args = parser.parse_args()
    prefix = 'gpu_tensor_portable_cache_'+args.stage+'_'
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    incoming = Path('build/portable_consumer_cache')

    def snapshot(directory):
        return {str(p.relative_to(directory)): sha(p) for p in sorted(directory.rglob('*-cache'))}

    report = dict(status='running', stage=args.stage, script_sha256=sha(__file__),
        started_unix=time.time(), commands=[], snapshots={},
        scope='Full-HHVV GPU cache relocation/cross-job experiment. Verify GPU first; '
              '26 resident points per capture. Disable only optional auxiliary XLA caches '
              'through the tested model-cache policy; JAX compatibility keys are unchanged. '
              'Numerical and cache-hit acceptance require independent audits.')
    save_json(output, report)

    def run(command, label):
        log = Path('reports')/(prefix+label+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save_json(output, report)
        with log.open('x') as stream:
            result = subprocess.run(['bash', 'scripts/run_tensor_lcg.sh', *command],
                stdout=stream, stderr=subprocess.STDOUT, env={**os.environ, 'JAX_LOG_COMPILES':'1'})
        row.update(returncode=result.returncode, finished_unix=time.time(),
                   status='passed' if result.returncode == 0 else 'failed')
        save_json(output, report)
        assert result.returncode == 0, label

    try:
        run(['scripts/verify_tensor_gpu.py', '--output', 'reports/'+prefix+'environment.json'], 'environment')
        if args.stage == 'producer':
            cold = Path('build/portable_producer_cache')
            assert not cold.exists() and not incoming.exists()
            rounds = [('cold', cold), ('relocated', incoming)]
        else:
            assert incoming.is_dir() and snapshot(incoming)
            rounds = [('cross_job', incoming)]
        for label, cache in rounds:
            if label == 'relocated':
                shutil.copytree(cold, incoming)
            report['snapshots'][label+'_before'] = snapshot(cache)
            save_json(output, report)
            captures = []
            for operation in ('nll', 'value_gradient'):
                name = label+'_'+operation
                path = 'reports/'+prefix+name+'.json'
                captures.append(path)
                run(['scripts/benchmark_tensor.py', '--model', 'build/tensor_hhvv_fit_import/model',
                    '--points', 'build/tensor_benchmark_inputs/hhvv/points.npz', '--backend', 'gpu',
                    '--operation', operation, '--repeat', '26', '--cache', str(cache), '--output', path], name)
                capture = json.loads(Path(path).read_text())
                assert capture['cache']['auxiliary_xla_caches'] == 'none'
                report['snapshots'][name+'_after'] = snapshot(cache)
                save_json(output, report)
            run(['scripts/audit_tensor_benchmarks.py', '--case', 'hhvv', '--reports', *captures,
                '--output', 'reports/'+prefix+label+'_values_audit.json'], label+'_values_audit')
        report['status'] = 'completed_pending_audit'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished_unix'] = time.time(); save_json(output, report)


if __name__ == '__main__':
    main()
