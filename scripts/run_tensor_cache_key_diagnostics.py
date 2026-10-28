"""Compare historical HHVV cache-key producers in an isolated GPU allocation."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    prefix = 'gpu_tensor_cache_key_'
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    report = dict(status='running', script_sha256=sha(__file__),
                  probe_sha256=sha('scripts/probe_tensor_cache_key.py'), commands=[],
                  started_unix=time.time(), source_roots={'producer':'d86f382', 'consumer':'b7b0f0b'},
                  scope='Dedicated A100 allocation. Historical producer/consumer sources use the identical neutral model; each fresh process verifies GPU first and captures unmodified cache-key components before compilation. No NLL evaluation or timing claim.')
    save_json(output, report)
    try:
        for operation in ('nll', 'value_gradient'):
            for root in ('producer', 'consumer'):
                label = root+'_'+operation
                result_path = Path('reports')/(prefix+label+'.json')
                log = result_path.with_suffix('.log')
                command = ['bash', 'scripts/run_tensor_lcg.sh', 'scripts/probe_tensor_cache_key.py',
                    '--backend', 'gpu', '--producer-root', root,
                    '--model', 'build/tensor_hhvv_fit_import/model', '--operation', operation,
                    '--cache', 'build/key_diagnostic_cache/'+label, '--output', str(result_path)]
                row = dict(command=command, log=str(log), started_unix=time.time(), status='running')
                report['commands'].append(row); save_json(output, report)
                with log.open('x') as stream:
                    result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                row.update(returncode=result.returncode, finished_unix=time.time(),
                           status='passed' if result.returncode == 0 else 'failed')
                save_json(output, report)
                assert result.returncode == 0, label
                capture = json.loads(result_path.read_text())
                assert capture['status'] == 'captured_before_compile' and capture['backend'] == 'gpu'
                assert capture['float64'] and len(capture['components']) == 8
                assert capture['script_sha256'] == report['probe_sha256']
        report['status'] = 'completed_pending_comparison'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished_unix'] = time.time(); save_json(output, report)


if __name__ == '__main__':
    main()
