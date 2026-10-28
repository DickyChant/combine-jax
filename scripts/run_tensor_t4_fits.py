"""Sequential same-node HHVV CPU, T4 and native Combine fit measurements."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import re

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', default='gpu_tensor_t4_fit_comparison_')
    args = parser.parse_args()
    assert re.fullmatch(r'gpu_tensor_[a-z0-9_]+_', args.prefix)
    prefix = args.prefix
    output = Path('reports')/(prefix+'driver.json')
    assert not output.exists()
    runtime_path = Path('build/t4_fit_runtime_manifest.json')
    runtime = json.loads(runtime_path.read_text())
    assert all(sha(p) == digest for p, digest in runtime['sources'].items())
    manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    report = dict(status='running', script_sha256=sha(__file__), started_unix=time.time(),
                  controller_pid=os.getpid(),
                  controller_start_ticks=int(Path('/proc/self/stat').read_text().rsplit(')', 1)[1].split()[19]),
                  runtime_manifest=str(runtime_path), runtime_manifest_sha256=sha(runtime_path),
                  source_revision=runtime['source_revision'], sources=runtime['sources'],
                  input_manifest_sha256=sha(manifest_path), commands=[],
                  scope='Sequential project workloads on pinned lxplus905, at most eight CPUs. '
                        'Compare the unchanged nominal-start solver on CPU and T4, then actual Combine v11. '
                        'Other users are uncontrolled; failed fits are failed-attempt timings only. '
                        'T4 FP64 throughput is about 1/32 FP32 per user hardware context, not a measured workload ratio.',
                  fit_policy='10000 tensor iterations, unchanged 1e-7 gradient target and raw final AD covariance. '
                             'One Hessian column per block on both CPU and T4, versus four in primary GPU fits. '
                             'Save first result; skip warm repetition if convergence or covariance is invalid. '
                             'Native executable has a six-hour guard; native startup, MIGRAD and HESSE are timed.')
    child = None

    def path(label):
        return str(Path('reports')/(prefix+label+'.json'))

    def run(command, label, *, required=True):
        nonlocal child
        log = Path('reports')/(prefix+label+'.log')
        row = dict(command=command, log=str(log), status='running', started_unix=time.time())
        report['commands'].append(row); save_json(output, report)
        print('start', label, flush=True)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            fields = Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()
            row.update(pid=child.pid, start_ticks=int(fields[19])); save_json(output, report)
            child.wait()
        row.update(returncode=child.returncode, finished_unix=time.time(), log_sha256=sha(log),
                   status='completed' if child.returncode == 0 else 'failed')
        save_json(output, report)
        print(row['status'], label, child.returncode, flush=True)
        if required:
            assert child.returncode == 0, row
        return child.returncode

    def lcg(script, *args):
        return ['bash', 'scripts/run_tensor_lcg.sh', script, *map(str, args)]

    def native(script, *args):
        return ['bash', 'scripts/run_tensor_native.sh', script, *map(str, args)]

    def interrupted(signum, frame):
        raise InterruptedError(f'Controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save_json(output, report)
    try:
        # First numerical process: require the requested GPU and float64.
        run(lcg('scripts/verify_tensor_gpu.py', '--output', path('environment')), 'environment')
        for case in ('counting', 'automcstats'):
            captures = []
            for backend in ('cpu', 'gpu'):
                label = case+'_'+backend+'_fit'
                captures.append(path(label))
                run(lcg('scripts/benchmark_tensor.py', '--model', manifest[case]['model'],
                        '--backend', backend, '--operation', 'fit', '--fit-max-iterations', '2000',
                        '--batch-size', '1', '--repeat', '1', '--output', path(label)), label)
            label = case+'_audit'
            run(lcg('scripts/audit_tensor_benchmarks.py', '--case', case, '--reports', *captures,
                    '--output', path(label)), label)
        model = manifest['hhvv']
        for backend in ('cpu', 'gpu'):
            captures = []
            for operation in ('nll', 'value_gradient'):
                label = backend+'_'+operation
                captures.append(path(label))
                run(lcg('scripts/benchmark_tensor.py', '--model', model['model'], '--points', model['points'],
                        '--backend', backend, '--operation', operation, '--repeat', '26',
                        '--output', path(label)), label)
            label = backend+'_values_audit'
            run(lcg('scripts/audit_tensor_benchmarks.py', '--case', 'hhvv', '--reports', *captures,
                    '--output', path(label)), label)
        run(native('scripts/benchmark_native_tensor.py', '--case', 'hhvv', '--repeat', '26',
                   '--output', path('native_nll')), 'native_nll')
        for backend in ('gpu', 'cpu'):
            label = backend+'_hhvv_fit'
            code = run(lcg('scripts/benchmark_tensor.py', '--model', model['model'], '--backend', backend,
                    '--operation', 'fit', '--fit-max-iterations', '10000', '--batch-size', '1',
                    '--compiler-memory-limit-gib', '8', '--repeat', '1', '--output', path(label)), label,
                    required=False)
            if code == 0:
                run(native('scripts/check_tensor_fit_candidate.py', '--fit', path(label),
                           '--output', path(label+'_native_point')), label+'_native_point', required=False)
        run(native('scripts/benchmark_combine_fit.py', '--case', 'hhvv',
                   '--directory', 'build/t4_fit_comparison_native', '--wall-limit', '21600',
                   '--output', path('native_fit')), 'native_fit', required=False)
        assert all(sha(p) == digest for p, digest in runtime['sources'].items())
        report['status'] = ('completed_pending_audit' if all(r['returncode'] == 0 for r in report['commands'])
                            else 'partial_failures_pending_audit')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                # Allow the native wrapper to reap its own separately grouped
                # Combine executable (its termination grace period is five seconds).
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            report['interrupted_child_returncode'] = child.returncode
        report['finished_unix'] = time.time(); save_json(output, report)


if __name__ == '__main__':
    main()
