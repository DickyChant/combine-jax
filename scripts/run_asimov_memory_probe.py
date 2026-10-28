"""Isolated Asimov value/gradient memory comparisons; never runs a fit."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import benchmark_dataset


def benchmark_arguments(case, backend, policy, output, limit):
    if backend not in ('cpu', 'gpu') or policy not in ('ad', 'structural', 'structural_sequential'):
        raise ValueError('Unknown memory comparison variant')
    return ['scripts/benchmark_tensor.py', '--model', case['model'], '--points', case['points'],
        '--backend', backend, '--operation', 'value_gradient', '--binned-gradient', policy,
        '--row-chunk-size', '1024', '--repeat', '3', '--compiler-memory-limit-gib', str(limit),
        '--output', str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('A100', 'H100', 'T4'), required=True)
    parser.add_argument('--policies', nargs='+', choices=('ad', 'structural', 'structural_sequential'),
                        default=['ad', 'structural', 'structural_sequential'])
    args = parser.parse_args()
    if len(set(args.policies)) != len(args.policies):
        parser.error('Each gradient policy must be selected once')
    prefix = 'reports/gpu_tensor_edm_memory_probe_'
    output = Path(prefix + 'driver.json')
    assert not output.exists()
    runtime_path = Path('build/edm_runtime_manifest.json')
    runtime = json.loads(runtime_path.read_text())
    assert all(sha(p) == h for p, h in runtime['sources'].items())
    report = dict(status='running', script_sha256=sha(__file__), source_revision=runtime['source_revision'],
        runtime_manifest_sha256=sha(runtime_path), started_unix=time.time(), commands=[],
        policies=args.policies,
        scope='Value+gradient memory comparison only, no fitting. Selected gradient policies use identical nominal pre-fit Asimov plus 25 random points. Results require independent native audits.')
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f'Memory probe received signal {signum}')

    def run(arguments):
        nonlocal child
        command = ['bash', 'scripts/run_tensor_lcg.sh', *arguments]
        row = dict(command=command, started_unix=time.time(), status='running')
        report['commands'].append(row)
        save_json(output, report)
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, start_new_session=True)
        row['pid'] = child.pid
        save_json(output, report)
        row['returncode'] = child.wait(timeout=3600)
        row.update(status='completed' if child.returncode == 0 else 'failed', finished_unix=time.time())
        save_json(output, report)
        return child.returncode

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        assert run(['scripts/verify_tensor_gpu.py', '--output', prefix + 'environment.json']) == 0
        environment = json.loads(Path(prefix + 'environment.json').read_text())
        assert len(environment['devices']) == 1 and args.family in environment['devices'][0]['kind']
        manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
        assert sha(manifest_path) == runtime['manifest_sha256']
        case = json.loads(manifest_path.read_text())['hhvv']
        metadata = json.loads((Path(case['model']) / 'model.json').read_text())['metadata']
        assert benchmark_dataset(case, SimpleNamespace(metadata=metadata), label='hhvv') == 'asimov_prefit'
        assert all(sha(p) == h for p, h in runtime['input_files'].items())
        gpu = environment['resources']['gpu']
        assert gpu.get('error') is None and 0 <= gpu['device_baseline_mib'] < gpu['device_total_mib']
        limit = min(28., .70 * gpu['device_total_mib'] / 1024,
                    .75 * (gpu['device_total_mib'] - gpu['device_baseline_mib']) / 1024)
        report.update(dataset='asimov_prefit', dataset_identity=metadata['provenance']['dataset']['identity'],
            environment_sha256=sha(prefix + 'environment.json'), gpu_static_guard_gib=limit,
            cpu_static_guard_gib=8., family=args.family,
            t4_context='Tesla T4 FP64 throughput is about 1/32 of FP32; this is hardware context, not a measured workload speed ratio.')
        for backend in ('gpu', 'cpu'):
            for policy in args.policies:
                path = Path(prefix + backend + '_' + policy + '.json')
                run(benchmark_arguments(case, backend, policy, path, limit if backend == 'gpu' else 8.))
                # Guard failures are evidence, not grounds to raise the limit.
                assert path.exists(), 'A failed command produced no benchmark evidence'
        assert all(sha(p) == h for p, h in runtime['sources'].items())
        report['status'] = 'measurements_finished_pending_numerical_audit'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            report['interrupted_child_returncode'] = child.returncode
        report['finished_unix'] = time.time()
        save_json(output, report)


if __name__ == '__main__':
    main()
