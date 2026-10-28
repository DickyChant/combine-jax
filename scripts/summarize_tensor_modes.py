"""Summarize same-node HHVV modes only after their full native-policy audits."""
import argparse
from datetime import datetime
import json
from pathlib import Path

import numpy as np

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', choices=('t4', 'a100', 'h100'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--native-report', type=Path,
                        help='Preserved native retry capture, separate from the original campaign')
    parser.add_argument('--native-checkpoint', type=Path,
                        help='Required download/provenance checkpoint for --native-report')
    args = parser.parse_args()
    if bool(args.native_report) != bool(args.native_checkpoint):
        parser.error('Native retry report and checkpoint must be supplied together')
    assert not args.output.exists()
    t4 = args.campaign == 't4'
    a100 = args.campaign == 'a100'
    prefix = {'t4':'gpu_tensor_t4_complete_', 'a100':'gpu_tensor_primary_phase3_',
              'h100':'gpu_tensor_primary_modes_'}[args.campaign]
    revision = {'t4':'c8b9072', 'a100':'b7b0f0b', 'h100':'fe13d0a'}[args.campaign]
    def path(label): return Path('reports')/(prefix+label+'.json')
    environment_path = path('environment')
    environment = json.loads(environment_path.read_text())
    assert environment['status'] == 'gpu_verified' and environment['float64']
    verified_source_record(environment, 'scripts/verify_tensor_gpu.py', revision)
    checkpoint_path = path('download_checkpoint_v2')
    checkpoint = json.loads(checkpoint_path.read_text())
    commands = checkpoint['driver_snapshot']['commands']
    for previous, current in zip(commands, commands[1:]):
        assert previous['finished_unix'] <= current['started_unix']
    modes = [
        ('jax_cpu_jit', 'jax_cpu' if t4 else 'jax_hhvv_cpu',
         'jax_cpu_native_gradient_audit' if t4 else 'cpu_native_gradient_audit'),
        ('jax_gpu_jit', 'jax_gpu' if t4 else ('cross_cache' if a100 else 'jax_hhvv_gpu'),
         'jax_gpu_native_gradient_audit' if t4 else ('cache_native_audit' if a100 else 'h100_native_gradient_audit')),
    ]
    for mode in ('cpu_eager', 'gpu_eager', 'gpu_cuda_graph'):
        label = ('torch_' if t4 else 'torch_hhvv_')+mode
        modes.append(('torch_'+mode, label, label+'_native_gradient_audit'))
    rows = []
    model_identity = None
    for mode, label, audit_label in modes:
        audit_path = path(audit_label)
        audit = json.loads(audit_path.read_text())
        assert audit['status'] == 'passed'
        assert audit['random']['status_counts'] == dict(passed=66300, fixed_bounds=100)
        for operation in ('nll', 'value_gradient'):
            report_path = path(label+'_'+operation)
            report = json.loads(report_path.read_text())
            assert report['status'] == 'measured_pending_validation' and report['float64']
            assert audit['benchmark_inputs'][str(report_path)] == sha(report_path)
            assert checkpoint['copied'][str(report_path)] == sha(report_path)
            assert sha(report['arrays']) == report['arrays_sha256']
            assert report['host'] == environment['host']
            assert report['cpu_model'] == environment['cpu_model']
            assert report['cpu_affinity'] == environment['cpu_affinity']
            identity = (report['model_sha256'], report['points_source_sha256'])
            if model_identity is None: model_identity = identity
            assert identity == model_identity
            samples = report['timings']['warm_seconds']
            assert len(samples) == 26 and np.isfinite(samples).all() and min(samples) > 0
            assert float(np.median(samples)) == report['timings']['warm_median_seconds']
            rows.append(dict(mode=mode, operation=operation, report=str(report_path),
                report_sha256=sha(report_path), native_audit=str(audit_path), native_audit_sha256=sha(audit_path),
                timings=report['timings'], resources=report['resources'],
                allocators={k:report[k] for k in ('jax_allocator','torch_allocator') if k in report},
                nll_max_absolute_errors=audit['nll_max_absolute_errors'],
                native_gradient_max_relative_error=max(audit['nominal']['max_relative_error'],
                                                       audit['random_max_relative_error'])))
    native_path = args.native_report or path('native_nll')
    native = dict(status='unavailable', reason='Initial native call failed before evaluation: workspace was missing in isolated directory. Retry pending.',
                  failure_log=str(native_path.with_suffix('.log')))
    if native_path.exists():
        native_checkpoint = (json.loads(args.native_checkpoint.read_text())
                             if args.native_checkpoint else checkpoint)
        report = json.loads(native_path.read_text())
        assert report['status'] == 'measured_native_validated' and report['host'] == environment['host']
        assert report['comparison']['passed'] and report['comparison']['roofit_evaluation_errors'] == 0
        assert report['root_version'] == '6.40.00' and report['combine_version'] == '11.0.0'
        assert native_checkpoint['copied'][str(native_path)] == sha(native_path)
        if args.native_checkpoint:
            assert native_checkpoint['status'] == 'native_retry_completed_after_campaign'
            assert native_checkpoint['driver_finished_unix'] <= native_checkpoint['retry_started_unix']
            assert sha(native_checkpoint['driver']) == native_checkpoint['driver_sha256']
            completed = json.loads(Path(native_checkpoint['driver']).read_text())
            assert completed['finished_unix'] == native_checkpoint['driver_finished_unix']
            assert completed['status'] in ('partial_failures', 'completed_pending_audit')
            assert all(row['status'] in ('passed', 'failed') for row in completed['commands'])
            assert sha(native_checkpoint['retry_driver']) == native_checkpoint['retry_driver_sha256']
            retry = json.loads(Path(native_checkpoint['retry_driver']).read_text())
            assert retry['status'] == 'completed_pending_audit' and retry['returncode'] == 0
            assert retry['started_unix'] == native_checkpoint['retry_started_unix']
            assert retry['preceding_driver_sha256'] == native_checkpoint['driver_sha256']
            assert retry['command'][-2:] == ['--output', str(native_path)]
            assert sha(retry['log']) == retry['log_sha256']
            assert retry['started_unix'] <= datetime.fromisoformat(report['started_utc']).timestamp() <= retry['finished_unix']
        verified_source_record(report, 'scripts/benchmark_native_tensor.py', revision)
        assert sha(report['arrays']) == report['arrays_sha256']
        reference = json.loads(Path('reports/hhvv_random_values.json').read_text())
        with np.load(report['arrays'], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays['parameter_names'], reference['parameter_names'])
            np.testing.assert_array_equal(arrays['points'], [p['theta'] for p in reference['cases']])
            error = float(np.max(abs(arrays['values']-np.array([p['combine_nll'] for p in reference['cases']]))))
        assert error < 1e-6 and len(report['timings']['warm']) == 26
        native = dict(status='passed', report=str(native_path), report_sha256=sha(native_path),
                      max_absolute_error=error, timings=report['timings'],
                      runtime='Separate v11/ROOT 6.40 process; loaded native libraries checked against its own venv.')
        if args.native_checkpoint:
            native.update(checkpoint=str(args.native_checkpoint), checkpoint_sha256=sha(args.native_checkpoint),
                          scope='Retry after the sequential campaign completed; original failed log preserved.')
    elif args.native_report:
        raise FileNotFoundError(native_path)
    lookup = {(r['mode'], r['operation']):r['timings']['warm_median_seconds'] for r in rows}
    ratios = {}
    for operation in ('nll', 'value_gradient'):
        base = lookup['jax_gpu_jit', operation]
        ratios[operation] = {mode+'_over_jax_gpu_jit':lookup[mode, operation]/base
                            for mode in ('jax_cpu_jit', 'torch_cpu_eager', 'torch_gpu_eager', 'torch_gpu_cuda_graph')}
    save_json(args.output, dict(status='passed_five_modes_native_policy', script_sha256=sha(__file__),
        environment=str(environment_path), environment_sha256=sha(environment_path),
        checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
        source_revision=revision, host=environment['host'], cpu_model=environment['cpu_model'],
        cpu_affinity=environment['cpu_affinity'], devices=environment['devices'],
        gpu_driver=environment['resources']['gpu']['device_metadata'],
        lcg_view=environment['environment']['COMBINE_JAX_LCG_VIEW'], versions=environment['versions'],
        float64=True, cases=rows, native_nll=native, warm_median_ratios=ratios,
        cache_scope=('The A100 GPU JIT capture requested a transferred cache but missed and recompiled. '
                     'Warm execution timing is valid; this summary does not accept cross-job reuse.'
                     if a100 else 'Persistent cache disabled for these mode captures.'),
        t4_fp64_note='User-supplied hardware context: T4 FP64 throughput is about 1/32 FP32. This ratio was not measured here; H100/A100 are the primary GPU evidence.',
        scope='Same node, identical neutral model and 26 resident changing points; synchronized calls exclude host output copies. Sequential project commands; shared node/device activity from other users is not controlled. JAX cold lowering/compilation and CUDA graph capture are separate from warm calls. No fit/Hessian or cross-job-cache acceptance.'))
    print('passed', args.campaign, [(r['mode'], r['operation'], r['timings']['warm_median_seconds']) for r in rows], flush=True)


if __name__ == '__main__':
    main()
