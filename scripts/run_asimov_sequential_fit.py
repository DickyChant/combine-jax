"""One authorized GPU EDM fit and a conditional exact-HVP covariance row.

Native ROOT checks and timing acceptance are separate, after collection. This
worker imports only the neutral Asimov model into the LCG CUDA Python process.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


NOTE = 'Four accepted HHVV background clipping bins; covariance and uncertainties are not certified.'


def fit_arguments(model, output):
    return ['scripts/benchmark_tensor.py', '--model', str(model), '--backend', 'gpu',
        '--operation', 'fit', '--binned-gradient', 'structural_sequential',
        '--fit-covariance', 'none', '--fit-edm-validation', 'estimated',
        '--fit-stopping-criterion', 'edm', '--fit-edm-tolerance', '0.0001',
        '--fit-max-iterations', '10000', '--single-fit', '--repeat', '1',
        '--compiler-memory-limit-gib', '8', '--fit-uncertainty-note', NOTE,
        '--output', str(output)]


def converged(report):
    fit = report.get('fit_result', {})
    return (report.get('status') == 'measured_pending_validation'
            and fit.get('status') == 0 and fit.get('covariance_status') == 6
            and math.isfinite(fit.get('corrected_estimated_edm', math.nan))
            and 0 <= fit['corrected_estimated_edm'] < 1e-4)


def checked_reference(case, metadata, native, *, manifest_sha256):
    """Reject different data, start, bounds, physics configuration or policy."""
    from run_tensor_edm_campaign import hhvv_native_outcome
    from tensor_benchmark_dataset import benchmark_dataset
    from types import SimpleNamespace
    assert benchmark_dataset(case, SimpleNamespace(metadata=metadata), label='hhvv') == 'asimov_prefit'
    workspace_sha256 = case.get('workspace_sha256', case['hashes'][case['workspace']])
    outcome = hhvv_native_outcome(native, metadata, manifest_sha256=manifest_sha256,
                                  workspace_sha256=workspace_sha256, returncode=0)
    assert outcome['minimum_comparison_eligible']
    return outcome


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=('A100', 'H100'), required=True)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] != 'lxplus905'
    prefix = 'reports/gpu_tensor_edm_sequential_'
    output = Path(prefix + 'driver.json')
    assert not output.exists()
    runtime_path = Path('build/sequential_fit_runtime.json')
    runtime = json.loads(runtime_path.read_text())
    assert all(sha(p) == h for p, h in runtime['sources'].items())
    report = dict(status='running', dataset='asimov_prefit', family=args.family,
        script_sha256=sha(__file__), source_revision=runtime['source_revision'],
        runtime_manifest_sha256=sha(runtime_path), started_unix=time.time(), commands=[],
        static_guard_gib=8, float64=True,
        criterion=dict(native='Minuit2 Migrad strategy 1, tolerance 0.1, Up=0.5',
                       raw_nll_edm_target=1e-4, tensor='Device BFGS corrected estimated EDM'),
        scope='One gradient-only GPU fit, exact-HVP r_hhvv row conditional on convergence. No automatic retry, full Hessian or native fit. Numerical acceptance and fit timing ratios require subsequent independent native checks.')
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f'GPU fit campaign received signal {signum}')

    def run(arguments, timeout):
        nonlocal child
        command = ['bash', 'scripts/run_tensor_lcg.sh', *arguments]
        log = Path(arguments[arguments.index('--output') + 1]).with_suffix('.log')
        row = dict(command=command, log=str(log), started_unix=time.time(), status='running')
        report['commands'].append(row)
        save_json(output, report)
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            row['pid'] = child.pid
            save_json(output, report)
            row['returncode'] = child.wait(timeout=timeout)
        row.update(status='completed' if child.returncode == 0 else 'failed',
                   finished_unix=time.time(), log_sha256=sha(log))
        save_json(output, report)
        return row['returncode']

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        # Device discovery precedes model loading and every numerical operation.
        assert run(['scripts/verify_tensor_gpu.py', '--output', prefix + 'environment.json'], 180) == 0
        env = json.loads(Path(prefix + 'environment.json').read_text())
        assert len(env['devices']) == 1 and args.family in env['devices'][0]['kind']
        assert all(sha(p) == h for p, h in runtime['input_files'].items())
        manifest = Path(runtime['manifest'])
        assert sha(manifest) == runtime['manifest_sha256']
        case = json.loads(manifest.read_text())['hhvv']
        native_path = Path(runtime['native_references'][args.family])
        native = json.loads(native_path.read_text())
        assert native['host'] == socket.gethostname(), 'Same-node native timing is required'
        from native_fit_model import load_native_fit_model
        from combine_jax.tensor.parameter_info import model_parameter_info
        import numpy as np
        model, loader = load_native_fit_model(case['model'])
        info = model_parameter_info(model)
        state = checked_reference(case, model.metadata, native, manifest_sha256=sha(manifest))
        np.testing.assert_array_equal(native['initial_parameters'], info.metadata['names'])
        np.testing.assert_array_equal(native['initial_values'], info.arrays['initial'])
        report.update(native_reference=str(native_path), native_reference_sha256=sha(native_path),
                      native_state_gate=state, native_model_loader=loader,
                      same_host=True, same_initial_point=True,
                      dataset_identity=model.metadata['provenance']['dataset']['identity'])
        assert run(['scripts/benchmark_tensor.py', '--model', case['model'],
            '--points', case['points'], '--backend', 'gpu', '--operation', 'value_gradient',
            '--binned-gradient', 'structural_sequential', '--compiler-memory-limit-gib', '8',
            '--repeat', '3', '--output', prefix + 'value_gradient.json'], 3600) == 0
        fit_path = Path(prefix + 'fit.json')
        code = run(fit_arguments(case['model'], fit_path), 21600)
        fit = json.loads(fit_path.read_text())
        report['optimizer_converged'] = code == 0 and converged(fit)
        if report['optimizer_converged']:
            assert sha(fit['arrays']) == fit['arrays_sha256']
            with np.load(fit['arrays'], allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['parameter_names'], info.metadata['names'])
                values = saved['fit_values'].copy()
                assert np.isfinite(values).all() and int(saved['fit_status']) == 0
                np.testing.assert_array_equal(saved['points'][0], info.arrays['initial'])
            point = Path(prefix + 'fitted_point.npz')
            np.savez_compressed(point, points=values[None, :], parameter_names=np.asarray(info.metadata['names']))
            report['fit_summary'] = dict(final_nll=fit['fit_result']['value'],
                r_hhvv=float(values[info.metadata['names'].index('r_hhvv')]),
                iteration=fit['fit_result']['iteration'], evaluation_counts=fit['evaluation_counts'],
                timings=fit['timings'], resources=fit['resources'], jax_allocator=fit.get('jax_allocator'),
                fitted_point=str(point), fitted_point_sha256=sha(point))
            save_json(output, report)
            covariance = Path(prefix + 'covariance_row.json')
            run(['scripts/benchmark_tensor.py', '--model', case['model'], '--points', str(point),
                '--backend', 'gpu', '--operation', 'covariance_rows', '--covariance-parameters', 'r_hhvv',
                '--covariance-max-iterations', '1000', '--curvature-recompute-point',
                '--binned-gradient', 'structural_sequential', '--compiler-memory-limit-gib', '8',
                '--repeat', '1', '--output', str(covariance)], 21600)
            report['covariance_row_status'] = json.loads(covariance.read_text()).get('covariance_result')
        else:
            report['covariance_skipped'] = 'Fit did not converge under the unchanged EDM criterion and memory guard.'
        assert all(sha(p) == h for p, h in runtime['sources'].items())
        report['status'] = 'finished_pending_native_audit' if report['optimizer_converged'] else 'fit_failed_or_guarded'
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
    print(report['status'], flush=True)


if __name__ == '__main__':
    main()
