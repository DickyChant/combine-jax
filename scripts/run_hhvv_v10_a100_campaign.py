"""One approved A100 fit and same-node native-v10 Migrad comparison, Asimov only."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    prefix = 'reports/gpu_hhvv_v10_a100_'
    output = Path(prefix+'driver.json')
    assert not output.exists()
    manifest = json.loads(Path('build/hhvv_v10_runtime.json').read_text())
    for name, checksum in manifest['files'].items():
        assert sha(name) == checksum, name
    report = dict(status='running', dataset='asimov_prefit', native_reference_version='10.3.3',
        template_convention='v10_density', static_guard_gib=8, float64=True,
        script_sha256=sha(__file__), manifest_sha256=sha('build/hhvv_v10_runtime.json'),
        commands=[], source_revision=manifest['source_revision'],
        scope='One A100 gradient-only EDM fit and one single-core native-v10 Migrad on the same worker; identical nominal Asimov, start, bounds and physical configuration.')
    save_json(output, report)

    def run(wrapper, arguments, timeout):
        command = ['bash', wrapper, *arguments]
        destination = Path(arguments[arguments.index('--output')+1])
        log = destination.with_suffix('.log')
        entry = dict(command=command, log=str(log), started_unix=time.time())
        report['commands'].append(entry)
        save_json(output, report)
        with log.open('x') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout)
        entry.update(returncode=completed.returncode, finished_unix=time.time(), log_sha256=sha(log))
        save_json(output, report)
        if completed.returncode:
            raise RuntimeError('Campaign command failed: '+str(destination))
        return json.loads(destination.read_text())

    try:
        env = run('scripts/run_tensor_lcg.sh', ['scripts/verify_tensor_gpu.py', '--output', prefix+'environment.json'], 180)
        assert len(env['devices']) == 1 and 'A100' in env['devices'][0]['kind']
        preflight = run('scripts/run_native_v10_cmssw.sh', ['scripts/run_hhvv_v10_native_fit.py',
            '--workspace', 'build/hhvv_v10/workspace_asimov.root', '--configuration', 'build/hhvv_v10/configuration.json',
            '--directory', 'build/hhvv_v10/worker_preflight', '--preflight-only',
            '--output', prefix+'native_preflight.json'], 900)
        assert preflight['status'] == 'passed_preflight_only'
        model = 'build/hhvv_v10/model'
        common = ['scripts/benchmark_tensor.py', '--model', model, '--backend', 'gpu',
                  '--binned-gradient', 'structural_sequential', '--compiler-memory-limit-gib', '8', '--repeat', '1']
        vg = run('scripts/run_tensor_lcg.sh', common+['--operation', 'value_gradient',
            '--points', 'build/hhvv_v10/points.npz', '--output', prefix+'value_gradient.json'], 3600)
        import numpy as np
        with np.load(vg['arrays'], allow_pickle=False) as a, np.load('build/hhvv_v10/reference.npz', allow_pickle=False) as b:
            np.testing.assert_array_equal(a['points'], b['points'])
            np.testing.assert_array_equal(a['parameter_names'], b['parameter_names'])
            errors = np.abs(a['values']-b['values'])
            report['nll_gate'] = dict(points=len(errors), max_absolute_error=float(errors.max()), tolerance=1e-6)
            assert np.isfinite(a['gradients']).all() and np.all(errors < 1e-6)
        fit = run('scripts/run_tensor_lcg.sh', common+['--operation', 'fit', '--fit-covariance', 'none',
            '--fit-edm-validation', 'estimated', '--fit-stopping-criterion', 'edm', '--fit-edm-tolerance', '0.0001',
            '--fit-max-iterations', '10000', '--single-fit', '--fit-uncertainty-note',
            'Four accepted HHVV clipping bins; covariance and uncertainties are not certified.',
            '--output', prefix+'fit.json'], 21600)
        report['optimizer_converged'] = fit['fit_result']['status'] == 0 and 0 <= fit['fit_result']['corrected_estimated_edm'] < 1e-4
        assert report['optimizer_converged']
        native = run('scripts/run_native_v10_cmssw.sh', ['scripts/run_hhvv_v10_native_fit.py',
            '--workspace', 'build/hhvv_v10/workspace_asimov.root', '--configuration', 'build/hhvv_v10/configuration.json',
            '--directory', 'build/hhvv_v10/native_fit', '--tensor-fit', fit['arrays'],
            '--output', prefix+'native_fit.json'], 45000)
        assert native['host'] == fit['host']
        with np.load(fit['arrays'], allow_pickle=False) as a:
            names = list(a['parameter_names'])
            np.testing.assert_array_equal(names, native['fit']['parameters'])
            np.testing.assert_array_equal(a['points'][0], native['initial_values'])
            value = a['fit_values']
            reference = np.asarray(native['fit']['values'])
            sigma = np.asarray(native['fit']['errors'])
            delta = value-reference
            pulls = np.divide(abs(delta), sigma, out=np.zeros_like(delta), where=sigma > 0)
            parameter_pass = bool(np.all(abs(delta) <= np.maximum(.01*abs(reference), .01*sigma)))
            delta_nll = float(a['fit_value'])-native['fit']['canonical_nll']
            same_point_pass = native['tensor_point']['error'] < 1e-6 and native['tensor_point']['roofit_errors'] == 0
            passed = native['status'] == 'converged_pending_gpu_comparison' and same_point_pass and abs(delta_nll) <= 2e-4 and parameter_pass
            def parameter(i):
                return dict(name=str(names[i]), tensor=float(value[i]), native=float(reference[i]),
                            native_error=float(sigma[i]), delta_over_sigma=float(pulls[i]))
            report['comparison'] = dict(status='passed_same_minimum' if passed else 'failed_same_minimum',
                same_point_nll_error=native['tensor_point']['error'], delta_nll=delta_nll,
                nll_tolerance=2e-4, parameter_tolerance='max(1% of native value, 0.01 native sigma)',
                parameter_gate_passed=parameter_pass, max_parameter_pull=float(pulls.max()),
                r_hhvv=parameter(names.index('r_hhvv')),
                top_parameters=[parameter(i) for i in np.argsort(pulls)[::-1][:10]],
                native_migrad_seconds=native['migrad_seconds'], gpu_timings=fit['timings'],
                gpu_resources=fit['resources'], gpu_jax_allocator=fit['jax_allocator'],
                native_evaluation_counts=native['evaluation_counts'], gpu_evaluation_counts=fit['evaluation_counts'],
                gpu_iterations=fit['fit_result']['iteration'],
                gpu_corrected_estimated_edm=fit['fit_result']['corrected_estimated_edm'],
                native_edm=native['fit']['edm'],
                accepted_execution_speed_ratio=(native['migrad_seconds']/fit['timings']['first_execute_seconds']) if passed else None,
                timing_scope='Same worker: native single-core Migrad versus one GPU fit execution; GPU preparation, lowering and compilation are reported separately. Ratio accepted only after both convergence and same-minimum gates.')
        report['status'] = report['comparison']['status']
        save_json(output, report)
    except BaseException as exc:
        report.update(status='failed_execution', error=repr(exc))
        raise
    finally:
        report['finished_unix'] = time.time()
        save_json(output, report)
    print(report['status'], report.get('comparison'), flush=True)


if __name__ == '__main__':
    main()
