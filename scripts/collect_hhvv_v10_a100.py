"""Independently audit completed Asimov job 16972366 before any limit campaign."""
import json
from pathlib import Path
import shutil

import numpy as np

from tensor_benchmark_common import save_json, sha


def main():
    base = Path('build/hhvv_v10_campaign_20261024')
    raw = base/'results'
    prefix = 'gpu_hhvv_v10_a100_'
    load = lambda suffix: json.loads((raw/(prefix+suffix+'.json')).read_text())
    driver, native, fit, vg = (load(n) for n in ['driver', 'native_fit', 'fit', 'value_gradient'])
    assert driver['dataset'] == native['dataset'] == 'asimov_prefit'
    assert driver['template_convention'] == 'v10_density'
    assert native['combine_version'] == '10.3.3' and native['root_version'] == '6.30/07'
    assert native['host'] == fit['host'] and fit['float64'] and not fit['root_imported']
    assert sha(base/'generation/workspace_asimov.root') == native['workspace_sha256']
    for r in [fit, vg]:
        assert sha(raw/Path(r['arrays']).name) == r['arrays_sha256']
        assert r['model_sha256'] == {n: sha(base/'tensor/model'/n) for n in ['model.json', 'arrays.npz']}
        assert r['compiler_allocation_limit_bytes'] == 8*2**30
    with np.load(raw/Path(vg['arrays']).name, allow_pickle=False) as a, np.load(base/'reference.npz', allow_pickle=False) as b:
        np.testing.assert_array_equal(a['parameter_names'], b['parameter_names'])
        np.testing.assert_array_equal(a['points'], b['points'])
        nll_error = float(np.max(abs(a['values']-b['values'])))
        assert np.isfinite(a['gradients']).all()
    with np.load(raw/Path(fit['arrays']).name, allow_pickle=False) as a:
        names = list(a['parameter_names'])
        np.testing.assert_array_equal(names, native['fit']['parameters'])
        np.testing.assert_array_equal(a['points'][0], native['initial_values'])
        delta = a['fit_values']-np.asarray(native['fit']['values'])
        sigma = np.asarray(native['fit']['errors'])
        pulls = np.divide(abs(delta), sigma, out=np.full_like(delta, np.inf), where=sigma > 0)
        pulls[(sigma == 0) & (delta == 0)] = 0
        tolerance = np.maximum(.01*abs(np.asarray(native['fit']['values'])), .01*sigma)
        failed = abs(delta) > tolerance
        difference = float(a['fit_value'])-native['fit']['canonical_nll']
        gpu_converged = int(a['fit_status']) == 0 and 0 <= float(a['fit_corrected_estimated_edm']) < 1e-4
        native_converged = native['fit']['status'] == 0 and 0 <= native['fit']['edm'] < 1e-4
        passed = (gpu_converged and native_converged and nll_error < 1e-6
                  and native['tensor_point']['error'] < 1e-6
                  and abs(difference) <= 2e-4 and not failed.any())
        rows = [dict(name=str(names[i]), gpu=float(a['fit_values'][i]),
                     native=float(native['fit']['values'][i]), native_sigma=float(sigma[i]),
                     absolute_delta=float(abs(delta[i])), absolute_pull=float(pulls[i]),
                     parameter_gate_passed=bool(not failed[i]))
                for i in np.argsort(pulls)[::-1][:10]]
        poi = names.index('r_hhvv')
        poi_row = dict(gpu=float(a['fit_values'][poi]), native=native['fit']['values'][poi],
                       native_sigma=float(sigma[poi]), absolute_pull=float(pulls[poi]))
    assert driver['comparison']['status'] == ('passed_same_minimum' if passed else 'failed_same_minimum')
    assert driver['comparison']['delta_nll'] == difference
    for name in ['driver', 'environment', 'native_fit', 'native_preflight', 'fit', 'value_gradient']:
        shutil.copyfile(raw/(prefix+name+'.json'), Path('reports')/(prefix+name+'.json'))
    jobs = json.loads(Path('build/completed_jobs_audit_20261028/queue_before.json').read_text())
    job = next(j for j in jobs if j['ClusterId'] == 16972366)
    assert job['JobStatus'] == 4 and job['ExitCode'] == 0
    static = lambda r: sum(r['compiler_memory'][n] for n in ['argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes'])-r['compiler_memory']['alias_size_in_bytes']
    report = dict(status='passed_same_minimum' if passed else 'failed_same_minimum',
        job_id='16972366.0', script_sha256=sha(__file__), dataset='asimov_prefit',
        gate_allows_limit_campaign=passed, new_campaign_submitted=False,
        scheduler=job, worker_exit=json.loads((raw/'worker_exit.json').read_text()),
        collected_files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in raw.iterdir() if p.is_file()},
        nll_gate=dict(points=26, max_absolute_error=nll_error, tolerance=1e-6),
        same_point_nll_error=native['tensor_point']['error'],
        gpu_converged=gpu_converged, native_converged=native_converged,
        gpu_nll=fit['fit_result']['value'], native_nll=native['fit']['canonical_nll'],
        delta_nll=difference, delta_nll_tolerance=2e-4,
        parameters_failing_gate=int(failed.sum()), parameter_tolerance='max(1% of native value, 0.01 native sigma)',
        maximum_parameter_pull=float(pulls.max()), top_parameters=rows, r_hhvv=poi_row,
        gpu_edm=fit['fit_result']['corrected_estimated_edm'], native_edm=native['fit']['edm'],
        gpu_iterations=fit['fit_result']['iteration'], gpu_evaluations=fit['evaluation_counts'],
        native_evaluations=native['evaluation_counts'],
        native_migrad_seconds=native['migrad_seconds'], native_executable_seconds=native['executable_wall_seconds'],
        gpu_timings=fit['timings'], gpu_static_gib=static(fit)/2**30,
        value_gradient_static_gib=static(vg)/2**30,
        gpu_resources=fit['resources'], jax_allocator=fit['jax_allocator'],
        accepted_speed_ratio=None if not passed else native['migrad_seconds']/fit['timings']['first_execute_seconds'],
        scope='Same node, identical native-v10 nominal Asimov and checked starting state. Estimated EDM success does not waive the minimum gate. Native Migrad errors are not certified Hesse errors; the four accepted clipping bins are unchanged.')
    save_json('reports/gpu_hhvv_v10_a100_collected_audit.json', report)
    print(report['status'], 'delta NLL', difference, 'failed parameters', int(failed.sum()),
          'max pull', float(pulls.max()), flush=True)


if __name__ == '__main__':
    main()
