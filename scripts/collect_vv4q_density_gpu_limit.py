"""Validate and freeze completed GPU job 16972363.0 against the native-v10 gate."""
import json
from pathlib import Path
import shutil

from tensor_benchmark_common import save_json, sha


def main():
    raw = Path('build/vv4q_density_gpu_campaign_20261024/results')
    source = raw/'gpu_vv4q_noVH_MX4000_density_limit.json'
    report = json.loads(source.read_text())
    worker = json.loads((raw/'worker_exit.json').read_text())
    reference_path = Path('reports/asymptotic_vv4q_noVH_MX4000_density_limit_comparison.json')
    cpu = json.loads(reference_path.read_text())
    assert report['reference_sha256'] == sha(reference_path)
    assert report['script_sha256'] == sha('scripts/run_vv4q_density_gpu_limit.py')
    assert report['status'] == 'passed' and worker['exit_code'] == 0
    assert report['float64'] and report['static_guard_gib'] == 8 and not report['root_imported']
    assert report['template_convention'] == 'v10_density' and report['all_fits_converged']
    assert len(report['limits']) == 5 and all(p['status'] == 0 for p in report['profile_fits'])
    rows = []
    for value, old in zip(report['limits'], cpu['rows'], strict=True):
        error = abs(value-old['native_v10'])/abs(old['native_v10'])
        assert error <= 1e-6 and value == old['combine_jax']
        rows.append(dict(quantile=old['quantile'], native_v10=old['native_v10'],
                         cpu=old['combine_jax'], gpu=value, relative_error=error))
    static = max(v['total_bytes'] for v in report['compiler_memory'].values())
    assert static <= 8*2**30
    output = Path('reports/gpu_vv4q_noVH_MX4000_density_limit.json')
    shutil.copyfile(source, output)
    summary = dict(status='passed', job_id='16972363.0', script_sha256=sha(__file__),
        raw_report=str(output), raw_report_sha256=sha(output), worker_exit=worker,
        rows=rows, relative_tolerance=1e-6, maximum_relative_error=max(r['relative_error'] for r in rows),
        all_fits_converged=True, fits=len(report['profile_fits']), evaluations=report['evaluation_counts'],
        gpu_timings=report['timings'], artifact_load_seconds=report['artifact_load_seconds'],
        prepared_model_wall_seconds=report['prepared_model_wall_seconds'],
        gpu_process_seconds=report['process_seconds'],
        shared_cpu_preparation_seconds=report['preparation']['preparation_seconds'],
        gpu_total_with_shared_preparation_seconds=report['total_method_seconds_with_shared_preparation'],
        gpu_process_plus_shared_preparation_seconds=report['process_seconds']+report['preparation']['preparation_seconds'],
        cpu_total_seconds=cpu['total_method_seconds'], cpu_execution_seconds=cpu['execution_seconds'],
        native_v10_seconds=cpu['native_seconds'],
        execution_cpu_over_gpu=cpu['execution_seconds']/report['timings']['execution_seconds'],
        total_cpu_over_gpu=cpu['total_method_seconds']/report['total_method_seconds_with_shared_preparation'],
        native_over_gpu_total=cpu['native_seconds']/report['total_method_seconds_with_shared_preparation'],
        same_node_speedup_certified=False,
        timing_scope=report['timing_scope']+' Worker process time additionally includes GPU discovery/startup; queue and transfer time are excluded.',
        static_gib=static/2**30, jax_peak_gib=report['jax_allocator_final']['peak_bytes_in_use']/2**30,
        jax_peak_pool_gib=report['jax_allocator_final']['peak_pool_bytes']/2**30,
        resources=report['resources'], environment=report['environment'], versions=report['versions'],
        devices=report['devices'], host=report['host'])
    save_json('reports/gpu_vv4q_noVH_MX4000_density_limit_comparison.json', summary)
    print(summary['status'], summary['maximum_relative_error'], summary['gpu_timings'],
          summary['gpu_process_seconds'], summary['gpu_process_plus_shared_preparation_seconds'])


if __name__ == '__main__':
    main()
