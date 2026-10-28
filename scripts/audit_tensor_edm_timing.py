"""Report single-run fit timings only after the native-policy minimum gate."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import ROOT, verified_source_record


def gpu_memory_summary(tensor):
    """Keep unavailable NVML process attribution separate from measured peaks.

    Condor PID namespaces can hide the NVML host PID. Device-wide samples and
    allocator counters remain useful measurements, but neither is total process
    GPU memory. Missing attribution must not be represented as a zero or inferred
    by subtracting the device baseline.
    """
    gpu=tensor['resources'].get('gpu')
    assert gpu and gpu['error'] is None
    assert gpu['samples']>0 and math.isfinite(gpu['period_seconds']) and gpu['period_seconds']>0
    total,peak,baseline=(gpu[k] for k in ('device_total_mib','device_peak_mib','device_baseline_mib'))
    assert all(math.isfinite(x) for x in (total,peak,baseline))
    assert 0<=baseline<=peak<=total and total>0
    process=gpu['process_peak_mib']
    if gpu['process_seen']:
        assert process is not None and math.isfinite(process) and 0<process<=peak
    else:
        assert process is None
    allocator=tensor.get('jax_allocator',{}).get('devices',[])
    assert len(allocator)==1 and allocator[0]['available'], 'One assigned GPU allocator capture required'
    counters=allocator[0]['counters']
    live=counters['peak_bytes_in_use']
    assert math.isfinite(live) and 0<live<=total*2**20
    return dict(process_attribution_available=gpu['process_seen'],
        sampled_process_peak_mib=process, sampled_assigned_device_peak_mib=peak,
        assigned_device_baseline_mib=baseline, jax_peak_live_bytes=live,
        scope='NVML device-wide samples include other users and are not process memory. '
              'JAX allocator peak excludes driver/context and non-JAX allocations. '
              'Unavailable process attribution stays null; no baseline-subtracted estimate is made.')


def timing_summary(native, tensor, comparison, *, native_validation=None):
    """Validate eligibility before calculating any ratio (no numerical work)."""
    if comparison['case'] == 'hhvv':
        assert comparison.get('dataset') == native.get('dataset') == 'asimov_prefit'
        assert native_validation is not None, 'HHVV speed claims require the completed Asimov native gradient gate'
        assert native_validation['status'] == 'passed' and native_validation['nll_passed']
        assert native_validation['dataset'] == 'asimov_prefit' and len(native_validation['points']) == 26
        assert sorted(p['index'] for p in native_validation['points']) == list(range(26))
        assert native_validation['workspace_sha256'] == native['input_workspace_sha256']
        assert native_validation['model_sha256'] == tensor['model_sha256']
        assert native_validation['backend'] == tensor['backend']
    assert comparison['status'] == 'passed_same_minimum_gates_no_speed_claim'
    assert all(comparison[k] for k in ('same_start', 'same_workspace_dataset',
        'same_fit_state', 'parameter_gate_passed', 'minimum_nll_gate_passed'))
    assert comparison['same_point_nll_error'] < 1e-6
    assert abs(comparison['minimum_nll_difference']) <= 2e-4
    assert native['status'] == 'measured_fit_pending_agreement' and native['fit_scope'] == 'migrad'
    assert tensor['status'] == 'measured_pending_validation' and tensor['operation'] == 'fit'
    assert native['host'] == tensor['host'] == comparison['same_host']
    assert native['cpu_model'] == tensor['cpu_model']
    assert native['cpu_affinity'] == tensor['cpu_affinity'] and len(tensor['cpu_affinity']) <= 8
    assert native['root_version'] == '6.40.00' and native['combine_version'] == '11.0.0'
    assert tensor['float64'] and not tensor['root_imported']
    assert native['fit']['status'] == tensor['fit_result']['status'] == 0
    assert 0 <= native['fit']['edm'] < 1e-4
    assert 0 <= tensor['fit_result']['corrected_estimated_edm'] < 1e-4
    assert native['fit_state'] == native['preflight']['initial_state']
    assert native['preflight']['status'] == 'passed'
    assert tensor['fit_warm_policy']['requested_repetitions'] == 0
    assert tensor['timings']['warm_seconds'] == []
    assert tensor['first_fit_checkpoint']['arrays_sha256'] == tensor['arrays_sha256']
    native_counts, tensor_counts = native['evaluation_counts'], tensor['evaluation_counts']
    assert native_counts['actual_nll_evaluations'] == native_counts['function'] > 0
    assert native_counts['numerical_gradient_evaluations'] > 0
    assert tensor_counts['nll_total'] == tensor_counts['nll_optimizer'] + tensor_counts['hessian_linearizations']
    assert tensor_counts['gradient_total'] == tensor_counts['gradient_optimizer'] + tensor_counts['hessian_linearizations']
    assert tensor_counts['gradient_optimizer'] > 0
    assert native['conservative_combined_peak_rss_mib'] <= native['rss_limit_mib']
    assert tensor['resources']['peak_rss_mib'] <= tensor['resources']['rss_limit_mib']
    gpu_memory = None
    if tensor['backend'] == 'gpu':
        gpu_memory = gpu_memory_summary(tensor)
        assert tensor['environment']['COMBINE_JAX_LCG_VIEW']
    timings = tensor['timings']
    execution = timings['first_execute_seconds']
    native_wall, migrad = native['executable_wall_seconds'], native['migrad_seconds']
    for value in (execution, native_wall, migrad):
        assert math.isfinite(value) and value > 0
    assert migrad <= native_wall
    for key in ('lower_seconds', 'compile_or_cache_load_seconds', 'load_seconds', 'prepare_and_transfer_seconds'):
        assert math.isfinite(timings[key]) and timings[key] >= 0
    return dict(status='passed_same_minimum_single_run_timing', case=comparison['case'], dataset=native.get('dataset','data_obs'),
        host=native['host'], cpu_model=native['cpu_model'], cpu_affinity=native['cpu_affinity'],
        backend=tensor['backend'], devices=tensor['devices'], jax_version=tensor['jax_version'],
        jaxlib_version=tensor['jaxlib_version'], float64=True, environment=tensor['environment'],
        native_versions=dict(combine=native['combine_version'], root=native['root_version']),
        native_executable_wall_seconds=native_wall, native_migrad_seconds=migrad,
        tensor_execution_seconds=execution, tensor_preparation_and_compile_seconds={
            k: timings[k] for k in ('load_seconds', 'prepare_and_transfer_seconds',
                                   'lower_seconds', 'compile_or_cache_load_seconds')},
        ratios=dict(native_executable_over_tensor_execution=native_wall / execution,
                    native_migrad_over_tensor_execution=migrad / execution),
        counts=dict(native=native_counts, tensor=tensor_counts),
        minimum_nll_difference=comparison['minimum_nll_difference'],
        same_point_nll_error=comparison['same_point_nll_error'],
        max_parameter_delta_over_sigma=comparison['max_parameter_delta_over_sigma'],
        poi=comparison['poi'], leading_parameters=comparison['leading_parameters'],
        covariance_certified=comparison['covariance_certified'],
        native_edm=comparison['native_edm'], tensor_edm=comparison['tensor_edm'],
        tensor_exact_hessian_edm=comparison['tensor_exact_hessian_edm'],
        resources=tensor['resources'], jax_allocator=tensor.get('jax_allocator'), gpu_memory=gpu_memory,
        scope='One synchronized tensor fit including its final exact Hessian, versus one native MIGRAD run. '
              'Native executable wall includes startup/input/output; MIGRAD time excludes them. '
              'Tensor model preparation, lowering and compilation are separate. Ratios describe these '
              'single runs, not a statistical speedup estimate. Other users on the node are uncontrolled. '
              'HHVV covariance remains uncertified at the accepted clipping surfaces.')


def fit_command_order(driver, native_path, tensor_path):
    """Establish completed, nonoverlapping fit intervals in either run order."""
    commands=[]
    assert Path(native_path).name!=Path(tensor_path).name
    for path in (native_path,tensor_path):
        matches=[row for row in driver['commands'] if '--output' in row['command']
            and Path(row['command'][row['command'].index('--output')+1]).name==Path(path).name]
        assert len(matches)==1
        row=matches[0]
        assert row['status']=='completed' and row['returncode']==0
        start,end=row['started_unix'],row['finished_unix']
        assert math.isfinite(start) and math.isfinite(end) and start<end
        commands.append(row)
    if commands[0]['finished_unix']<=commands[1]['started_unix']:return 'native-first'
    assert commands[1]['finished_unix']<=commands[0]['started_unix'], 'Fit commands overlap'
    return 'tensor-first'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('native', 'tensor', 'comparison', 'driver', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--native-validation', type=Path,
                        help='Required for HHVV: successful 26-point Asimov native NLL/gradient audit')
    args = parser.parse_args()
    assert not args.output.exists()
    comparison = json.loads(args.comparison.read_text())
    if comparison['case'] == 'hhvv':
        assert comparison.get('dataset') == 'asimov_prefit', 'HHVV timing requires an Asimov comparison'
    native, tensor, driver = (json.loads(p.read_text()) for p in
        (args.native, args.tensor, args.driver))
    source = {label: verified_source_record(report, producer, args.source_revision)
        for label, report, producer in (
            ('native', native, 'scripts/benchmark_combine_fit.py'),
            ('tensor', tensor, 'scripts/benchmark_tensor.py'))}
    commit = source['native']['source_commit']
    assert driver['source_revision'] == commit
    for report, producer in ((driver, 'scripts/run_tensor_edm_campaign.py'),
                             (comparison, 'scripts/audit_tensor_edm_comparison.py')):
        blob = subprocess.check_output(['git', '-C', str(ROOT), 'cat-file', 'blob', commit + ':' + producer])
        assert hashlib.sha256(blob).hexdigest() == report['script_sha256']
    for path in (args.native, args.tensor):
        matching = [digest for name, digest in comparison['artifacts'].items() if Path(name).name == path.name]
        assert matching == [sha(path)], 'Comparison must bind the exact measured report'
    # Completed commands establish sequencing even while a later Hessian runs.
    fit_order=fit_command_order(driver,args.native,args.tensor)
    validation=json.loads(args.native_validation.read_text()) if args.native_validation else None
    result = timing_summary(native, tensor, comparison, native_validation=validation)
    result.update(script_sha256=sha(__file__), source_evidence=source,
        artifacts={str(p): sha(p) for p in (args.native, args.tensor, args.comparison, args.driver)},
        fit_commands_sequential=True,fit_order=fit_order,
        native_then_tensor_sequential=fit_order=='native-first')
    if args.native_validation:
        result['artifacts'][str(args.native_validation)]=sha(args.native_validation)
    save_json(args.output, result)
    print(result['status'], result['ratios'], flush=True)


if __name__ == '__main__':
    main()
