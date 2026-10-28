"""Freeze returned HHVV GPU limit numbers, including failed runs, without a fit."""
import argparse
import json
import math
from pathlib import Path
import shutil

from tensor_benchmark_common import save_json, sha
from combine_jax.methods.asymptotic import expected_nll_crossing


def convergence_by_quantile(report):
    """Associate each crossing evaluation with its recorded conditional fit.

    Native-style geometric stopping can return an unevaluated midpoint. A
    successful crossing therefore certifies its evaluated fits, not a new fit
    at that returned midpoint.
    """
    fits = report.get('profile_fits', report.get('profiles', []))
    trace = report.get('trace', [])
    if not report.get('limits'):
        return []
    assert len(fits) == len(trace)+1 and fits[0]['kind'] == 'global'
    assert report['profile_calls'] == len(trace)
    assert report['quantiles'] == [.025, .16, .5, .84, .975]
    assert len(report['limits']) == len(report['crossing_reasons']) == 5
    grouped = {q: [] for q in report['quantiles']}
    for point, fit in zip(trace, fits[1:], strict=True):
        assert fit['kind'] == 'conditional' and point['mu'] == fit['mu']
        assert point['nll'] == fit['value']
        grouped[point['quantile']].append(fit)
    rows = []
    for q, value, reason in zip(report['quantiles'], report['limits'], report['crossing_reasons'], strict=True):
        evaluated = grouped[q]
        assert evaluated and math.isfinite(value) and value >= 0
        edms = [r['corrected_estimated_edm'] for r in evaluated]
        assert all(math.isfinite(x) for x in edms)
        target = expected_nll_crossing(q, report['confidence_level'])
        deltas = [r['value']-report['minimum_nll'] for r in evaluated]
        bracketed = min(deltas) <= target <= max(deltas)
        unresolved = not bracketed and reason == 'parameter_accuracy'
        rows.append(dict(quantile=q, limit=value, conditional_fits=len(evaluated),
            all_fits_converged=fits[0]['status'] == 0 and all(r['status'] == 0 for r in evaluated),
            statuses=sorted(set(r['status'] for r in evaluated)),
            max_corrected_estimated_edm=max(edms), raw_nll_edm_threshold=1e-4,
            edm_threshold_passed=all(x < 1e-4 for x in edms), crossing_reason=reason,
            target_delta_nll=target, last_delta_nll=deltas[-1],
            minimum_evaluated_delta_nll=min(deltas),
            last_crossing_residual=deltas[-1]-target,
            evaluated_bracket=bracketed,
            crossing_audit='unresolved_no_evaluated_bracket' if unresolved else 'no_bracket_anomaly_detected',
            last_evaluated_mu=evaluated[-1]['mu'],
            returned_limit_evaluated=any(r['mu'] == value for r in evaluated),
            evaluation_counts={k:sum(r[k] for r in evaluated) for k in
                ('iteration','function_evaluations','gradient_evaluations')}))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submission', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, default=Path('reports'))
    parser.add_argument('--staging', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--collection', type=Path)
    args = parser.parse_args()
    submission = json.loads(args.submission.read_text())
    job = submission['job_id']; mass = submission['mass']
    worker = json.loads((args.results/'worker_exit.json').read_text())
    source = args.results/('gpu_hhvv_MX'+str(mass)+'_expected_limit.json')
    report = json.loads(source.read_text())
    assert report['mass'] == mass and report['float64'] and report['static_guard_gib'] == 8.
    assert report['template_convention'] == 'v10_density' and not report['root_imported']
    assert report['discrete_policy'] == submission['discrete_policy']
    staging = json.loads(args.staging.read_text()) if args.staging else None
    if staging:
        assert sha(args.staging) == submission['staging_sha256']
        assert report['manifest_sha256'] == staging['manifest_sha256']
        assert report['preparation']['model_sha256'] == staging['model_sha256']
        assert report['preparation']['dataset'] == staging['dataset']
        assert report['dataset']['mode'] == 'prefit' and report['dataset']['signal_strength'] == 0.
        assert report['dataset']['dataset_identity'] == staging['dataset']['identity']
        assert report['fixed_discrete_choices'] == staging['fixed_discrete_choices']
    if args.manifest:
        assert sha(args.manifest) == report['manifest_sha256']
        manifest = json.loads(args.manifest.read_text())
        assert report['script_sha256'] == manifest['files']['scripts/run_hhvv_gpu_limit.py']
        assert report['source_revision'] == manifest['source_revision']
    collection = json.loads(args.collection.read_text()) if args.collection else None
    if collection:
        assert collection['job_id'] == job and collection['transfer']['returncode'] == 0
        for name, proof in collection['files'].items():
            path = args.collection.parent/name
            assert path.stat().st_size == proof['bytes'] and sha(path) == proof['sha256']
    prefix = 'gpu_hhvv_MX'+str(mass)+'_'+job.replace('.', '_')
    raw = args.output_directory/(prefix+'_limit.json')
    if raw.exists():
        assert sha(raw) == sha(source), 'Preserve the previous collection rather than replacing different evidence'
    else:
        shutil.copyfile(source, raw)
    rows = convergence_by_quantile(report)
    profiles = report.get('profile_fits', report.get('profiles', []))
    all_converged = bool(profiles) and all(p['status'] == 0 for p in profiles)
    complete = worker['exit_code'] == 0 and len(rows) == 5 and all_converged
    memory = report.get('compiler_memory', {})
    static = max((v['total_bytes'] for v in memory.values()), default=0)
    if complete:
        assert 0 < static <= 8*2**30
        assert all(r['all_fits_converged'] and r['edm_threshold_passed'] for r in rows)
        assert report['evaluation_counts'] == {k:sum(r[k] for r in profiles) for k in
            ('iteration','function_evaluations','gradient_evaluations')}
    preparation = report.get('preparation', {})
    unresolved = [r['quantile'] for r in rows if r['crossing_audit'] == 'unresolved_no_evaluated_bracket']
    status = ('computed_with_unresolved_crossings' if unresolved else 'computed_pending_native_validation') if complete else 'failed_execution'
    result = dict(status=status,
        job_id=job, mass=mass, script_sha256=sha(__file__), worker_exit=worker,
        submission_sha256=sha(args.submission), raw_report=str(raw), raw_report_sha256=sha(raw),
        worker_status=report['status'], error=report.get('error'), quantiles=rows,
        all_fits_converged=all_converged, fits=len(profiles), timings=report.get('timings'),
        global_fit=profiles[0] if profiles else None,
        evaluation_counts=report.get('evaluation_counts'), process_seconds=report.get('process_seconds'),
        prepared_model_wall_seconds=report.get('prepared_model_wall_seconds'),
        artifact_load_seconds=report.get('artifact_load_seconds'),
        generation_seconds=preparation.get('generation', {}).get('generation_seconds'),
        import_seconds=preparation.get('import_seconds'), resources=report.get('resources'),
        static_peak_gib=static/2**30 if memory else None,
        jax_allocator_final=report.get('jax_allocator_final'),
        environment=report.get('environment'), versions=report.get('versions'), devices=report.get('devices'),
        host=report.get('host'), discrete_policy=report['discrete_policy'],
        fixed_discrete_choices=submission['fixed_discrete_choices'],
        native_quantile_validation='pending_reference_and_matching_discrete_configuration',
        native_relative_tolerance=1e-6, speedup_certified=False,
        scope='Expected Asimov only. Fixed-discrete results are conditional on the recorded PDF choices. No native agreement or speed ratio is claimed until the matching reference is compared.')
    result.update(unresolved_crossing_quantiles=unresolved,
        convergence_scope='Fit status/estimated EDM are audited separately from crossing validity. A returned midpoint need not have a fit at that exact value. No new fits or reference values were used for this audit.')
    if complete:
        global_threshold = .002*.1*expected_nll_crossing(.5, report['confidence_level'])
        result['global_raw_nll_edm_threshold'] = global_threshold
        result['global_edm_threshold_passed'] = profiles[0]['corrected_estimated_edm'] < global_threshold
        assert result['global_edm_threshold_passed']
    if collection:
        ads = collection.get('queue', []) or collection.get('history', [])
        assert len(ads) == 1 and ads[0]['JobStatus'] == 4 and ads[0]['ExitCode'] == worker['exit_code']
        result.update(scheduler=ads[0], scheduler_wall_seconds=ads[0]['RemoteWallClockTime'],
            collection_sha256=sha(args.collection))
    if staging:
        setup = sum(staging[k] for k in ('workspace_build_seconds', 'asimov_generation_seconds', 'import_seconds'))
        result.update(workspace_build_seconds=staging['workspace_build_seconds'],
            successful_cpu_preparation_seconds=setup,
            worker_plus_successful_preparation_seconds=report['process_seconds']+setup,
            timing_scope='GPU process includes discovery, verification, load, setup, lowering, compilation and fit execution. Scheduler wall additionally includes worker startup/untar/environment. CPU preparation is reported separately; retries, queue and transfer time are excluded from their sum.')
    target = args.output_directory/(prefix+'_summary.json')
    save_json(target, result)
    print(job, mass, result['status'], rows, result['timings'], flush=True)


if __name__ == '__main__':
    main()
