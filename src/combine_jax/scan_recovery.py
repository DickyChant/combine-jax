"""Auditable restart and bidirectional continuation for a likelihood scan.

This is a local-minimum diagnostic, not proof of a global minimum. Original
fits and failed attempts remain evidence; the likelihood and EDM rule do not
change. The callback performs the same continuous fit as the ordinary driver.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from .scan_comparison import check_global_baseline


def load_previous_scan(path, *, model_sha256, dataset_identity, dataset, analysis, parameter_names):
    path = Path(path)
    report = json.loads(path.read_text())
    if (report.get('model_sha256') != model_sha256 or report.get('dataset_identity') != dataset_identity
            or report.get('dataset') != dataset or report.get('dataset_selection', {}).get('analysis') != analysis):
        raise ValueError('Restart requires identical model, dataset and analysis provenance')
    records = []
    for row in report['fits']:
        name = row['capture']
        if Path(name).name != name:
            raise ValueError('Restart captures must be sibling files')
        capture = path.parent/name
        if hashlib.sha256(capture.read_bytes()).hexdigest() != row['capture_sha256']:
            raise ValueError('Restart capture checksum differs')
        with np.load(capture, allow_pickle=False) as arrays:
            if not np.array_equal(arrays['parameter_names'], parameter_names):
                raise ValueError('Restart parameter bindings differ')
            values = arrays['values'].copy()
            if values.shape != (len(parameter_names),) or not np.isfinite(values).all():
                raise ValueError('Restart vector is not finite or correctly sized')
            if float(arrays['value']) != float(row['value']) or int(arrays['status']) != int(row['status']):
                raise ValueError('Restart capture and report fit results differ')
        records.append({**row, 'values': values, 'origin': 'original', 'source_capture': name})
    return records, dict(report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                        captures_verified=len(records), dataset=dataset, analysis=analysis)


def recover_global_fit(fit, previous, *, poi):
    """Refit one lowest conditional vector, retaining the original scan points.

    This repairs only the necessary global-denominator check. No conditional
    point is rerun or re-zeroed, and a better existing unrestricted fit is kept.
    """
    def valid(row):
        return int(row['status'])==0 and np.isfinite(float(row['value']))

    original=[r for r in previous if set(r.get('fixed',{}))=={poi}]
    if not original or any(not valid(r) for r in original):
        raise ValueError('Recovery requires finite converged original conditional fits')
    if len({float(r['fixed'][poi]) for r in original})!=len(original):
        raise ValueError('Duplicate original scan coordinates')
    seed=min(original,key=lambda r:float(r['value']))
    result=fit('recovery_refit',seed['values'],{})
    attempt={**result,'task':'recovery_refit','fixed':{},'origin':'global_recovery',
             'start_task':seed['task'],'start_value':float(seed['value'])}
    old_globals=[r for r in previous if not r.get('fixed') and valid(r)]
    choices=old_globals+([attempt] if valid(attempt) else [])
    best=min(choices,key=lambda r:float(r['value'])) if choices else attempt
    selected=sorted(original,key=lambda r:float(r['fixed'][poi]))
    baseline=check_global_baseline([best,*selected])
    converged=valid(attempt)
    status=('recovery_global_not_converged' if not converged else
            'invalid_global_baseline' if not baseline['valid'] else
            'global_recovery_requires_native_comparison')
    return dict(status=status,poi=poi,seed_task=seed['task'],attempts=[attempt],
        selected=selected,global_fit=best,baseline_check=baseline,
        all_attempts_converged=converged,conditional_fits_rerun=0,
        selected_policy='Original conditional fits unchanged; lowest converged unrestricted fit retained',
        agreement_claimed=False)


def recover_scan(fit, previous, *, poi, minimum):
    """Refit the lowest conditional vector, then walk both ways on the high side.

    At the upper boundary there is no higher neighbour: start the downward walk
    from the best existing endpoint vector. Each subsequent step uses the last
    converged neighbour. The lower envelope includes the original conditional
    fit, so recovery cannot discard an already known lower converged solution.
    """
    def valid(row):
        return int(row['status']) == 0 and np.isfinite(float(row['value']))

    original = [r for r in previous if set(r.get('fixed', {})) == {poi}]
    if not original or any(not valid(r) for r in original):
        raise ValueError('Recovery requires finite converged original conditional fits')
    points = {float(r['fixed'][poi]): r for r in original}
    if len(points) != len(original):
        raise ValueError('Duplicate original scan coordinates')
    selected_points = sorted(x for x in points if x >= minimum)
    lower_points = sorted(x for x in points if x < minimum)
    if not selected_points or not lower_points:
        raise ValueError('High-side recovery requires points and a lower neighbour')
    attempts = []

    def run(label, start, fixed, origin):
        result = fit(label, start['values'], fixed)
        row = {**result, 'task': label, 'fixed': fixed, 'origin': origin,
               'start_task': start['task'], 'start_value': float(start['value'])}
        attempts.append(row)
        return row

    seed = min(original, key=lambda r: float(r['value']))
    global_fit = run('recovery_refit', seed, {}, 'global_recovery')
    if not valid(global_fit):
        return dict(status='recovery_global_not_converged', seed_task=seed['task'], attempts=attempts,
                    selected=[], global_fit=global_fit, baseline_check={'valid': False})
    candidates = {x: [r] for x,r in points.items()}
    start = points[lower_points[-1]]
    for i,x in enumerate(selected_points):
        row = run('up_%03d'%i, start, {poi:x}, 'ascending_neighbour')
        candidates[x].append(row)
        if valid(row): start = row
    start = min((r for r in candidates[selected_points[-1]] if valid(r)), key=lambda r: float(r['value']))
    boundary_start = start['task']
    for i,x in enumerate(reversed(selected_points)):
        row = run('down_%03d'%i, start, {poi:x}, 'descending_neighbour')
        candidates[x].append(row)
        if valid(row): start = row
    selected = [min((r for r in candidates[x] if valid(r)), key=lambda r: float(r['value'])) for x in sorted(points)]
    best = min(selected, key=lambda r: float(r['value']))
    if not check_global_baseline([global_fit,best])['valid']:
        final = run('recovery_refit_final', best, {}, 'global_recovery_after_scan')
        if valid(final) and float(final['value']) < float(global_fit['value']): global_fit = final
    baseline = check_global_baseline([global_fit,*selected])
    all_converged = all(valid(r) for r in attempts)
    status = ('shape_study_requires_native_comparison' if baseline['valid'] and all_converged
              else 'invalid_global_baseline' if not baseline['valid'] else 'recovery_attempt_failures')
    return dict(status=status,seed_task=seed['task'],minimum=minimum,poi=poi,attempts=attempts,
        selected=selected,global_fit=global_fit,baseline_check=baseline,all_attempts_converged=all_converged,
        upper_boundary_start=boundary_start,upper_boundary_policy='Best existing endpoint; no higher neighbour exists',
        selected_policy='Lowest converged NLL among original, ascending and descending fits',
        agreement_claimed=False)


def summarize_recovery(result):
    """Keep vectors in their checked captures, not duplicated in JSON reports."""
    def row(r):
        return {k:r[k] for k in ('task','fixed','origin','start_task','start_value','value','status',
                                 'source_capture') if k in r}
    summary = {k:v for k,v in result.items() if k not in ('attempts','selected','global_fit')}
    summary.update(attempts=[row(r) for r in result['attempts']],
                   selected=[row(r) for r in result['selected']],global_fit=row(result['global_fit']))
    if result['selected']:
        minimum = min(float(r['value']) for r in result['selected'])
        summary['shape_only'] = dict(reference='Minimum of selected conditional grid; not a certified global minimum',
            nll_offset=minimum,agreement_claimed=False,
            points=[[float(r['fixed'][result['poi']]),2*(float(r['value'])-minimum)] for r in result['selected']])
    return summary
