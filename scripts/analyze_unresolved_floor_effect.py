"""Quantify a measured floor transition using an explicit modified-branch diagnostic.

This reads saved native samples only. Modified-branch values are counterfactual,
not Combine references, and must never be substituted into an acceptance gate.
The native mixture, QCD contribution and yields are retained; only the change
between two reconstructed template floor rules is added to saved intensities.
"""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os
import resource
import sys
import time

import numpy as np

from hhvv_export_cache import load_export


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert os.sched_getaffinity(0) == {5}
    started = time.perf_counter()
    output = Path('reports/gpu_unresolved_floor_effect.json')
    if output.exists():
        raise FileExistsError(output)
    report_path = Path('reports/gpu_unresolved_curvature_bins.json')
    audit_path = Path('reports/gpu_unresolved_curvature_bins_audit.json')
    preflight_path = Path('reports/gpu_unresolved_curvature_bins_preflight.json')
    native, audit, preflight = [json.loads(p.read_bytes()) for p in (report_path, audit_path, preflight_path)]
    assert native['status'] == 'diagnostic_complete' and audit['status'] == 'diagnostic_verified'
    assert audit['producer_sha256'] == digest(report_path)
    assert digest(native['arrays']) == native['arrays_sha256'] == audit['arrays_sha256']
    export = Path('build/gpu_exports/hhvv.pkl')
    assert digest(export) == preflight['frozen_sources'][str(export)]
    model = load_export(export, native['source'])
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 < 1000
    channel = next(c for c in model.channels if c.name == native['channel'])
    with np.load(native['arrays'], allow_pickle=False) as stream:
        arrays = {k: stream[k] for k in stream.files}
    np.testing.assert_array_equal(arrays['initial'], model.initial)
    np.testing.assert_array_equal(arrays['parameter_names'], model.parameter_names)
    np.testing.assert_array_equal(arrays['weights'], channel.data.weights)
    np.testing.assert_array_equal(arrays['positions'], np.column_stack([
        channel.data.coordinates[name] for name in channel.observables]))
    assert native['row_observables'] == list(channel.observables)
    geometry = []
    volume, indices, coefficient_indices = [], [], []
    for name in native['components']:
        node = model.nodes[name]
        assert node.class_name == 'FastVerticalInterpHistPdf2D2'
        axes, edges = node.attrs['axes'], node.attrs['edges']
        centers = [(e[:-1]+e[1:])/2 for e in edges]
        coordinates = np.array(np.meshgrid(*centers, indexing='ij')).reshape(2, -1).T
        row_positions = arrays['positions'][:, [list(channel.observables).index(a) for a in axes]]
        mapping = []
        for row in row_positions:
            matches = np.flatnonzero(np.all(coordinates == row, axis=1))
            assert len(matches) == 1
            mapping.append(int(matches[0]))
        assert sorted(mapping) == list(range(20)), 'This diagnostic requires one row per bin'
        sizes = np.prod(np.array(np.meshgrid(*[np.diff(e) for e in edges], indexing='ij')), axis=0).reshape(-1)
        assert sizes.shape == (20,) and np.all(sizes > 0)
        volume.append(sizes)
        indices.append(mapping)
        coefficient_indices.append(native['mixture_components'].index(name))
        geometry.append(dict(name=name, axes=list(axes), edges=[e.tolist() for e in edges],
            row_to_bin=mapping, bin_volumes=sizes.tolist(),
            coefficient_index=coefficient_indices[-1]))
    volume = np.asarray(volume, dtype=np.longdouble)
    raw = arrays['raw_bins'].astype(np.longdouble)
    floor = np.longdouble(native['floor'])
    actual_floor = np.maximum(raw, floor)
    integral = np.sum(actual_floor*volume[None, :, :], axis=2, dtype=np.longdouble)
    assert np.all(integral > 0)
    rebuilt = actual_floor/integral[:, :, None]
    np.testing.assert_allclose(rebuilt, arrays['root_bins'], atol=1e-14, rtol=1e-13)
    density_error = float(np.max(abs(rebuilt-arrays['root_bins'])))
    density_relative = float(np.max(abs(rebuilt-arrays['root_bins'])/arrays['root_bins']))
    nominal_mask = raw[0] < floor
    selected_steps = native['selection_pair']
    selected_bins = sorted({(b['component'], b['bin']) for sample in native['samples']
        if sample['step'] in selected_steps for b in sample['changed_bins']})
    assert selected_bins == [(11, 12)], 'Selection must follow the saved native observation'
    selected_geometry = []
    for component, b in selected_bins:
        row = indices[component].index(b)
        selected_geometry.append(dict(component=component, bin=b,
            name=native['components'][component], row=row,
            row_coordinates=arrays['positions'][row].tolist(),
            row_observables=native['row_observables'],
            nominal_raw=float(raw[0, component, b]), nominal_floor_active=bool(nominal_mask[component, b])))
    weights = arrays['weights']
    counted = weights > 0
    nominal_intensity = arrays['intensities'][0]
    stencil = np.asarray([-1., 16., 16., -1.], dtype=np.longdouble)

    def nll_delta(intensity, expected):
        assert np.isfinite(intensity).all() and np.all(intensity[counted] > 0)
        ratio = (intensity[counted]-nominal_intensity[counted])/nominal_intensity[counted]
        return expected-arrays['expected'][0]-np.sum(weights[counted]*np.log1p(ratio), dtype=np.longdouble)

    original_delta = np.asarray([nll_delta(i, e) for i, e in zip(
        arrays['intensities'], arrays['expected'], strict=True)], dtype=np.longdouble)
    delta_error = float(np.max(abs(original_delta-np.asarray([s['stable_delta'] for s in native['samples']]))))
    assert delta_error < 1e-12
    analyses = []
    for mode in ('selected_bin_nominal_branch', 'all_template_nominal_branches'):
        alternate = actual_floor.copy()
        if mode == 'selected_bin_nominal_branch':
            for component, b in selected_bins:
                alternate[:, component, b] = floor if nominal_mask[component, b] else raw[:, component, b]
        else:
            alternate = np.where(nominal_mask[None, :, :], floor, raw)
        rows, deltas = [], []
        for i, sample in enumerate(native['samples']):
            if np.any(alternate[i] < 0):
                rows.append(dict(sample=i, step=sample['step'], multiple=sample['multiple'],
                    status='outside_nonnegative_template_domain', nll_delta=None))
                deltas.append(None)
                continue
            norms = np.sum(alternate[i]*volume, axis=1, dtype=np.longdouble)
            assert np.all(norms > 0)
            density_change = alternate[i]/norms[:, None]-rebuilt[i]
            intensity_change = np.zeros(20, dtype=np.longdouble)
            for component, coefficient in enumerate(coefficient_indices):
                intensity_change += np.longdouble(arrays['coefficients'][i, coefficient])*density_change[component, indices[component]]
            if np.array_equal(alternate[i], actual_floor[i]):
                np.testing.assert_array_equal(intensity_change, np.zeros(20))
            intensity = arrays['intensities'][i]+intensity_change
            if not np.all(intensity[counted] > 0):
                rows.append(dict(sample=i, step=sample['step'], multiple=sample['multiple'],
                    status='outside_positive_counted_intensity_domain', nll_delta=None))
                deltas.append(None)
                continue
            delta = nll_delta(intensity, arrays['expected'][i])
            deltas.append(delta)
            rows.append(dict(sample=i, step=sample['step'], multiple=sample['multiple'], status='computed_counterfactual',
                nll_delta=float(delta), change_from_native=float(delta-original_delta[i]),
                maximum_intensity_change=float(np.max(abs(intensity_change)))))
        assert deltas[0] == original_delta[0] == 0
        probes = []
        for i, original in enumerate(native['probes']):
            sl = slice(1+4*i, 5+4*i)
            h = np.longdouble(original['step'])
            actual = float(np.sum(stencil*original_delta[sl], dtype=np.longdouble)/(12*h*h))
            assert abs(actual-original['stable_channel_curvature']) < 1e-4
            values = deltas[sl]
            modified = (float(np.sum(stencil*np.asarray(values, dtype=np.longdouble), dtype=np.longdouble)/(12*h*h))
                        if all(v is not None for v in values) else None)
            probes.append(dict(step=original['step'], native_curvature=original['stable_channel_curvature'],
                reconstructed_native_curvature=actual, counterfactual_curvature=modified,
                native_minus_counterfactual=None if modified is None else actual-modified))
        selected = [next(p for p in probes if p['step'] == step) for step in selected_steps]
        assert all(p['counterfactual_curvature'] is not None for p in selected)
        native_change = selected[1]['native_curvature']-selected[0]['native_curvature']
        counterfactual_change = selected[1]['counterfactual_curvature']-selected[0]['counterfactual_curvature']
        analyses.append(dict(mode=mode, samples=rows, probes=probes,
            selected_pair=dict(steps=selected_steps, native_curvature_change=native_change,
                counterfactual_curvature_change=counterfactual_change,
                change_removed=native_change-counterfactual_change)))
    assert 'ROOT' not in sys.modules
    inputs = [report_path, audit_path, preflight_path, export, Path(native['arrays'])]
    report = dict(status='counterfactual_diagnostic_complete', created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=digest(__file__), inputs={str(p): digest(p) for p in inputs},
        source=native['source'], channel=native['channel'], direction=native['direction_label'],
        original_combine_version=native['combine_version'], original_root_version=native['root_version'],
        root_imported=False, new_native_evaluations=0, ad_arrays_read=False,
        geometry=geometry, selected_bins=selected_geometry,
        native_density_reconstruction_max_absolute_error=density_error,
        native_density_reconstruction_max_relative_error=density_relative,
        native_nll_delta_reconstruction_max_absolute_error=delta_error,
        normalization='Sum of floored native raw-bin density times exact exported bin volume; normalized per template in long double',
        intervention='Add to each saved native intensity only coefficient times the density difference between the explicit alternate floor rule and reconstructed actual floor rule. Keep yields, QCD and all other contributions unchanged. When rules coincide, the intensity change is exactly zero.',
        analyses=analyses, cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_seconds=time.perf_counter()-started,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        scope='Counterfactual explanation of the measured native channel behavior at unchanged points only. The changed-branch values are not Combine references or a modified production likelihood. No AD comparison, reference promotion or acceptance-policy change.')
    with output.open('x') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(output, report['status'], 'density errors', density_error, density_relative, 'NLL delta error', delta_error)
    print('selected bin', selected_geometry)
    for analysis in analyses:
        print(analysis['mode'], analysis['selected_pair'])
        print('curvatures', [(p['step'], p['counterfactual_curvature']) for p in analysis['probes']])


if __name__ == '__main__':
    main()
