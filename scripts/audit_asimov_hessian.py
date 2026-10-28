"""Compare full nominal Asimov Hessians and replay independent native derivatives.

All inputs are explicit. Historical fitted-point and observed-data references
are neither defaults nor fallbacks. Matrix agreement is not a native curvature
or covariance certificate.
"""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from audit_asimov_native_reference import (add_centered_references,
    checked_native_point, compare_coordinate, verified_cached_deltas, verified_refinements)
from compare_local_cpu_hessian import matrix_errors, TOLERANCE
from native_fit_model import load_native_fit_model
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import benchmark_dataset
from tensor_benchmark_provenance import verified_source_record


def native_reference(gate_path, case, info, model_hashes):
    """Reconstruct native nominal references without using a tested AD value."""
    gate = json.loads(Path(gate_path).read_text())
    assert gate['status'] == 'passed' and gate['dataset'] == 'asimov_prefit'
    assert gate['script_sha256'] == sha('scripts/audit_asimov_native_reference.py')
    assert gate['workspace_sha256'] == sha(case['workspace'])
    assert gate['generation_report_sha256'] == sha(case['generation_report'])
    assert gate['model_sha256'] == model_hashes
    assert gate['nll_passed'] and 0 <= gate['max_nll_absolute_error'] < 1e-6
    artifacts = gate['artifacts']
    assert all(sha(path) == checksum for path, checksum in artifacts.items())
    fixed = info.arrays['lower'] == info.arrays['upper']
    counts = {k: v for k, v in dict(passed=int((~fixed).sum()), fixed_bounds=int(fixed.sum())).items() if v}
    assert gate['status_counts'] == {k: 26*v for k, v in counts.items()}
    assert [p['index'] for p in gate['points']] == list(range(26))
    for point in gate['points']:
        assert point['status_counts'] == counts and not point['unresolved_or_failed']
        assert 0 <= point['max_relative_error_on_certified_references'] < 1e-4
    base_path = gate['points'][0]['native_reference_report']
    assert base_path in artifacts
    native = json.loads(Path(base_path).read_text())
    assert native['dataset'] == 'asimov_prefit'
    for key in ('workspace_sha256', 'generation_report_sha256'):
        assert native[key] == gate[key]
    assert native['root_version'] == '6.40.00' and native['combine_version'] == '11.0.0'
    assert native['script_sha256'] == sha('scripts/capture_asimov_native_reference.py')
    assert native['fd_script_sha256'] == sha('scripts/native_fd_scan.py')
    assert sha(native['nll_arrays']) == native['nll_arrays_sha256'] == sha(case['reference'])
    names = np.asarray(info.metadata['names'])
    with np.load(native['nll_arrays'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['parameter_names'], names)
        points, values = saved['points'], saved['values']
    assert points.shape == (26, len(names)) and values.shape == (26,)
    assert points.dtype == values.dtype == np.float64
    assert np.isfinite(points).all() and np.isfinite(values).all()
    assert len(np.unique(points, axis=0)) == 26
    with np.load(case['points'], allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['parameter_names'], names)
        np.testing.assert_array_equal(saved['points'], points)
    np.testing.assert_array_equal(points[0], info.arrays['initial'])
    records = [p for p in native['points'] if p['index'] == 0]
    assert len(records) == 1
    raw = checked_native_point(records[0], names, points, values)

    # Rebuild all declared replacements with the original AD-independent
    # selection functions. The passed audit supplies identities, not estimates.
    paths = {method: set() for method in ('native_zero_point_subtraction',
        'native_cached_longdouble_delta', 'native_centered_histogram_score')}
    for row in gate['refinements_applied']:
        assert artifacts[row['source']] == row['source_sha256']
        paths[row['reference_method']].add(Path(row['source']))
        if 'previous_native_reference' in row:
            previous = row['previous_native_reference']
            assert artifacts[previous['source']] == previous['source_sha256']
            paths[previous['reference_method']].add(Path(previous['source']))
    replacements = verified_refinements(sorted(paths['native_zero_point_subtraction']), native, names, points, values)
    cached = verified_cached_deltas(sorted(paths['native_cached_longdouble_delta']), native, names, points, values)
    assert not set(replacements) & set(cached)
    replacements.update(cached)
    centered = verified_cached_deltas(sorted(paths['native_centered_histogram_score']),
        native, names, points, values, centered=True)
    add_centered_references(replacements, centered)
    declared = {(row['point'], row['index']) for row in gate['refinements_applied']}
    assert len(declared) == len(gate['refinements_applied']) and set(replacements) == declared
    references = [replacements.get((0, j), {'case': c})['case'] for j, c in enumerate(raw['cases'])]
    for j, reference in enumerate(references):
        assert reference['coordinate'] == points[0, j]
        assert (reference['status'] == 'fixed_bounds') == fixed[j]
        assert reference['status'] in ('fixed_bounds', 'converged')
    return references, float(values[0]), dict(gate=str(gate_path), gate_sha256=sha(gate_path),
        nominal_native_artifact_sha256=records[0]['artifact_sha256'],
        native_values_sha256=native['nll_arrays_sha256'], nominal_replacements=sum(i == 0 for i, _ in replacements),
        selection='Replayed native finite differences/cache deltas/centered scores; no AD input to selection.')


def nominal_errors(actual, references, native_value):
    results = [compare_coordinate(value, reference)
               for value, reference in zip(actual['gradients'][0], references, strict=True)]
    counts = dict(Counter(row['status'] for row in results))
    error = float(abs(actual['values'][0] - native_value))
    return dict(passed=bool(error < 1e-6 and set(counts) <= {'passed', 'fixed_bounds'}),
        nll_absolute_error=error, gradient_status_counts=counts,
        gradient_max_relative_error=max((r['max_relative_error'] for r in results if 'max_relative_error' in r), default=0.),
        failed_coordinates=[i for i, r in enumerate(results) if r['status'] not in ('passed', 'fixed_bounds')])


def checked_capture(path, info, model_hashes, structure_key, revision=None):
    report = json.loads(Path(path).read_text())
    assert report['status'] == 'measured_pending_validation' and report['stage'] == 'complete'
    assert report['operation'] == 'hessian' and report['float64'] and not report['root_imported']
    assert report['backend'] in ('cpu', 'gpu') and report['mode'] == 'jit'
    assert report['model_sha256'] == model_hashes and report['structure_key'] == structure_key
    source = verified_source_record(report, 'scripts/benchmark_tensor.py', revision)
    assert report['points_source'] == 'model initial' and report['points_source_sha256'] is None
    assert sha(report['arrays']) == report['arrays_sha256']
    with np.load(report['arrays'], allow_pickle=False) as saved:
        actual = {key: saved[key] for key in ('parameter_names', 'points', 'values', 'gradients', 'hessians')}
    names = info.metadata['names']; size = len(names)
    assert report['parameters'] == size and report['points'] == 1
    np.testing.assert_array_equal(actual['parameter_names'], names)
    np.testing.assert_array_equal(actual['points'], info.arrays['initial'][None, :])
    for key, shape in (('points', (1, size)), ('values', (1,)),
                       ('gradients', (1, size)), ('hessians', (1, size, size))):
        assert actual[key].shape == shape and actual[key].dtype == np.float64
        assert np.isfinite(actual[key]).all()
    fixed = info.arrays['lower'] == info.arrays['upper']
    np.testing.assert_array_equal(actual['gradients'][0, fixed], 0.)
    np.testing.assert_array_equal(actual['hessians'][0, fixed, :], 0.)
    np.testing.assert_array_equal(actual['hessians'][0, :, fixed], 0.)
    return report, actual, source


def full_matrix_errors(cpu, candidate, info):
    from combine_jax.tensor.parameter_info import fit_coordinates
    coordinates = fit_coordinates(info)
    free = coordinates['free_indices']
    common = dict(parameter_names=np.asarray(info.metadata['names'])[free],
        point=info.arrays['initial'][free], parameter_scales=coordinates['scale'])
    return matrix_errors(dict(common, hessian=cpu['hessians'][0][np.ix_(free, free)]),
                         dict(common, hessian=candidate['hessians'][0][np.ix_(free, free)]))


def compare(*, manifest, cpu, benchmark, native_gate, cpu_revision=None, source_revision=None):
    from combine_jax.tensor.parameter_info import model_parameter_info
    case = json.loads(Path(manifest).read_text())['hhvv']
    # Reject observed input before opening any capture or reference report.
    if case.get('dataset') != 'asimov_prefit':
        raise ValueError('Hessian comparison requires nominal pre-fit Asimov input')
    model, loader = load_native_fit_model(case['model'])
    benchmark_dataset(case, model, label='hhvv')
    info = model_parameter_info(model)
    assert all(sha(path) == checksum for path, checksum in case['hashes'].items())
    model_hashes = {n: sha(Path(case['model'])/n) for n in ('model.json', 'arrays.npz')}
    structure = json.loads((Path(case['model'])/'model.json').read_text())['structure_key']
    references, native_value, native_evidence = native_reference(native_gate, case, info, model_hashes)
    left, a, left_source = checked_capture(cpu, info, model_hashes, structure, cpu_revision)
    right, b, right_source = checked_capture(benchmark, info, model_hashes, structure, source_revision)
    assert left['backend'] == 'cpu'
    assert Path(cpu).resolve() != Path(benchmark).resolve()
    assert Path(left['arrays']).resolve() != Path(right['arrays']).resolve()
    errors = full_matrix_errors(a, b, info)
    nominal = dict(cpu=nominal_errors(a, references, native_value),
                   benchmark=nominal_errors(b, references, native_value))
    passed = errors['passed'] and all(r['passed'] for r in nominal.values())
    fixed = info.arrays['lower'] == info.arrays['upper']
    return dict(status='passed_full_matrix_and_nominal_native_gates' if passed else 'failed',
        script_sha256=sha(__file__), dataset='asimov_prefit',
        dataset_identity=model.metadata['provenance']['dataset']['identity'],
        workspace_sha256=sha(case['workspace']), model_sha256=model_hashes,
        artifacts={str(p): sha(p) for p in (manifest, cpu, benchmark, native_gate)},
        arrays_sha256=dict(cpu=left['arrays_sha256'], benchmark=right['arrays_sha256']),
        source_evidence=dict(cpu=left_source, benchmark=right_source), native_reference=native_evidence,
        native_model_loader=loader, nominal_native_comparison=nominal, full_free_matrix=errors,
        fixed_parameters=list(np.asarray(info.metadata['names'])[fixed]), exact_fixed_zeros=True,
        scaled_absolute_tolerance=TOLERANCE,
        scale_policy='Native positive error, otherwise 0.1*max(abs(imported initial),1); not fitted uncertainties.',
        runtimes={key: {k: r.get(k) for k in ('host', 'backend', 'jax_version', 'jaxlib_version',
            'devices', 'environment', 'timings', 'resources', 'jax_allocator')} for key, r in (('cpu', left), ('benchmark', right))},
        covariance_certified=False,
        scope='Every free nominal Hessian entry compared with an explicit Asimov CPU capture. Fixed rows/columns and gradients exactly zero. Both nominal NLL/gradients directly checked against reconstructed native references. No native Hessian/covariance certificate, fit acceptance or timing ratio.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'cpu', 'benchmark', 'native-gate', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--cpu-revision')
    parser.add_argument('--source-revision')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior audit evidence')
    result = compare(manifest=args.manifest, cpu=args.cpu, benchmark=args.benchmark,
        native_gate=args.native_gate, cpu_revision=args.cpu_revision, source_revision=args.source_revision)
    save_json(args.output, result)
    print(result['status'], result['full_free_matrix'])
    raise SystemExit(0 if result['status'] == 'passed_full_matrix_and_nominal_native_gates' else 1)


if __name__ == '__main__':
    main()
