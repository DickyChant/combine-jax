"""Reject incomplete or mismatched curvature evidence, without running a fit."""
import copy
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_hessian import compare, checked_capture, full_matrix_errors, nominal_errors
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info
from tensor_benchmark_common import provenance, save_json, sha


@pytest.fixture
def evidence(tmp_path):
    """Small synthetic artifacts exercise the complete file-validation path."""
    names = ['a', 'fixed', 'b']; initial = np.array([1., 2., 3.])
    metadata = dict(schema='combine-jax.tensor.v1', kind='model', schedule=[],
        parameter_names=names, components={}, parameter_info=dict(schema='combine-jax.tensor.v1',
            kind='parameter_info', schedule=[], names=names, roles=['nuisance']*3))
    arrays = dict(parameter_info__initial=initial, parameter_info__lower=np.array([-5., 2., -5.]),
        parameter_info__upper=np.array([5., 2., 5.]), parameter_info__error=np.array([.1, 1., 10.]),
        scalars__parameter_initial=initial, scalars__parameter_fixed=np.array([False, True, False]),
        scalars__parameter_indices=np.array([0, 2]), scalars__parameter_slots=np.array([0, 2]),
        scalars__initial_values=initial)
    workspace = tmp_path/'workspace.root'; workspace.write_bytes(b'synthetic workspace identity fixture')
    generation = tmp_path/'generation.json'
    identity = {'sha256': 'synthetic dataset identity'}
    save_json(generation, dict(status='generated_and_identity_verified', generation_kind='combine_prefit_asimov',
        dataset=dict(identity=identity), workspace_sha256=sha(workspace)))
    metadata['provenance'] = dict(dataset=dict(name='asimov_prefit', identity=identity),
        dataset_generation=dict(kind='combine_prefit_asimov', MH=125., dataset_identity=identity,
            generation_report_sha256=sha(generation)))
    model = TensorBundle(metadata, arrays); model.save(tmp_path/'model')
    info = model_parameter_info(model)
    model_hashes = {n: sha(tmp_path/'model'/n) for n in ('model.json', 'arrays.npz')}
    points = np.repeat(initial[None, :], 26, axis=0); points[:, 0] += np.arange(26)*.01
    points_path = tmp_path/'points.npz'; np.savez(points_path, parameter_names=names, points=points)
    reference = tmp_path/'reference.npz'
    np.savez(reference, parameter_names=names, points=points, values=np.repeat(10., 26))
    case = dict(model=str(tmp_path/'model'), workspace=str(workspace), dataset='asimov_prefit',
        generation_report=str(generation), points=str(points_path), reference=str(reference), hashes={})
    manifest = tmp_path/'manifest.json'; save_json(manifest, dict(hhvv=case))
    cases = [dict(index=j, name=name, coordinate=initial[j], status='fixed_bounds' if j == 1 else 'converged',
        selected_first_probe_index=0, probes=[] if j == 1 else [dict(status='valid', derivative=2.)]*3)
        for j, name in enumerate(names)]
    raw = tmp_path/'native.json.gz'
    raw.write_bytes(gzip.compress(json.dumps(dict(index=0, cases=cases, canonical_nll=10.,
        zero_point_nll=0., restoration_error=0.)).encode()))
    native = tmp_path/'native.json'
    save_json(native, dict(dataset='asimov_prefit', workspace_sha256=sha(workspace),
        generation_report_sha256=sha(generation), root_version='6.40.00', combine_version='11.0.0',
        script_sha256=sha('scripts/capture_asimov_native_reference.py'), fd_script_sha256=sha('scripts/native_fd_scan.py'),
        nll_arrays=str(reference), nll_arrays_sha256=sha(reference), points=[dict(index=0,
            artifact=str(raw), artifact_sha256=sha(raw), restoration_error=0.)]))
    gate = tmp_path/'gate.json'
    save_json(gate, dict(status='passed', dataset='asimov_prefit',
        script_sha256=sha('scripts/audit_asimov_native_reference.py'), workspace_sha256=sha(workspace),
        generation_report_sha256=sha(generation), model_sha256=model_hashes, nll_passed=True,
        max_nll_absolute_error=0., artifacts={str(native): sha(native)},
        status_counts=dict(passed=52, fixed_bounds=26), refinements_applied=[],
        points=[dict(index=i, status_counts=dict(passed=2, fixed_bounds=1), unresolved_or_failed=[],
            max_relative_error_on_certified_references=0., native_reference_report=str(native)) for i in range(26)]))
    captures = []
    for backend in ('cpu', 'gpu'):
        numeric = tmp_path/(backend+'.npz')
        np.savez(numeric, parameter_names=names, points=initial[None, :], values=np.array([10.]),
            gradients=np.array([[2., 0., 2.]]), hessians=np.array([np.diag([4., 0., 9.])]))
        path = tmp_path/(backend+'.json')
        save_json(path, dict(provenance(), status='measured_pending_validation', stage='complete',
            operation='hessian', float64=True, root_imported=False, backend=backend, mode='jit',
            model_sha256=model_hashes, structure_key=model.structure_key,
            script_sha256=sha('scripts/benchmark_tensor.py'), points_source='model initial',
            points_source_sha256=None, arrays=str(numeric), arrays_sha256=sha(numeric), parameters=3, points=1))
        captures.append(path)
    return dict(manifest=manifest, cpu=captures[0], benchmark=captures[1], native_gate=gate), info, model_hashes, model.structure_key


def test_complete_comparison_has_separate_native_and_matrix_gates(evidence):
    args, _, _, _ = evidence
    result = compare(**args)
    assert result['status'] == 'passed_full_matrix_and_nominal_native_gates'
    assert result['full_free_matrix']['compared_entries'] == 4
    assert result['exact_fixed_zeros'] and result['fixed_parameters'] == ['fixed']
    assert all(r['passed'] for r in result['nominal_native_comparison'].values())
    assert not result['covariance_certified']


@pytest.mark.parametrize('fault', ['point', 'name', 'subset', 'float32', 'nan', 'fixed_row', 'fixed_column', 'fixed_gradient'])
def test_capture_coordinate_dtype_shape_and_fixed_zeros(evidence, fault):
    args, info, hashes, structure = evidence
    report = json.loads(args['benchmark'].read_text())
    with np.load(report['arrays']) as saved:
        data = {k: saved[k] for k in saved.files}
    if fault == 'point': data['points'][0, 0] += .01
    elif fault == 'name': data['parameter_names'][[0, 2]] = data['parameter_names'][[2, 0]]
    elif fault == 'subset': data['hessians'] = data['hessians'][:, :2, :2]
    elif fault == 'float32': data['hessians'] = data['hessians'].astype(np.float32)
    elif fault == 'nan': data['hessians'][0, 0, 0] = np.nan
    elif fault == 'fixed_row': data['hessians'][0, 1, 0] = 1e-20
    elif fault == 'fixed_column': data['hessians'][0, 0, 1] = 1e-20
    else: data['gradients'][0, 1] = 1e-20
    np.savez(report['arrays'], **data)
    report['arrays_sha256'] = sha(report['arrays']); save_json(args['benchmark'], report)
    with pytest.raises(AssertionError):
        checked_capture(args['benchmark'], info, hashes, structure)


@pytest.mark.parametrize('fault', ['operation', 'stage', 'float64', 'root', 'model', 'structure', 'source', 'point_source', 'hash'])
def test_capture_provenance_rejected(evidence, fault):
    args, info, hashes, structure = evidence
    path = args['benchmark']; report = json.loads(path.read_text())
    key, value = {'operation': ('operation', 'fit'), 'stage': ('stage', 'capture'),
        'float64': ('float64', False), 'root': ('root_imported', True), 'model': ('model_sha256', {}),
        'structure': ('structure_key', 'other'), 'source': ('script_sha256', 'other'),
        'point_source': ('points_source', 'fitted point'), 'hash': ('arrays_sha256', 'other')}[fault]
    report[key] = value; save_json(path, report)
    with pytest.raises(AssertionError): checked_capture(path, info, hashes, structure)


@pytest.mark.parametrize('fault', ['partial', 'duplicate_point', 'gradient_error', 'workspace', 'artifact'])
def test_native_gate_must_be_complete_and_match_inputs(evidence, fault):
    args, _, _, _ = evidence; path = args['native_gate']; gate = json.loads(path.read_text())
    if fault == 'partial': gate['status_counts']['passed'] -= 1
    elif fault == 'duplicate_point': gate['points'][-1]['index'] = 24
    elif fault == 'gradient_error': gate['points'][0]['max_relative_error_on_certified_references'] = 1e-4
    elif fault == 'workspace': gate['workspace_sha256'] = 'other'
    else: gate['artifacts'][next(iter(gate['artifacts']))] = 'other'
    save_json(path, gate)
    with pytest.raises(AssertionError): compare(**args)


def test_observed_manifest_rejected_before_capture_files_are_opened(tmp_path):
    path = tmp_path/'manifest.json'; save_json(path, dict(hhvv=dict(dataset='data_obs')))
    with pytest.raises(ValueError, match='pre-fit Asimov'):
        compare(manifest=path, cpu='unread', benchmark='unread', native_gate='unread')


def test_one_bad_off_diagonal_entry_fails_full_matrix(evidence):
    args, info, hashes, structure = evidence
    _, cpu, _ = checked_capture(args['cpu'], info, hashes, structure)
    candidate = copy.deepcopy(cpu); candidate['hessians'][0, 2, 0] = 1.1e-6
    result = full_matrix_errors(cpu, candidate, info)
    assert not result['passed'] and result['compared_entries'] == 4
    assert result['worst_scaled_entry']['parameters'] == ['b', 'a']


def test_native_nominal_gate_checks_all_six_selected_estimates():
    reference = dict(status='converged', selected_first_probe_index=0, reference_probe_count=6,
        probes=[dict(status='valid', derivative=value) for value in [1.]*5+[1.00009999]])
    actual = dict(values=np.array([10.]), gradients=np.array([[.99995]]))
    result = nominal_errors(actual, [reference], 10.)
    assert not result['passed'] and result['failed_coordinates'] == [0]


@pytest.mark.parametrize('field,increment', [('values', 1.01e-6), ('gradients', .001)])
def test_matrix_equality_cannot_mask_failed_native_values(evidence, field, increment):
    args, _, _, _ = evidence; report = json.loads(args['benchmark'].read_text())
    with np.load(report['arrays']) as saved: data = {k: saved[k] for k in saved.files}
    data[field].flat[0] += increment
    np.savez(report['arrays'], **data); report['arrays_sha256'] = sha(report['arrays'])
    save_json(args['benchmark'], report)
    result = compare(**args)
    assert result['status'] == 'failed' and result['full_free_matrix']['passed']
    assert not result['nominal_native_comparison']['benchmark']['passed']
