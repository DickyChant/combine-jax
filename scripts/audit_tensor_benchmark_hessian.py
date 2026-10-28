"""Compare portable HHVV Hessians with frozen full matrices and native projections.

This consumes saved arrays only. It does not evaluate a model, refit, change the
reference selection, or interpret unresolved native directions as agreement.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_gpu_cached_curvature import compare as native_compare
from compare_local_cpu_hessian import load_arrays, matrix_errors
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def verify_point_source(measured, curvature_points):
    """Both supported inputs still require exact array identity to the oracle."""
    digest = measured['points_source_sha256']
    if digest is None:
        assert measured['points_source'] == 'model initial'
        return 'imported nominal; exact saved point checked against the frozen reference'
    assert digest == sha(curvature_points)
    return 'frozen curvature point file; exact saved point checked against the frozen reference'


def compare_arrays(reference, actual, fixed_names):
    """Check coordinate identity, every free entry, and exact fixed-coordinate zeros."""
    names = reference['parameter_names']
    size = len(names)
    np.testing.assert_array_equal(actual['parameter_names'], names)
    np.testing.assert_array_equal(actual['points'], reference['point'][None, :])
    hessian = actual['hessians']
    assert hessian.shape == (1, size, size) and hessian.dtype == np.float64
    assert np.isfinite(hessian).all()
    fixed = np.isin(names, fixed_names)
    assert len(set(fixed_names)) == len(fixed_names) == int(fixed.sum())
    np.testing.assert_array_equal(hessian[0, fixed, :], 0.)
    np.testing.assert_array_equal(hessian[0, :, fixed], 0.)
    free = ~fixed
    base = {k: reference[k][free] for k in ('parameter_names', 'point', 'parameter_scales')}
    return matrix_errors(dict(base, hessian=reference['hessian'][np.ix_(free, free)]),
                         dict(base, hessian=hessian[0][np.ix_(free, free)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision',
                        help='Explicit committed producer revision for a historical capture')
    args = parser.parse_args()
    assert not args.output.exists()
    adapter = args.output.with_name(args.output.stem+'_native_input.json')
    adapted_arrays = Path('build')/(args.output.stem+'_native_input.npz')
    native_output = args.output.with_name(args.output.stem+'_native_projections.json')
    assert not any(p.exists() for p in (adapter, adapted_arrays, native_output))
    measured = json.loads(args.benchmark.read_text())
    assert measured['status'] == 'measured_pending_validation' and measured['stage'] == 'complete'
    assert measured['operation'] == 'hessian' and measured['float64'] and not measured['root_imported']
    assert sha(measured['arrays']) == measured['arrays_sha256']
    source_evidence = verified_source_record(measured, 'scripts/benchmark_tensor.py', args.source_revision)
    manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
    case = json.loads(manifest_path.read_text())['hhvv']
    assert all(sha(p) == digest for p, digest in case['hashes'].items())
    assert all(sha(p) == digest for p, digest in case['reference_provenance'].items())
    assert measured['model_sha256'] == {n: sha(Path(case['model'])/n) for n in ('model.json', 'arrays.npz')}
    point_source = verify_point_source(measured, case['curvature_points'])
    accepted_path = Path('reports/tensor_hhvv_hessian_audit.json')
    accepted = json.loads(accepted_path.read_text())
    assert accepted['status'] == 'passed_old_ad_with_native_reference_uncertainty'
    assert all(sha(p) == digest for p, digest in accepted['checker_sources'].items())
    assert sha(accepted['producer']) == accepted['producer_sha256']
    producer = json.loads(Path(accepted['producer']).read_text())
    assert sha(producer['arrays']) == producer['arrays_sha256'] == accepted['arrays_sha256']
    assert sha(producer['reference_report']) == producer['reference_report_sha256']
    old_report = json.loads(Path(producer['reference_report']).read_text())
    assert old_report['arrays_sha256'] == producer['reference_arrays_sha256']
    oracle = load_arrays(old_report)
    fixed = accepted['excluded_fixed_parameters']
    assert len(fixed) == 4 and len(oracle['point']) == 2656
    # The fit-metadata import preserved every likelihood array; retain that
    # provenance while comparing the changed runtime directly to both matrices.
    import_audit_path = Path('reports/tensor_parameter_import_audit.json')
    import_audit = json.loads(import_audit_path.read_text())
    assert import_audit['status'] == 'passed'
    assert sha(import_audit['import_report']) == import_audit['import_report_sha256']
    assert import_audit['import_report'] in case['reference_provenance']
    with np.load(measured['arrays'], allow_pickle=False) as a:
        actual = {k: a[k] for k in a.files}
    with np.load(producer['arrays'], allow_pickle=False) as a:
        cpu = {k: a[k] for k in a.files}
    for key in ('parameter_names', 'point', 'parameter_scales'):
        np.testing.assert_array_equal(cpu[key], oracle[key])
    old_errors = compare_arrays(oracle, actual, fixed)
    cpu_errors = compare_arrays(cpu, actual, fixed)
    with np.load(case['reference'], allow_pickle=False) as ref:
        np.testing.assert_array_equal(actual['parameter_names'], ref['parameter_names'])
        np.testing.assert_array_equal(actual['points'], ref['points'][:1])
        assert actual['values'].shape == (1,) and actual['values'].dtype == np.float64
        assert actual['gradients'].shape == (1, 2656) and actual['gradients'].dtype == np.float64
        assert np.isfinite(actual['values']).all() and np.isfinite(actual['gradients']).all()
        nll_error = float(np.max(np.abs(actual['values']-ref['values'][:1])))
        grad_difference = np.abs(actual['gradients']-ref['gradients'][:1])
        grad_relative = float(np.max(grad_difference/np.maximum(np.abs(ref['gradients'][:1]), 1e-8)))
        np.testing.assert_array_equal(actual['gradients'][:, np.isin(actual['parameter_names'], fixed)], 0.)
    np.savez_compressed(adapted_arrays, parameter_names=actual['parameter_names'],
                        point=actual['points'][0], hessian=actual['hessians'][0])
    save_json(adapter, dict(source=producer['source'], arrays=str(adapted_arrays),
        arrays_sha256=sha(adapted_arrays), completed_operations=['hessian'],
        producer=str(args.benchmark), producer_sha256=sha(args.benchmark),
        scope='Lossless array adapter for the unchanged native curvature checker.'))
    assert sha(accepted['native_projections']) == accepted['native_projections_sha256']
    prior_native = json.loads(Path(accepted['native_projections']).read_text())
    ref_path = Path(prior_native['reference_audit']['reference'])
    assert sha(ref_path) == prior_native['reference_audit']['reference_sha256']
    native = native_compare(adapter, ref_path)
    save_json(native_output, native)
    assert native['reference_audit'] == prior_native['reference_audit']
    assert native['unresolved_directions'] == prior_native['unresolved_directions']
    passed = old_errors['passed'] and cpu_errors['passed'] and nll_error < 1e-6 and grad_relative < 1e-4 and not native['failed_directions']
    report = dict(status='passed_old_ad_with_native_reference_uncertainty' if passed else 'failed',
        script_sha256=sha(__file__), benchmark=str(args.benchmark), benchmark_sha256=sha(args.benchmark),
        source_evidence=source_evidence,
        arrays_sha256=measured['arrays_sha256'], accepted_reference=str(accepted_path),
        point_source_verification=point_source,
        accepted_reference_sha256=sha(accepted_path), import_audit=str(import_audit_path),
        import_audit_sha256=sha(import_audit_path), manifest_sha256=sha(manifest_path),
        old_ad_free_hessian_comparison=old_errors, cpu_tensor_free_hessian_comparison=cpu_errors,
        exact_zero_fixed_rows_columns=fixed, nll_absolute_error=nll_error,
        gradient_max_absolute_error=float(grad_difference.max()), gradient_max_relative_error=grad_relative,
        native_projections=str(native_output), native_projections_sha256=sha(native_output),
        native_status=native['status'], native_failed_directions=native['failed_directions'],
        native_unresolved_directions=native['unresolved_directions'],
        timings=measured['timings'], resources=measured['resources'], jax_allocator=measured.get('jax_allocator'),
        runtime={k: measured[k] for k in ('host','backend','mode','jax_version','jaxlib_version','devices','environment')},
        scope='All free nominal Hessian entries compared with the full old CPU AD matrix and accepted tensor CPU matrix. Fixed rows/columns are exactly zero. Native projections keep their unresolved references; no fit or covariance acceptance.')
    save_json(args.output, report)
    print(report['status'], old_errors, native['status'], flush=True)
    assert passed


if __name__ == '__main__':
    main()
