"""Audit saved benchmark outputs against frozen native and old-AD references."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tensor_benchmark_common import save_json, sha


def available_reference_status(rows):
    if not all(row['passed'] for row in rows):
        return 'failed'
    if any(row.get('native_gradient_validation') == 'pending' for row in rows):
        return 'passed_available_gates_native_gradients_pending'
    return 'passed'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('build/tensor_benchmark_inputs/manifest.json'))
    parser.add_argument('--case', required=True)
    parser.add_argument('--reports', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    case = json.loads(args.manifest.read_text())[args.case]
    from tensor_benchmark_dataset import benchmark_dataset
    metadata = json.loads((Path(case['model'])/'model.json').read_text())['metadata']
    dataset_name = benchmark_dataset(case, SimpleNamespace(metadata=metadata), label=args.case)
    for path, digest in case['hashes'].items():
        assert sha(path) == digest, path
    with np.load(case['reference'], allow_pickle=False) as archive:
        ref = {k: archive[k] for k in archive.files}
    report = dict(status='checking', script_sha256=sha(__file__), case=args.case, dataset=dataset_name,
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        reference=str(case['reference']), reference_sha256=sha(case['reference']), cases=[])
    for path in args.reports:
        measured = json.loads(path.read_text())
        assert measured['status'] == 'measured_pending_validation'
        assert measured['float64'] and not measured['root_imported']
        assert sha(measured['arrays']) == measured['arrays_sha256']
        assert measured['model_sha256'] == {n: sha(Path(case['model'])/n) for n in ('model.json','arrays.npz')}
        with np.load(measured['arrays'], allow_pickle=False) as archive:
            actual = {k: archive[k] for k in archive.files}
        np.testing.assert_array_equal(actual['parameter_names'], ref['parameter_names'])
        row = dict(report=str(path), report_sha256=sha(path), backend=measured['backend'],
            mode=measured['mode'], operation=measured['operation'], gates={})
        report['cases'].append(row)
        if measured['operation'] == 'fit':
            assert args.case != 'hhvv', 'HHVV has no accepted covariance/fit reference'
            native_path = Path('reports/tensor_device_fits.json') if args.case in ('counting','automcstats') else Path('reports/tensor_analytic_roundoff_fits.json')
            prior = json.loads(native_path.read_text())
            old = next(c for c in prior['cases'] if c.get('kind', c.get('label')) == args.case)
            if 'native' in old:
                native = old['native']
                names = [p['name'] for p in old['parameters']]
            else:
                native = {k: old['native_'+k] for k in ('values','errors','nll','covariance')}
                names = old['names']
            np.testing.assert_array_equal(names, actual['parameter_names'])
            nv = np.array(native['values']); ne = np.array(native['errors'])
            row.update(native_fit_reference=str(native_path), native_fit_reference_sha256=sha(native_path))
            row['projected_gradient_max'] = float(np.max(np.abs(actual['fit_projected_gradient']), initial=0.))
            row['gates']['convergence'] = bool(actual['fit_status'] == 0 and row['projected_gradient_max'] <= 1e-7)
            row['gates']['covariance_status'] = bool(actual['fit_covariance_status'] == 0)
            error = abs(float(actual['fit_value'])-native['nll'])
            row['nll_absolute_error'] = error
            row['gates']['nll'] = error < 1e-6
            parameter_ratio = np.abs(actual['fit_values']-nv)/np.maximum(np.maximum(.01*np.abs(nv), .01*ne), 1e-30)
            row['max_parameter_error_over_tolerance'] = float(np.max(parameter_ratio))
            row['gates']['parameters'] = bool(np.all(parameter_ratio < 1.))
            error_ratio = np.abs(actual['fit_reported_errors']-ne)/np.maximum(ne, 1e-30)
            row['max_reported_error_relative_difference'] = float(np.max(error_ratio))
            row['gates']['reported_errors'] = bool(np.all(actual['fit_reported_error_status'] == 0) and np.all(error_ratio < .01))
            covariance = actual['fit_covariance']
            inverse_error = float(np.max(np.abs(actual['fit_hessian']@covariance-np.eye(len(nv)))))
            row['raw_inverse_error'] = inverse_error
            row['gates']['raw_covariance'] = bool(inverse_error < 1e-6 and np.all(np.linalg.eigvalsh(covariance) > 0))
            row['gates']['native_covariance'] = bool(np.allclose(covariance, native['covariance'], rtol=.01, atol=1e-5))
        else:
            indices = []
            for point in actual['points']:
                match = np.flatnonzero(np.all(ref['points'] == point, axis=1))
                assert len(match) == 1
                indices.append(int(match[0]))
            error = np.abs(actual['values']-ref['values'][indices])
            row['points'] = len(indices)
            row['max_nll_absolute_error'] = float(np.max(error))
            row['gates']['nll'] = bool(np.isfinite(actual['values']).all() and np.all(error < 1e-6))
            if 'gradients' in actual:
                if args.case == 'hhvv' and case.get('reference_validation', {}).get('native_gradients') == 'pending':
                    row['native_gradient_validation'] = 'pending'
                    row['gradient_scope'] = 'Captured for the separate native FD audit; available references cannot certify native gradients.'
                for key in ('gradients', 'native_fd_gradients'):
                    if key not in ref:
                        continue
                    expected = ref[key][indices]
                    difference = np.abs(actual['gradients']-expected)
                    relative = difference/np.maximum(np.abs(expected), 1e-8)
                    row[key+'_max_absolute_error'] = float(np.max(difference))
                    row[key+'_max_relative_error'] = float(np.max(relative))
                    row['gates'][key] = bool(np.isfinite(actual['gradients']).all() and np.all(relative < 1e-4))
            if 'hessians' in actual:
                assert 'hessians' in ref, 'HHVV full Hessian requires the complete saved-matrix audit'
                expected = ref['hessians'][indices]
                difference = np.abs(actual['hessians']-expected)
                row['hessian_max_absolute_error'] = float(np.max(difference))
                row['gates']['hessian'] = bool(np.isfinite(actual['hessians']).all() and np.allclose(actual['hessians'], expected, rtol=1e-8, atol=1e-8))
        row['passed'] = all(row['gates'].values())
        save_json(args.output, report)
    report['status'] = available_reference_status(report['cases'])
    save_json(args.output, report)
    print(report['status'], [(r['operation'], r['passed']) for r in report['cases']], flush=True)
    assert report['status'] != 'failed'


if __name__ == '__main__':
    main()
