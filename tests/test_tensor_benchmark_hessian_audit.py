"""Guard the saved-Hessian audit against coordinate and excluded-row mistakes."""
from pathlib import Path
import sys
import hashlib

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_tensor_benchmark_hessian import compare_arrays, verify_point_source


@pytest.fixture
def matrices():
    reference = dict(parameter_names=np.array(['a','fixed','b']), point=np.array([1.,2.,3.]),
                     parameter_scales=np.array([.1,1.,10.]), hessian=np.diag([2.,5.,3.]))
    actual = dict(parameter_names=reference['parameter_names'].copy(),
                  points=reference['point'][None, :].copy(), hessians=np.array([np.diag([2.,0.,3.])]))
    return reference, actual


def test_complete_free_matrix_is_checked_with_reference_scales(matrices):
    reference, actual = matrices
    assert compare_arrays(reference, actual, ['fixed'])['passed']
    actual['hessians'][0,2,0] += 2e-6
    result = compare_arrays(reference, actual, ['fixed'])
    assert not result['passed'] and result['compared_entries'] == 4
    assert result['worst_scaled_entry']['parameters'] == ['b','a']


@pytest.mark.parametrize('corruption', ['names','point','fixed_row','fixed_column','nonfinite','float32'])
def test_corrupt_capture_is_rejected(matrices, corruption):
    reference, actual = matrices
    if corruption == 'names':
        actual['parameter_names'][[0,2]] = actual['parameter_names'][[2,0]]
    elif corruption == 'point':
        actual['points'][0,0] += 1e-4
    elif corruption == 'fixed_row':
        actual['hessians'][0,1,0] = 1e-15
    elif corruption == 'fixed_column':
        actual['hessians'][0,0,1] = 1e-15
    elif corruption == 'nonfinite':
        actual['hessians'][0,0,0] = np.nan
    else:
        actual['hessians'] = actual['hessians'].astype(np.float32)
    with pytest.raises(AssertionError):
        compare_arrays(reference, actual, ['fixed'])


def test_imported_nominal_path_still_rejects_a_different_point(matrices, tmp_path):
    report = dict(points_source='model initial', points_source_sha256=None)
    assert verify_point_source(report, tmp_path/'unused.npz').startswith('imported nominal')
    reference, actual = matrices
    assert compare_arrays(reference, actual, ['fixed'])['passed']
    actual['points'][0, 0] += 1e-4
    with pytest.raises(AssertionError):
        compare_arrays(reference, actual, ['fixed'])


def test_explicit_point_file_requires_its_frozen_hash(tmp_path):
    path = tmp_path/'points.npz'
    path.write_bytes(b'frozen point-file fixture')
    report = dict(points_source=str(path), points_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert verify_point_source(report, path).startswith('frozen curvature point file')
    path.write_bytes(b'different points')
    with pytest.raises(AssertionError):
        verify_point_source(report, path)


def test_missing_point_file_hash_is_not_an_imported_nominal():
    with pytest.raises(AssertionError):
        verify_point_source(dict(points_source='points.npz', points_source_sha256=None), Path('points.npz'))
