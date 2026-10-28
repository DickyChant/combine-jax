"""Prevent cppyy multidimensional-view strides from corrupting native covariance."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from benchmark_combine_fit import copy_native_covariance


@pytest.mark.parametrize('size', [0,3,70])
def test_covariance_copy_matches_distinct_native_entries(root, size):
    matrix = root.TMatrixDSym(size)
    expected = np.empty((size,size))
    for i in range(size):
        for j in range(size):
            matrix[i][j] = expected[i,j] = 1./(1+i+j)
    copied = copy_native_covariance(matrix)
    np.testing.assert_array_equal(copied, expected)
    assert copied.dtype == np.float64
    if size:
        matrix[0][0] = -1.
        assert copied[0,0] == 1.
