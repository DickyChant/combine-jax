from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from measure_lcg_recovered_hessian import assemble_hessian, checked_catalog


def catalog_pair():
    records = [dict(index=i, name=f"term{i}", parameters=1, compiler_options={},
        kernel="hessian_columns", executable=f"build/new/{i}.bin") for i in range(524)]
    prep = dict(status="prepared", recovered_terms=records[:90], new_terms=records[90:],
        completed_terms=524, expected_terms=524, executable_directory="build/new",
        recovered_reference_checks=[dict(index=i, status="passed", max_scaled_error=0.)
                                    for i in (0, 2)])
    producer = dict(terms=524, compile_progress=dict(operation="hessian",
        term_timings=[{k: v for k, v in r.items() if k != "index"} for r in records[:90]]))
    return prep, producer


def test_complete_catalog_accepted():
    prep, producer = catalog_pair()
    assert len(checked_catalog(prep, producer)) == 524


@pytest.mark.parametrize("problem", [
    "running", "missing", "reordered", "duplicate", "failed_probe",
    "missing_probe", "binary_path", "compiler_options", "changed_original",
])
def test_catalog_rejects_invalid_provenance(problem):
    prep, producer = catalog_pair()
    prep = deepcopy(prep)
    if problem == "running":
        prep["status"] = "running"
    elif problem == "missing":
        prep["new_terms"].pop()
    elif problem == "reordered":
        prep["new_terms"][0], prep["new_terms"][1] = prep["new_terms"][1], prep["new_terms"][0]
    elif problem == "duplicate":
        prep["new_terms"][1] = prep["new_terms"][0]
    elif problem == "failed_probe":
        prep["recovered_reference_checks"][0]["status"] = "failed"
    elif problem == "missing_probe":
        prep["recovered_reference_checks"].pop()
    elif problem == "binary_path":
        prep["new_terms"][0]["executable"] = "build/other.bin"
    elif problem == "compiler_options":
        prep["new_terms"][0]["compiler_options"] = {"xla_disable_hlo_passes": "fusion"}
    elif problem == "changed_original":
        prep["recovered_terms"][0]["name"] = "changed"
    with pytest.raises(AssertionError):
        checked_catalog(prep, producer)


def test_global_assembly_overlapping_reordered_and_empty_terms():
    point = np.arange(7, dtype=np.float64)
    indices = [np.array([6, 2, 1, 3, 0]), np.array([], dtype=np.int64), np.array([3, 1])]
    matrices = [np.arange(25.).reshape(5, 5), np.empty((0, 0)), np.array([[2., 3.], [3., 5.]])]
    matrices[0] += matrices[0].T.copy()
    terms = [SimpleNamespace(indices=i) for i in indices]

    class Kernel:
        def __init__(self, matrix, expected_point):
            self.matrix, self.point = matrix, expected_point
            self.enters = self.exits = self.calls = 0

        def __enter__(self):
            self.enters += 1
            return self

        def __call__(self, local, directions):
            np.testing.assert_array_equal(local, self.point)
            self.calls += 1
            return directions @ self.matrix.T

        def __exit__(self, *args):
            self.exits += 1

    stores = [Kernel(m, point[i]) if len(i) else None for m, i in zip(matrices, indices)]
    actual = assemble_hessian(SimpleNamespace(terms=terms), stores, point)
    expected = np.zeros((7, 7))
    for i, matrix in zip(indices, matrices):
        expected[np.ix_(i, i)] += matrix
    np.testing.assert_array_equal(actual, expected)
    assert [(k.enters, k.exits, k.calls) for k in stores if k] == [(1, 1, 2), (1, 1, 1)]
    assert np.all(actual[4:6] == 0)
