from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_local_cpu_hessian import digest
from compare_lcg_recovered_hessian import verify_saved_repeats
from measure_lcg_recovered_hessian import compare_hessian_repeat


def arrays():
    return dict(hessian=np.array([[2., .005], [.005, 3.]]),
        point=np.array([0., 1.]), parameter_names=np.array(["free", "fixed"]),
        parameter_scales=np.array([1., 2.]))


def test_roundoff_is_recorded_without_requiring_bitwise_equality():
    reference = arrays()
    candidate = deepcopy(reference)
    candidate["hessian"][1, 0] += 3e-14
    check = compare_hessian_repeat(reference, candidate, reference["hessian"])
    assert check["passed"] and not check["bitwise_equal_to_first"]
    assert check["cpu_reference"]["compared_entries"] == 4
    assert 0 < check["cpu_reference"]["maximum_scaled_absolute_error"] < 1e-6


@pytest.mark.parametrize("target", ["free", "fixed", "repeat"])
def test_errors_at_or_above_the_unchanged_tolerance_cannot_pass(target):
    reference = arrays()
    candidate = deepcopy(reference)
    first = reference["hessian"].copy()
    if target == "free":
        candidate["hessian"][0, 0] += 2e-6
    elif target == "fixed":
        candidate["hessian"][1, 1] += 1e-6
    else:
        first[0, 1] += 2e-6
    assert not compare_hessian_repeat(reference, candidate, first)["passed"]


@pytest.mark.parametrize("problem", ["nan", "inf", "float32", "shape", "point", "order", "scales"])
def test_invalid_arrays_or_metadata_are_rejected(problem):
    reference = arrays()
    candidate = deepcopy(reference)
    if problem == "nan":
        candidate["hessian"][0, 0] = np.nan
    elif problem == "inf":
        candidate["hessian"][1, 0] = np.inf
    elif problem == "float32":
        candidate["hessian"] = candidate["hessian"].astype(np.float32)
    elif problem == "shape":
        candidate["hessian"] = candidate["hessian"][:1]
    elif problem == "point":
        candidate["point"][0] = 1.
    elif problem == "order":
        candidate["parameter_names"] = candidate["parameter_names"][::-1]
    else:
        candidate["parameter_scales"][1] = 1.
    with pytest.raises(AssertionError):
        compare_hessian_repeat(reference, candidate, reference["hessian"])


def saved_benchmark(tmp_path):
    reference = arrays()
    first = deepcopy(reference)
    second = deepcopy(reference)
    second["hessian"][0, 1] += 3e-14
    calls = []
    for index, candidate in enumerate([first, second]):
        path = tmp_path/f"call{index}.npz"
        np.savez_compressed(path, **candidate)
        result = compare_hessian_repeat(reference, candidate, first["hessian"])
        calls.append(dict(index=index, arrays=str(path), arrays_sha256=digest(path),
            validation=result, bitwise_equal_to_first=result["bitwise_equal_to_first"],
            artifact_and_validation_seconds=.1))
    report = dict(parameters=2, arrays=calls[0]["arrays"], arrays_sha256=calls[0]["arrays_sha256"],
        calls=calls, repeat_validation=dict(policy="every_call_cpu_reference_and_first_matrix",
            scaled_absolute_tolerance=1e-6, bitwise_equality_required=False,
            checker_sha256=digest("scripts/compare_local_cpu_hessian.py")))
    return report, reference, first


def test_each_saved_matrix_is_independently_rechecked(tmp_path):
    report, reference, first = saved_benchmark(tmp_path)
    result = verify_saved_repeats(report, reference, first)
    assert len(result) == 2 and all(c["passed"] for c in result)
    assert not result[1]["bitwise_equal_to_first"]


@pytest.mark.parametrize("problem", [
    "missing", "checksum", "relabelled", "forged_error", "forged_bitwise",
    "changed_matrix", "relaxed_tolerance", "failed_comparison",
])
def test_saved_repeat_evidence_cannot_be_forged(tmp_path, problem):
    report, reference, first = saved_benchmark(tmp_path)
    call = report["calls"][1]
    if problem == "missing":
        Path(call["arrays"]).unlink()
    elif problem == "checksum":
        call["arrays_sha256"] = "0"*64
    elif problem == "relabelled":
        call["arrays"] = report["calls"][0]["arrays"]
    elif problem == "forged_error":
        call["validation"]["cpu_reference"]["maximum_absolute_error"] = 0.
    elif problem == "forged_bitwise":
        call["bitwise_equal_to_first"] = True
    elif problem in ("changed_matrix", "failed_comparison"):
        changed = deepcopy(reference)
        changed["hessian"][1, 1] += 2e-6
        np.savez_compressed(call["arrays"], **changed)
        call["arrays_sha256"] = digest(call["arrays"])
        if problem == "failed_comparison":
            call["validation"] = compare_hessian_repeat(reference, changed, first["hessian"])
    else:
        report["repeat_validation"]["scaled_absolute_tolerance"] = 1e-5
    with pytest.raises((AssertionError, FileNotFoundError)):
        verify_saved_repeats(report, reference, first)
