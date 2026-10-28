"""The full-matrix comparison includes off-diagonal and fixed-coordinate entries."""

from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import compare_local_cpu_hessian as comparison
from compare_local_cpu_hessian import digest, load_arrays, matrix_errors, reassemble_local


def arrays():
    return dict(parameter_names=np.array(["free", "fixed"]), point=np.array([0., 1.]),
                parameter_scales=np.array([.1, 10.]), hessian=np.array([[2., .5], [.5, 1.]]))


def test_identical_complete_matrix():
    result = matrix_errors(arrays(), arrays())
    assert result["passed"] and result["compared_entries"] == 4
    assert result["differing_entries"] == result["maximum_scaled_absolute_error"] == 0


@pytest.mark.parametrize("coordinate", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_no_entry_is_excluded(coordinate):
    reference, candidate = arrays(), arrays()
    scales = reference["parameter_scales"]
    candidate["hessian"][coordinate] += 2e-6/(scales[coordinate[0]]*scales[coordinate[1]])
    result = matrix_errors(reference, candidate)
    assert not result["passed"] and result["differing_entries"] == 1
    assert result["worst_scaled_entry"]["indices"] == list(coordinate)


@pytest.mark.parametrize("key", ["parameter_names", "point", "parameter_scales"])
def test_different_order_point_or_scales_are_rejected(key):
    candidate = arrays()
    candidate[key] = candidate[key][::-1].copy()
    with pytest.raises(AssertionError):
        matrix_errors(arrays(), candidate)


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_nonfinite_matrix_rejected(value):
    candidate = arrays()
    candidate["hessian"][0, 0] = value
    with pytest.raises(AssertionError):
        matrix_errors(arrays(), candidate)


def test_blocks_reassemble_with_shared_indices_and_detect_corruption(tmp_path):
    reference = arrays()
    records = []
    for i, indices in enumerate((np.array([0, 1]), np.array([1]))):
        path = tmp_path/f"{i}.npz"
        matrix = reference["hessian"].copy() if i == 0 else np.array([[.25]])
        if i == 0:
            matrix[1, 1] -= .25
        np.savez_compressed(path, indices=indices, point=reference["point"][indices], hessian=matrix)
        records.append(dict(index=i, name=f"term_{i}", parameters=len(indices),
                            arrays=str(path), arrays_sha256=digest(path)))
    report = dict(terms=records, expected_terms=2)
    assert reassemble_local(report, reference)["reassembly_bitwise_equal"]
    incomplete = deepcopy(report)
    incomplete["terms"].pop()
    with pytest.raises(AssertionError):
        reassemble_local(incomplete, reference)
    wrong = deepcopy(reference)
    wrong["hessian"][1, 1] += .25
    with pytest.raises(AssertionError):
        reassemble_local(report, wrong)
    Path(records[0]["arrays"]).write_bytes(b"changed")
    with pytest.raises(AssertionError, match="block changed"):
        reassemble_local(report, reference)


def test_artifact_shape_precision_and_hash_are_verified(tmp_path):
    path = tmp_path/"full.npz"
    reference = arrays()
    np.savez_compressed(path, **reference)
    report = dict(arrays=str(path), arrays_sha256=digest(path), parameters=2)
    loaded = load_arrays(report)
    assert matrix_errors(reference, loaded)["passed"]
    reference["hessian"] = reference["hessian"].astype(np.float32)
    np.savez_compressed(path, **reference)
    with pytest.raises(AssertionError, match="artifact changed"):
        load_arrays(report)
    report["arrays_sha256"] = digest(path)
    with pytest.raises(AssertionError, match="hessian"):
        load_arrays(report)


def test_historical_source_requires_explicit_revision_and_exact_hash(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("builder.py")
    path.write_bytes(b"current")
    old = b"original"
    checksum = hashlib.sha256(old).hexdigest()
    revision = "a"*40
    with pytest.raises(ValueError, match="explicit historical revision"):
        comparison.source_evidence(path, checksum)
    with pytest.raises(ValueError, match="full historical"):
        comparison.source_evidence(path, checksum, "HEAD")

    def git_show(command):
        assert command == ["git", "show", revision+":builder.py"]
        return old

    monkeypatch.setattr(comparison.subprocess, "check_output", git_show)
    result = comparison.source_evidence(path, checksum, revision)
    assert result["verified_from"] == "git_blob" and result["revision"] == revision
    assert result["current_sha256"] != result["reported_sha256"]
    with pytest.raises(AssertionError, match="Historical source hash"):
        comparison.source_evidence(path, "0"*64, revision)
