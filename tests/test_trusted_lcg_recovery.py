import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from trusted_lcg_recovery import all_columns, checked_path, checked_record, records


def fixture_record(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("build/gpu_executables/counting_gpu_scoped/hessian/0.bin")
    path.parent.mkdir(parents=True)
    path.write_bytes(b"own-test-binary")
    record = dict(name="channel", parameters=3, kernel="hessian_columns",
        compiler_options={}, executable=str(path), serialized_bytes=path.stat().st_size,
        executable_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    term = SimpleNamespace(name="channel", indices=np.arange(3))
    producer = dict(model="counting", report_tag="scoped")
    return record, term, producer


@pytest.mark.parametrize("change", [
    {"name": "wrong"}, {"parameters": 2}, {"kernel": "hessian"},
    {"compiler_options": {"xla_disable_hlo_passes": "fusion"}},
    {"serialized_bytes": 1}, {"executable_sha256": "0"*64},
])
def test_reject_mismatched_term_and_binary(tmp_path, monkeypatch, change):
    record, term, producer = fixture_record(tmp_path, monkeypatch)
    checked_record(record, term, 0, producer)
    with pytest.raises(AssertionError):
        checked_record(dict(record, **change), term, 0, producer)


def test_binary_mutation_is_detected(tmp_path, monkeypatch):
    record, term, producer = fixture_record(tmp_path, monkeypatch)
    Path(record["executable"]).write_bytes(b"changed-binary")
    with pytest.raises(AssertionError):
        checked_record(record, term, 0, producer)


def test_wrong_term_index_rejected(tmp_path, monkeypatch):
    record, term, producer = fixture_record(tmp_path, monkeypatch)
    with pytest.raises(AssertionError):
        checked_record(record, term, 1, producer)


def test_artifact_symlink_and_escape_rejected(tmp_path, monkeypatch):
    record, term, producer = fixture_record(tmp_path, monkeypatch)
    path = Path(record["executable"])
    Path("alias").symlink_to(path)
    for candidate in ("alias", "../escape", str(path.resolve())):
        with pytest.raises(ValueError):
            checked_path(candidate)


def test_only_fully_recorded_terms_returned():
    partial = dict(compile_progress=dict(operation="hessian", completed_terms=1,
                                        term_timings=[{"name": "a"}]))
    assert records(partial) == [{"name": "a"}]
    partial["compile_progress"]["completed_terms"] = 2
    with pytest.raises(AssertionError):
        records(partial)
    with pytest.raises(AssertionError):
        records(dict(completed_operations=["hessian"], terms=2,
                     timings={"hessian": {"term_timings": [{}]}}))


@pytest.mark.parametrize("size", [1, 3, 4, 5, 9])
def test_all_hessian_columns_including_partial_last_batch(size):
    rng = np.random.default_rng(280)
    matrix = rng.normal(size=(size, size))
    matrix = matrix+matrix.T
    calls = []

    def kernel(point, directions):
        calls.append(directions.copy())
        return directions @ matrix.T

    actual = all_columns(kernel, np.zeros(size))
    np.testing.assert_array_equal(actual, matrix)
    assert len(calls) == (size+3)//4
    assert all(c.shape == (min(4, size), size) for c in calls)
