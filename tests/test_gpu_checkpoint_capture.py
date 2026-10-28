"""Checkpoint integrity/lifecycle checks with explicitly synthetic payloads."""

import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from capture_spooled_gpu_gradient import capture, digest, fetch


def fixture_payload(*, random_points=True, corrupt=False):
    point = np.zeros(2656, dtype=np.float64)
    gradient = point.copy()
    if corrupt:
        gradient[0] = np.nan
    arrays = dict(point=point, gradient=gradient, vg_value=np.asarray(1.),
                  validation_points=np.tile(point, (26, 1)))
    if random_points:
        arrays.update(validation_gradients=np.tile(gradient, (26, 1)),
                      validation_vg_values=np.ones(26))
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    report = dict(arrays_sha256=digest(payload), terms=524, parameters=2656,
        timings=dict(value_and_grad=dict(term_timings=[{} for _ in range(524)],
            warm_seconds=[1.]*5, first_execution_seconds=2.,
            spooled_timed_calls=[dict(kernel_calls=524, total_seconds=t)
                                for t in [2., *([1.]*5)]])))
    return json.dumps(report).encode(), report, payload


def prepare(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("build/gpu_results").mkdir(parents=True)


@pytest.mark.parametrize("stage", ["timed", "all_points"])
def test_capture_preserves_exact_payload_and_rejects_overwrite(tmp_path, monkeypatch, stage):
    prepare(tmp_path, monkeypatch)
    raw, report, payload = fixture_payload(random_points=stage == "all_points")
    result = capture(raw, report, payload, stage)
    saved = json.loads(Path(result["report"]).read_text())
    assert saved["status"] == "checkpoint"
    assert Path(saved["checkpoint_raw"]).read_bytes() == raw
    assert Path(saved["arrays"]).read_bytes() == payload
    assert saved["checkpoint_parent_report_sha256"] == digest(raw)
    before = {p: p.read_bytes() for p in Path(".").rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        capture(raw, report, payload, stage)
    assert before == {p: p.read_bytes() for p in Path(".").rglob("*") if p.is_file()}


@pytest.mark.parametrize("fault", ["checksum", "partial_terms", "calls", "timing", "nonfinite",
                                 "missing_random"])
def test_capture_rejects_incomplete_or_inconsistent_evidence(tmp_path, monkeypatch, fault):
    prepare(tmp_path, monkeypatch)
    raw, report, payload = fixture_payload(random_points=fault != "missing_random",
                                          corrupt=fault == "nonfinite")
    timing = report["timings"]["value_and_grad"]
    if fault == "checksum":
        report["arrays_sha256"] = "wrong"
    elif fault == "partial_terms":
        timing["term_timings"].pop()
    elif fault == "calls":
        timing["spooled_timed_calls"][0]["kernel_calls"] -= 1
    elif fault == "timing":
        timing["warm_seconds"][0] = 3.
    with pytest.raises((AssertionError, KeyError)):
        capture(raw, report, payload, "all_points")
    assert not list(Path("reports").iterdir())
    assert not list(Path("build/gpu_results").iterdir())


@pytest.mark.parametrize("path", ["/etc/passwd", "build/../../escape", "other/file"])
def test_remote_path_rejection_precedes_ssh(path):
    with pytest.raises(ValueError, match="authorized project"):
        fetch(path)
