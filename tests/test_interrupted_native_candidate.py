"""An interrupted native fit supplies a verified point, never a final minimum."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_combine_fit_restart import interrupted_best_point


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    parent, raw, audit_path = [tmp_path/name for name in ("parent.json", "raw.json", "audit.json")]
    original = dict(status="running", best_seen_values=[.2, 2.],
                    best_seen_zero_point_nll=-3., parameter_names=["x", "fixed"])
    raw.write_text(json.dumps(original))
    previous = dict(original, status="interrupted", producer_last_status="running",
        interruption=dict(raw_report=str(raw), raw_report_sha256=digest(raw),
                          audit=str(audit_path), tool_session=123, tool_command_exit_code=143))
    parent.write_text(json.dumps(previous))
    audit = dict(status="interrupted_without_final_fit",
        annotated_reports_sha256={str(parent): digest(parent)},
        raw_reports={str(parent): dict(path=str(raw), sha256=digest(raw))},
        sources_unchanged=True, final_minimum_present=False, final_covariance_present=False,
        tool_session=123, tool_command_exit_code=143)
    audit_path.write_text(json.dumps(audit))
    return parent, previous, raw, audit_path


def test_preserved_point_is_available_without_convergence(evidence):
    parent, previous, _, _ = evidence
    point, nll = interrupted_best_point(parent, previous)
    np.testing.assert_array_equal(point, [.2, 2.])
    assert point.dtype == np.float64 and nll == -3.
    assert "final" not in previous


@pytest.mark.parametrize("field,value", [
    ("status", "passed"), ("sources_unchanged", False),
    ("final_minimum_present", True), ("final_covariance_present", True),
    ("tool_session", 124), ("tool_command_exit_code", 0),
])
def test_inconsistent_interruption_audit_is_rejected(evidence, field, value):
    parent, previous, _, audit_path = evidence
    audit = json.loads(audit_path.read_bytes())
    audit[field] = value
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(AssertionError):
        interrupted_best_point(parent, previous)


def test_changed_raw_evidence_is_rejected(evidence):
    parent, previous, raw, _ = evidence
    raw.write_text(raw.read_text()+" ")
    with pytest.raises(AssertionError):
        interrupted_best_point(parent, previous)


def test_changed_parent_is_rejected(evidence):
    parent, previous, _, _ = evidence
    parent.write_text(parent.read_text()+" ")
    with pytest.raises(AssertionError):
        interrupted_best_point(parent, previous)


@pytest.mark.parametrize("field,value", [
    ("best_seen_values", [.3, 2.]),
    ("best_seen_zero_point_nll", -4.),
    ("parameter_names", ["fixed", "x"]),
    ("final", {"is_valid": True}),
])
def test_even_rebound_parent_must_preserve_raw_point(evidence, field, value):
    parent, previous, _, audit_path = evidence
    previous[field] = value
    parent.write_text(json.dumps(previous))
    audit = json.loads(audit_path.read_bytes())
    audit["annotated_reports_sha256"][str(parent)] = digest(parent)
    audit_path.write_text(json.dumps(audit))
    with pytest.raises(AssertionError):
        interrupted_best_point(parent, previous)
