"""Measurement collection cannot promote incomplete references to acceptance."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_tensor_benchmarks import available_reference_status
from audit_tensor_edm_timing import timing_summary


def test_pending_native_derivatives_are_explicit_and_failures_take_precedence():
    rows=[dict(passed=True),dict(passed=True,native_gradient_validation='pending')]
    assert available_reference_status(rows)=='passed_available_gates_native_gradients_pending'
    rows[0]['passed']=False
    assert available_reference_status(rows)=='failed'
    assert available_reference_status([dict(passed=True)])=='passed'


def test_asimov_speed_ratio_requires_completed_gradient_validation():
    native=dict(dataset='asimov_prefit')
    comparison=dict(case='hhvv',dataset='asimov_prefit')
    with pytest.raises(AssertionError,match='completed Asimov native gradient gate'):
        timing_summary(native,{},comparison)
    for status in ('incomplete_or_failed_native_gradient_gate','checking'):
        with pytest.raises(AssertionError):
            timing_summary(native,{},comparison,native_validation=dict(status=status))
