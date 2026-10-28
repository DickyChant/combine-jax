import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fd_confirmation import select_confirmed_reference
from audit_asimov_native_reference import compare_coordinate


def probes(derivatives):
    return [dict(step=10.**(-i), derivative=d, status='valid', stencil='central', direction=1.)
            for i, d in enumerate(derivatives)]


def test_short_false_plateau_needs_an_independent_second_window():
    rows = probes([1., 1., 1., 2., 2., 2., 3., 3., 3., 3., 3., 3.])
    result = select_confirmed_reference(rows)
    assert result['selected_first_probe_index'] == 6
    assert result['reference_probe_count'] == 6
    assert select_confirmed_reference(rows[:9])['status'] == 'unresolved'


@pytest.mark.parametrize('change', ['invalid', 'repeated_step', 'stencil'])
def test_all_six_must_be_distinct_and_compatible(change):
    rows = probes([1.]*6)
    if change == 'invalid': rows[4]['status'] = 'invalid'
    if change == 'repeated_step': rows[4]['step'] = rows[3]['step']
    if change == 'stencil': rows[4]['stencil'] = 'one-sided'
    assert select_confirmed_reference(rows)['status'] == 'unresolved'


def test_ad_gate_checks_both_windows_without_weakening_tolerance():
    rows = probes([1e-10]*6)
    result = select_confirmed_reference(rows)
    assert result['status'] == 'roundoff_sensitive'
    case = dict(result, probes=rows); case['status'] = 'converged'
    assert compare_coordinate(1e-10, case)['status'] == 'passed'
    assert compare_coordinate(1.02e-10, case)['status'] == 'failed_gradient'
    changed = copy.deepcopy(case); changed['probes'][-1]['derivative'] = 1.02e-10
    assert compare_coordinate(1e-10, changed)['status'] == 'failed_gradient'
