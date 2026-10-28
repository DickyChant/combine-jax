"""A candidate point check must not silently accept an unfinished minimum."""

import copy
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from check_exported_fit_candidate import candidate_errors, checked_point


@pytest.fixture
def inputs():
    likelihood = SimpleNamespace(parameter_names=("x", "fixed"), initial=np.array([0., 2.]),
        parameters=(SimpleNamespace(minimum=-1., maximum=1.),
                    SimpleNamespace(minimum=2., maximum=2.)),
        discrete_choices={}, mask_configuration={})
    parent = dict(memory_policy="resident_value_spooled_gradient", float64=True, root_imported=False,
        parameter_names=["x", "fixed"], discrete_choices={}, channel_masks={},
        status="running", best_seen_values=[.1, 2.], best_seen_nll=3.,
        final=dict(values=[.1, 2.], nll=3.), stage="finished", evaluation_counters=dict(closed=True))
    return parent, likelihood


def test_running_snapshot_is_a_point_not_a_finished_minimum(inputs):
    parent, likelihood = inputs
    point, value = checked_point(parent, likelihood, "best_seen")
    np.testing.assert_array_equal(point, [.1, 2.])
    assert value == 3.
    with pytest.raises(AssertionError):
        checked_point(parent, likelihood, "final")


@pytest.mark.parametrize("mutation", [
    {"best_seen_values": [2., 2.]}, {"best_seen_values": [.1, 2.1]},
    {"best_seen_values": [.1]}, {"best_seen_values": [np.nan, 2.]},
    {"best_seen_nll": np.inf}, {"parameter_names": ["fixed", "x"]},
    {"discrete_choices": {"unexpected": {}}}, {"channel_masks": {"unexpected": {}}},
    {"float64": False}, {"root_imported": True},
])
def test_invalid_or_differently_configured_point_is_rejected(inputs, mutation):
    parent, likelihood = inputs
    parent.update(mutation)
    with pytest.raises(AssertionError):
        checked_point(parent, likelihood, "best_seen")


def test_final_point_requires_closed_producer_but_does_not_claim_convergence(inputs):
    parent, likelihood = inputs
    parent["status"] = "fit_incomplete"
    checked_point(parent, likelihood, "final")
    parent["evaluation_counters"]["closed"] = False
    with pytest.raises(AssertionError):
        checked_point(parent, likelihood, "final")


def test_direct_nll_offset_and_restore_checks_are_independent():
    arguments = [100., -2., 98., 100., 98.]
    errors, passed = candidate_errors(*arguments)
    assert passed and all(value == 0 for value in errors.values())
    for index in range(len(arguments)):
        changed = copy.copy(arguments)
        changed[index] += 1e-5
        assert not candidate_errors(*changed)[1]
        changed[index] = np.nan
        with pytest.raises(AssertionError):
            candidate_errors(*changed)
