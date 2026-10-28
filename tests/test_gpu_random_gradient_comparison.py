"""Acceptance checks against independent polynomial finite differences."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/"scripts/compare_gpu_random_gradients.py"
SPEC = importlib.util.spec_from_file_location("random_gradient_comparison", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def polynomial_reference():
    x = np.longdouble("0.7")
    probes = []
    for step in ("0.001", "0.0005", "0.0002"):
        h = np.longdouble(step)
        shifted = x+np.array([-2,-1,1,2], dtype=np.longdouble)*h
        values = 3*shifted+2*shifted**2+shifted**4
        derivative = float(np.dot(values, [1,-8,8,-1])/(12*h))
        probes.append(dict(status="valid", step=float(h), derivative=derivative))
    return dict(name="x", index=0, status="converged", probes=probes,
                selected_first_probe_index=0), float(3+4*x+4*x**3)


def test_independent_polynomial_and_perturbed_derivative():
    case, derivative = polynomial_reference()
    assert comparison.compare_coordinate(case, derivative)["status"] == "passed"
    assert comparison.compare_coordinate(case, derivative*1.01)["status"] == "failed"


def test_all_three_references_are_required():
    case, _ = polynomial_reference()
    for probe, derivative in zip(case["probes"], (100.,100.009,100.018)):
        probe["derivative"] = derivative
    result = comparison.compare_coordinate(case, 100.)
    assert result["status"] == "failed"
    assert result["selected_references"][0]["relative_error"] == 0
    assert result["selected_references"][2]["relative_error"] > 1e-4


@pytest.mark.parametrize("status", ("unresolved", "roundoff_sensitive"))
def test_uncertain_reference_cannot_pass(status):
    case, derivative = polynomial_reference()
    case["status"] = status
    assert comparison.compare_coordinate(case, derivative)["status"] == "reference_uncertain"


def test_cannot_switch_to_later_reference_triple():
    case, derivative = polynomial_reference()
    case["probes"] += case["probes"]
    case["selected_first_probe_index"] = 3
    with pytest.raises(AssertionError, match="FIRST|first"):
        comparison.compare_coordinate(case, derivative)


@pytest.mark.parametrize("status", ("converged", "fixed_bounds", "roundoff_sensitive"))
def test_nonfinite_ad_is_always_a_failure(status):
    case, _ = polynomial_reference()
    case["status"] = status
    result = comparison.compare_coordinate(case, np.nan)
    assert result["status"] == "failed" and result["ad_derivative"] is None


@pytest.mark.parametrize("outcome", ("passed","failed","bad_delta","changed_step","reference_uncertain"))
def test_stable_small_polynomial_reference(outcome):
    case, analytic = polynomial_reference()
    case.update(status="roundoff_sensitive",coordinate=.7)
    for p in case["probes"]:
        p["derivative"] *= 1e-7
    stable = dict(name="x",index=0,coordinate=.7,status="reference_ready",probes=[])
    scale = np.longdouble("1e-7")
    x = np.longdouble(.7)
    base = scale*(3*x+2*x*x+x**4)
    for probe in case["probes"]:
        coordinates = [.7+i*probe["step"] for i in (-2,-1,1,2)]
        shifted = np.array(coordinates,dtype=np.longdouble)
        values = scale*(3*shifted+2*shifted**2+shifted**4)-base
        derivative = float(np.dot(values,[1,-8,8,-1])/(12*np.longdouble(probe["step"])))
        stable["probes"].append(dict(step=probe["step"],derivative=derivative,
            delta_checks=[dict(coordinate=a,stable_delta=float(v),combine_delta=float(v),absolute_error=0.)
                          for a,v in zip(coordinates,values,strict=True)]))
    derivative = analytic*1e-7
    if outcome == "failed":
        derivative *= 1.01
    if outcome == "bad_delta":
        stable["probes"][0]["delta_checks"][0]["combine_delta"] += 1e-3
    if outcome == "changed_step":
        stable["probes"][0]["step"] *= 2
    if outcome == "reference_uncertain":
        stable["status"] = "reference_uncertain"
    if outcome in ("bad_delta","changed_step"):
        with pytest.raises(AssertionError):
            comparison.compare_stable_coordinate(case,stable,derivative)
    else:
        result = comparison.compare_stable_coordinate(case,stable,derivative)
        assert result["status"] == outcome
        assert result["reference_status"] == "roundoff_sensitive"
