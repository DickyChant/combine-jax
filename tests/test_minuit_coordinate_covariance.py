"""Independent AD/native checks for the explicitly separate coordinate helper."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
from iminuit import Minuit
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import minuit_coordinate_covariance as helper
from probe_minuit_bound_covariance import run_case


@pytest.mark.parametrize("kind", ["unbounded", "double_lower", "double_upper",
                                  "single_lower", "single_upper", "linear_double_lower"])
def test_helper_matches_independent_ad_probe_and_installed_minuit(kind):
    case = run_case(kind)
    result = helper.coordinate_covariance(case["values"], case["gradient"], case["external_hessian"],
        [case["bounds"], (None, None), (None, None)], np.array([0, 1]),
        case["native_actual_internal_coordinates"])
    for key, reference in (("internal_hessian", "internal_hessian"),
                           ("internal_covariance", "internal_covariance"),
                           ("external_covariance", "external_covariance")):
        np.testing.assert_allclose(result[key], case[reference], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result["error_fields"][:2], case["ad_transformed_error_fields"], rtol=1e-9)
    assert case["status"] == "passed"
    assert result["error_fields"][2] == 0
    native_errors = np.asarray(case["native_error_fields"])
    native_cov = np.asarray(case["native_covariance"])
    error = np.max(abs(result["error_fields"][:2]-native_errors)/native_errors)
    norm = np.maximum(np.sqrt(np.outer(np.diag(native_cov), np.diag(native_cov))), 1e-8)
    difference = np.max(abs(result["external_covariance"]-native_cov)/norm)
    assert error < .01 and difference < .01
    print(kind, "native relative error", error, "normalized covariance difference", difference)
    if kind == "double_lower":
        with pytest.raises(ValueError, match="Actual internal coordinates"):
            helper.coordinate_covariance(case["values"], case["gradient"], case["external_hessian"],
                [case["bounds"], (None, None), (None, None)], np.array([0, 1]),
                case["clipped_inverse_coordinates"])


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("errordef", [.5, 1.])
def test_large_internal_error_and_errordef_match_native(upper, errordef):
    mean = 11. if upper else -1.

    def objective(point):
        x, y = point
        return .005*(x-mean)**2+.5*(y-.3*x)**2

    native = Minuit(lambda x, y: float(objective((x, y))), x=9.8 if upper else .2, y=0.)
    native.errordef, native.tol = errordef, 1e-9
    native.limits["x"] = (0., 10.)
    native.migrad(iterate=1, use_simplex=False)
    native.hesse()
    assert native.valid and native.accurate and not native.fmin.has_made_posdef_covar
    free, internal = helper.capture_internal_point(native)
    point = np.asarray(native.values)
    gradient = np.asarray(jax.grad(objective)(jnp.asarray(point)))
    hessian = np.asarray(jax.hessian(objective)(jnp.asarray(point)))
    result = helper.coordinate_covariance(point, gradient, hessian,
        [(0., 10.), (None, None)], free, internal, errordef=errordef)
    assert np.sqrt(result["internal_covariance"][0, 0]) > 1.
    native_errors, native_cov = np.asarray(native.errors), np.asarray(native.covariance)
    error = np.max(abs(result["error_fields"]-native_errors)/native_errors)
    norm = np.maximum(np.sqrt(np.outer(np.diag(native_cov), np.diag(native_cov))), 1e-8)
    difference = np.max(abs(result["external_covariance"]-native_cov)/norm)
    assert error < .01 and difference < .01
    print("large internal error", upper, errordef, "native relative error", error,
          "normalized covariance difference", difference)


@pytest.mark.parametrize("branch", [-4., -.4, .4, 4.])
def test_signed_transform_chain_rule_against_independent_jax(branch):
    u = jnp.array([branch, branch, branch, .2])

    def transform(v):
        return jnp.array([5.*(1.+jnp.sin(v[0])), jnp.sqrt(1.+v[1]**2)-1.,
                          11.-jnp.sqrt(1.+v[2]**2), v[3]])

    center = transform(u)

    def objective(p):
        return .5*jnp.sum((p-center)**2)+1e-4*jnp.sum(p)+.1*(p[0]-center[0])*(p[1]-center[1])

    result = helper.coordinate_covariance(center, jax.grad(objective)(center),
        jax.hessian(objective)(center), [(0., 10.), (0., None), (None, 10.), (None, None)],
        np.arange(4), u)
    transformed_hessian = np.asarray(jax.hessian(lambda p: objective(transform(p)))(u))
    np.testing.assert_allclose(result["internal_hessian"], transformed_hessian, atol=1e-12, rtol=1e-12)
    jacobian = np.asarray(jax.jacfwd(transform)(u))
    expected = jacobian@np.linalg.inv(transformed_hessian)@jacobian.T
    np.testing.assert_allclose(result["external_covariance"], expected, atol=1e-12, rtol=1e-12)


def baseline():
    return dict(point=np.array([1., 4., 2.]), gradient=np.array([0., 1., 0.]),
        hessian=np.array([[2., 0., .3], [0., -1., 0.], [.3, 0., 3.]]),
        bounds=[(None, None)]*3, free=np.array([0, 2]), internal=np.array([1., 2.]))


def test_all_correlations_and_fixed_rows_are_preserved():
    data = baseline()
    original = data["hessian"].copy()
    result = helper.coordinate_covariance(**data)
    expected = np.zeros((3, 3))
    expected[np.ix_([0, 2], [0, 2])] = np.linalg.inv(original[np.ix_([0, 2], [0, 2])])
    np.testing.assert_allclose(result["external_covariance"], expected)
    np.testing.assert_allclose(result["error_fields"], np.sqrt(np.diag(expected)))
    np.testing.assert_array_equal(data["hessian"], original)


def test_all_fixed_is_explicit_zero_covariance():
    data = baseline()
    data.update(free=np.array([], dtype=int), internal=np.array([]))
    result = helper.coordinate_covariance(**data)
    np.testing.assert_array_equal(result["external_covariance"], np.zeros((3, 3)))
    np.testing.assert_array_equal(result["error_fields"], np.zeros(3))


def test_equal_bounds_are_allowed_only_for_fixed_coordinates():
    data = baseline()
    data["bounds"][1] = (4., 4.)
    result = helper.coordinate_covariance(**data)
    np.testing.assert_array_equal(result["external_covariance"][1], np.zeros(3))
    data["bounds"][0] = (1., 1.)
    with pytest.raises(ValueError, match="free coordinates"):
        helper.coordinate_covariance(**data)


@pytest.mark.parametrize("endpoint", [np.inf, -np.inf, np.nan])
def test_nonfinite_bounds_need_explicit_absence(endpoint):
    data = baseline()
    data["bounds"][1] = (endpoint, None)
    with pytest.raises(ValueError, match="absent bound"):
        helper.coordinate_covariance(**data)


@pytest.mark.parametrize("change", ["point", "gradient", "fixed_hessian", "order", "duplicate", "bounds",
                                    "outside", "inverse_point", "asymmetric", "nonpositive", "indefinite"])
def test_invalid_inputs_and_nonpositive_curvature_are_rejected(change):
    data = baseline()
    if change == "point":
        data["point"][0] = np.nan
    elif change == "gradient":
        data["gradient"][0] = np.inf
    elif change == "fixed_hessian":
        data["hessian"][1, 1] = np.nan
    elif change == "order":
        data["free"] = np.array([2, 0])
    elif change == "duplicate":
        data["free"] = np.array([0, 0])
    elif change == "bounds":
        data["bounds"][1] = (2., 1.)
    elif change == "outside":
        data["bounds"][1] = (0., 3.)
    elif change == "inverse_point":
        data["internal"][0] += 1e-5
    elif change == "asymmetric":
        data["hessian"][0, 2] += .1
    elif change == "nonpositive":
        data["hessian"][0, 0] = 0.
    else:
        data["hessian"][0, 2] = data["hessian"][2, 0] = 4.
    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        helper.coordinate_covariance(**data)


@pytest.mark.parametrize("errordef", [0., -1., np.inf, np.nan])
def test_invalid_errordef_rejected(errordef):
    with pytest.raises(ValueError, match="errordef"):
        helper.coordinate_covariance(**baseline(), errordef=errordef)


def test_unstable_inverse_rejected(monkeypatch):
    monkeypatch.setattr(helper.np.linalg, "inv", lambda x: np.eye(len(x)))
    with pytest.raises(ValueError, match="Unstable"):
        helper.coordinate_covariance(**baseline())


def test_private_state_capture_rejects_unvalidated_version(monkeypatch):
    monkeypatch.setattr(helper, "version", lambda _: "other")
    with pytest.raises(ValueError, match="2.33.0"):
        helper.capture_internal_point(None)


def test_state_capture_requires_minimization():
    with pytest.raises(ValueError, match="minimization"):
        helper.capture_internal_point(Minuit(lambda x: x*x, x=1.))


def test_capture_keeps_middle_fixed_coordinate_mapping_and_rejects_changed_state():
    native = Minuit(lambda x, y, z: (x-1.)**2+(y-2.)**2+(z-3.)**2, x=0., y=2., z=0.)
    native.fixed["y"] = True
    native.migrad()
    free, internal = helper.capture_internal_point(native)
    np.testing.assert_array_equal(free, [0, 2])
    np.testing.assert_array_equal(internal, np.asarray(native.values)[free])
    native.values["x"] += 1.
    with pytest.raises(ValueError, match="does not reproduce"):
        helper.capture_internal_point(native)
