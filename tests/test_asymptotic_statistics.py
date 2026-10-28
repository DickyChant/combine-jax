import math

import numpy as np
import pytest
from scipy.special import ndtri

from combine_jax.methods.asymptotic import (QUANTILES, LimitOptions, asymptotic_cls,
    expected_cls, expected_limits, expected_nll_crossing, qmu_tilde)


def test_qtilde_one_sided_and_boundary_denominator():
    assert qmu_tilde(1., 2., 10., 3.) == 0.
    assert qmu_tilde(1., 0., 10., 3.) == 14.
    assert qmu_tilde(1., 0., 3.-1e-12, 3.) == 0.
    with pytest.raises(ValueError):
        qmu_tilde(1., -.1, 10., 3.)


def test_cls_branches_join_and_median_normal_tail():
    q = float(ndtri(.975))**2
    assert asymptotic_cls(q, q)['cls'] == pytest.approx(.05)
    for eps in (-1e-10, 1e-10):
        assert asymptotic_cls(q+eps, q)['cls'] == pytest.approx(.05, abs=1e-10)
    assert asymptotic_cls(9., 4.)['cls'] == pytest.approx(.0005770250423907665/.10564977366685535)
    assert asymptotic_cls(0., 0.)['cls'] == 1.
    with pytest.raises(ValueError):
        asymptotic_cls(1., 0.)


def test_expected_thresholds_have_exact_requested_cls():
    for p in QUANTILES:
        assert expected_cls(2*expected_nll_crossing(p), p) == pytest.approx(.05, abs=1e-14)


def test_quadratic_profile_all_bands_and_rmax_expansion():
    sigma = 2.3
    profile = lambda mu: .5*(mu/sigma)**2 + 13.
    result = expected_limits(profile, 13., .5)
    exact = [sigma*math.sqrt(2*expected_nll_crossing(p)) for p in QUANTILES]
    np.testing.assert_allclose(result['limits'], exact, rtol=.01, atol=0.)
    assert result['final_r_max'] > .5 and result['profile_calls'] > 0


def test_strict_upper_bound_and_unresolved_sensitivity_are_distinct():
    result = expected_limits(lambda mu: .5*mu**2, 0., .1, options=LimitOptions(strict_bounds=True))
    np.testing.assert_allclose(result['limits'], .1, atol=.0005)
    with pytest.raises(RuntimeError, match='watchdog'):
        expected_limits(lambda mu: 0., 0., 1., options=LimitOptions(max_profile_calls=12))
    with pytest.raises(RuntimeError, match='nonfinite'):
        expected_limits(lambda mu: math.nan, 0., 1.)


def test_observed_quadratic_matches_independent_cls_root():
    from scipy.optimize import brentq
    from combine_jax.methods.asymptotic import observed_limit
    for estimate in (0., .8, 2.):
        data = lambda mu: 7.+.5*(mu-estimate)**2
        asimov = lambda mu: 3.+.5*mu**2
        result = observed_limit(data, asimov, 7., 3., estimate, 10., r_error=1.)
        exact = brentq(lambda mu: asymptotic_cls((mu-estimate)**2, mu**2)['cls']-.05,
                       estimate+1e-6, 10.)
        assert result['observed_limit'] == pytest.approx(exact, rel=.01)
        assert result['cls_bracket'][0] >= .05 >= result['cls_bracket'][1]
        assert result['trace'][-1]['mu'] == result['observed_limit']


def test_observed_no_sensitivity_is_not_a_limit():
    from combine_jax.methods.asymptotic import observed_limit
    with pytest.raises(RuntimeError, match='upper bracket'):
        observed_limit(lambda mu: 0., lambda mu: 0., 0., 0., 0., 1.)
