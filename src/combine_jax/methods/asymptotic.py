"""Combine v11 expected CLs conventions and profile-crossing procedure.

This module consumes a conditional-profile callable and contains no ROOT/JAX
objects. It does not generate datasets, fit observed data, or certify its inputs.
"""
from dataclasses import dataclass
import math

from scipy.special import log_ndtr, ndtr, ndtri


QUANTILES = (.025, .16, .5, .84, .975)


@dataclass(frozen=True)
class LimitOptions:
    confidence_level: float = .95
    r_abs_accuracy: float = .0005
    r_rel_accuracy: float = .005
    minimizer_tolerance: float = .1
    strict_bounds: bool = False
    max_profile_calls: int = 1000

    def __post_init__(self):
        if not .5 < self.confidence_level < 1:
            raise ValueError('Confidence level must be between 0.5 and 1')
        for value in (self.r_abs_accuracy, self.r_rel_accuracy, self.minimizer_tolerance):
            if not math.isfinite(value) or value <= 0:
                raise ValueError('Accuracies and minimizer tolerance must be positive and finite')
        if type(self.max_profile_calls) is not int or self.max_profile_calls < 1:
            raise ValueError('A positive profile-call watchdog is required')


def qmu_tilde(mu, mu_hat, conditional_nll, constrained_minimum_nll):
    """One-sided statistic with a denominator already minimized at mu >= 0."""
    if not all(math.isfinite(x) for x in (mu, mu_hat, conditional_nll, constrained_minimum_nll)):
        raise ValueError('Nonfinite statistic input')
    if mu < 0 or mu_hat < 0:
        raise ValueError('Use the nonnegative constrained estimate for q_mu tilde')
    return 0. if mu_hat > mu else max(0., 2 * (conditional_nll - constrained_minimum_nll))


def asymptotic_cls(q_mu, q_asimov):
    """LHC q-tilde CLs+b/CLb; log tails avoid numerical 0/0."""
    if not all(math.isfinite(x) and x >= 0 for x in (q_mu, q_asimov)):
        raise ValueError('Test statistics must be finite and nonnegative')
    if q_asimov == 0:
        if q_mu == 0:
            return dict(cl_sb=.5, cl_b=.5, cls=1.)
        raise ValueError('Positive statistic with zero Asimov sensitivity is undefined')
    a = math.sqrt(q_asimov)
    if q_mu <= q_asimov:
        x, y = math.sqrt(q_mu), a - math.sqrt(q_mu)
    else:
        x, y = (q_mu + q_asimov) / (2 * a), (q_asimov - q_mu) / (2 * a)
    return dict(cl_sb=float(ndtr(-x)), cl_b=float(ndtr(y)),
                cls=float(math.exp(float(log_ndtr(-x) - log_ndtr(y)))))


def expected_cls(q_asimov, quantile):
    if not math.isfinite(q_asimov) or q_asimov < 0 or not 0 < quantile < 1:
        raise ValueError('Invalid Asimov statistic or expected quantile')
    return float(ndtr(ndtri(quantile) - math.sqrt(q_asimov)) / quantile)


def expected_nll_crossing(quantile, confidence_level=.95):
    if not 0 < quantile < 1 or not .5 < confidence_level < 1:
        raise ValueError('Invalid expected quantile or confidence level')
    # -ndtri(alpha*p) is the inverse normal survival function without 1-x loss.
    return .5 * float(ndtri(quantile) - ndtri((1 - confidence_level) * quantile)) ** 2


def expected_limits(profile, minimum_nll, r_max, *, options=None):
    """Native default stepping median and bisection bands, with warm-start caller.

    ``profile(mu)`` must return a finite minimized NLL and raise on a failed fit.
    A watchdog failure raises; only explicit strict bounds permit saturation.
    As in native Combine, the geometric stopping rule can return an unevaluated
    final midpoint. The trace therefore distinguishes last evaluated and returned
    points and never labels an old NLL as the NLL of that midpoint.
    """
    options = options or LimitOptions()
    if not math.isfinite(minimum_nll) or not math.isfinite(r_max) or r_max <= 0:
        raise ValueError('A finite baseline and positive initial rMax are required')
    physical_max = float(r_max)
    trace = []

    def evaluate(mu, p):
        if len(trace) >= options.max_profile_calls:
            raise RuntimeError('Expected-limit profile-call watchdog exhausted; no limit accepted')
        value = float(profile(mu))
        if not math.isfinite(value):
            raise RuntimeError('Conditional profile returned a nonfinite NLL')
        trace.append(dict(mu=mu, quantile=p, nll=value, delta_nll=value-minimum_nll))
        return value

    def crossing(low, high, p, stepping=False):
        nonlocal physical_max
        target = minimum_nll + expected_nll_crossing(p, options.confidence_level)
        if options.strict_bounds:
            high = min(high, physical_max)
            if high == low:
                return high, 'strict_bound'
        cross, error = .5 * (low + high), .5 * (high - low)
        stride = .05 * high
        overshot = False
        if stepping:
            # Native evaluates the NLL at the midpoint before the first 5% step.
            # Profiling that midpoint is unnecessary: the adapter can expose the
            # current nuisance solution's unprofiled objective at another mu.
            raw = getattr(profile, 'unprofiled', None)
            there = float(raw(cross)) if raw else evaluate(cross, p)
            cross, error = .05 * high, high
        reason = 'parameter_accuracy'
        while error > max(options.r_rel_accuracy * cross, options.r_abs_accuracy):
            if cross >= physical_max:
                if options.strict_bounds:
                    cross = physical_max
                else:
                    physical_max = cross * 1.1
            here = evaluate(cross, p)
            if abs(here - target) < .05 * options.minimizer_tolerance:
                reason = 'nll_accuracy'
                break
            if stepping:
                if here < target:
                    if target - here < .5 * abs(target - there):
                        stride *= .5
                    if options.strict_bounds and cross == physical_max:
                        return cross, 'strict_bound'
                    cross += stride
                else:
                    stride *= .5
                    overshot = True
                    cross -= stride
                if overshot:
                    error = stride
                there = here
            else:
                if here < target:
                    low = cross
                else:
                    high = cross
                cross, error = .5 * (low + high), .5 * (high - low)
        return cross, reason

    median, median_reason = crossing(0., physical_max, .5, stepping=True)
    sigma = median / -float(ndtri(.5 * (1 - options.confidence_level)))
    limits, reasons = [], []
    for i, p in enumerate(QUANTILES):
        if i == 2:
            value, reason = median, median_reason
        else:
            low = 0. if i == 0 else limits[-1]
            high = median if i < 2 else median + (2 if i == 3 else 4) * sigma
            value, reason = crossing(low, high, p)
        limits.append(min(value, physical_max) if options.strict_bounds else value)
        reasons.append(reason)
    if not all(math.isfinite(x) and x >= 0 for x in limits) or any(
            a > b for a, b in zip(limits[:-1], limits[1:])):
        raise RuntimeError('Expected limits are nonfinite or unordered')
    return dict(quantiles=list(QUANTILES), limits=limits, crossing_reasons=reasons,
                profile_calls=len(trace), trace=trace, initial_r_max=r_max,
                final_r_max=physical_max, minimum_nll=minimum_nll,
                rule='CLs', confidence_level=options.confidence_level,
                qtilde=True, new_expected=True, median_algorithm='stepping',
                band_algorithm='bisection', options=options.__dict__)


def observed_limit(profile_data, profile_asimov, minimum_data, minimum_asimov,
                   mu_hat, r_max, *, r_error=None, options=None):
    """Combine runLimit/getCLs stepping, given two converged profile oracles.

    The input Asimov must be generated at the conditional background fit to
    the same real data, including its fitted global observables. Passing a
    pre-fit Asimov is only appropriate for explicit --noFitAsimov operation.
    r_error supplies the optimizer's POI error for native bracketing; without
    it the native 2%-of-range floor is used, not a claimed Hesse uncertainty.
    """
    options = options or LimitOptions()
    if not all(math.isfinite(x) for x in (minimum_data, minimum_asimov, mu_hat, r_max)):
        raise ValueError('Nonfinite observed-limit input')
    if mu_hat < 0 or r_max <= mu_hat:
        raise ValueError('Constrained estimate must be in the nonnegative POI range')
    error = .02*r_max if r_error is None else max(float(r_error), .02*r_max)
    if not math.isfinite(error) or error <= 0:
        raise ValueError('Invalid bracketing error')
    target = 1-options.confidence_level
    trace = []

    def evaluate(mu):
        if len(trace) >= options.max_profile_calls:
            raise RuntimeError('Observed-limit profile-call watchdog exhausted')
        d, a = float(profile_data(mu)), float(profile_asimov(mu))
        if not math.isfinite(d) or not math.isfinite(a):
            raise RuntimeError('Conditional profile returned a nonfinite NLL')
        q = qmu_tilde(mu, mu_hat, d, minimum_data)
        qa = max(0., 2*(a-minimum_asimov))
        tails = asymptotic_cls(q, qa)
        trace.append(dict(mu=float(mu), data_nll=d, asimov_nll=a,
                          q_mu=q, q_asimov=qa, **tails))
        return tails['cls']

    lo, hi = mu_hat, mu_hat+3*error
    if options.strict_bounds:
        hi = min(hi, r_max)
    cls_lo = 1.
    bracketed = False
    for _ in range(5):
        cls_hi = evaluate(hi)
        if cls_hi < target:
            bracketed = True
            break
        if options.strict_bounds and hi == r_max:
            raise RuntimeError('Observed CLs crossing exceeds strict rMax')
        hi = min(2*hi, r_max) if options.strict_bounds else 2*hi
    if not bracketed:
        raise RuntimeError('Observed CLs search did not establish an upper bracket')
    while True:
        half_width = .5*(hi-lo)
        if cls_lo < 3*target and cls_hi > .3*target:
            cross = lo+(hi-lo)*math.log(cls_lo/target)/math.log(cls_lo/cls_hi)
            mu = .8*cross+.2*(hi if cross-lo < hi-cross else lo)
        else:
            mu = .5*(lo+hi)
        cls = evaluate(mu)
        if cls > target:
            lo, cls_lo = mu, cls
        else:
            hi, cls_hi = mu, cls
        if half_width <= max(options.r_rel_accuracy*mu, options.r_abs_accuracy):
            break
    return dict(observed_limit=mu, limit_error=half_width, trace=trace,
        profile_calls=len(trace), evaluated_bracket=[lo, hi], cls_bracket=[cls_lo, cls_hi],
        returned_cls=cls, rule='CLs', confidence_level=options.confidence_level,
        test_statistic='q_mu_tilde', mu_hat=mu_hat, minimum_data_nll=minimum_data,
        minimum_asimov_nll=minimum_asimov, bracketing_error=error,
        bracketing_error_source='range_floor' if r_error is None else 'supplied_poi_error',
        options=vars(options), crossing_reason='parameter_accuracy')
