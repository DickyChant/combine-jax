from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_cached_deltas import extended_delta, log_remainder, poisson_delta


@pytest.mark.parametrize('x', ['1e-12', '-1e-12', '.01', '-.1', '2', '-.8'])
def test_log_remainder_against_decimal(x):
    with localcontext() as context:
        context.prec = 65
        value = Decimal(x)
        expected = value-(1+value).ln()
    actual = log_remainder(np.longdouble(x))
    assert abs(actual-np.longdouble(str(expected))) <= 5e-18*abs(actual)


def test_stationary_extended_delta_does_not_subtract_large_nlls():
    baseline = np.array([2**30, 2**29, 0], dtype=np.longdouble)
    current = baseline+np.array([2**-10, -2**-9, 2**-8], dtype=np.longdouble)
    actual = extended_delta(current, baseline, baseline, current.sum(), baseline.sum())
    with localcontext() as context:
        context.prec = 65
        # All powers of two and their sums are exactly representable here.
        b = [Decimal(2)**30, Decimal(2)**29, Decimal(0)]
        c = [b[0]+Decimal(2)**-10, b[1]-Decimal(2)**-9, Decimal(2)**-8]
        expected = sum(c)-sum(b)
        expected -= sum(w*(x/y).ln() for w, x, y in zip(b, c, b) if w)
    assert abs(actual-np.longdouble(str(expected))) < 2e-20


def test_zero_weight_rows_and_invalid_counted_density():
    assert extended_delta([1, -5], [1, 0], [2, 0], 3, 2) == 1
    with pytest.raises(ValueError, match='Nonpositive'):
        extended_delta([0], [1], [2], 3, 2)
    with pytest.raises(ValueError, match='Mismatched'):
        extended_delta([1, 2], [1], [2], 3, 2)


def test_stationary_small_bin_is_not_lost_among_large_constraint_means():
    baseline = np.array([2**50, 10], dtype=np.longdouble)
    mean = baseline+np.array([0, 2**-30], dtype=np.longdouble)
    result = poisson_delta(mean, baseline, baseline)
    with localcontext() as context:
        context.prec = 70
        delta = Decimal(2)**-30
        expected = delta-10*(1+delta/10).ln()
    assert abs(result-np.longdouble(str(expected))) < 1e-36
    # Subtracting total means first loses the shifted small bin completely.
    assert abs(extended_delta(mean, baseline, baseline, mean.sum(), baseline.sum())-result) > 1e-10


def test_poisson_zero_counts_and_invalid_arguments():
    assert poisson_delta([0, 2], [0, 1], [0, 0]) == 1
    with pytest.raises(ValueError, match='Nonpositive'):
        poisson_delta([0], [1], [2])
    with pytest.raises(ValueError, match='Invalid'):
        poisson_delta([-1], [1], [0])
