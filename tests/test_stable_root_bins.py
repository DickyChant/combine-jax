"""Check stable Poisson differences against independent 70-digit arithmetic."""

from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from stable_root_bins import poisson_delta


def decimal_value(x):
    return Decimal(np.format_float_scientific(np.longdouble(x),precision=35,unique=False))


@pytest.mark.parametrize("baseline,observed,shift", [
    ("100000000", "100000003", "0.01"),
    ("100000000", "100000003", "-0.01"),
    ("2.4", "17.5", "0.5"),
    ("3.0", "0.0", "-1.0"),
])
def test_poisson_delta_high_precision(baseline,observed,shift):
    baseline,observed,shift = map(np.longdouble,(baseline,observed,shift))
    mean = baseline+shift
    with localcontext() as context:
        context.prec = 70
        a,b,n = map(decimal_value,(mean,baseline,observed))
        expected = a-b-n*(a/b).ln()
    actual = poisson_delta(mean,baseline,observed)
    assert abs(decimal_value(actual)-expected) < Decimal("1e-17")*max(abs(expected),Decimal("1e-8"))


def test_poisson_delta_vector_and_invalid_means():
    baseline = np.array([1e8,1e8+1],dtype=np.longdouble)
    means = baseline+np.array([.01,-.01],dtype=np.longdouble)
    observed = np.array([1e8+3,1e8-4],dtype=np.longdouble)
    with localcontext() as context:
        context.prec = 70
        expected = sum(decimal_value(a)-decimal_value(b)-decimal_value(n)*(decimal_value(a)/decimal_value(b)).ln()
                       for a,b,n in zip(means,baseline,observed,strict=True))
    actual = poisson_delta(means,baseline,observed)
    assert abs(decimal_value(actual)-expected) < Decimal("1e-24")
    with pytest.raises(AssertionError):
        poisson_delta([0.,1.],baseline,observed)


def test_exact_empty_bin_limit():
    assert poisson_delta([0.,2.,0.],[0.,0.,3.],[0.,0.,0.]) == -1
    assert poisson_delta(0.,0.,0.) == 0
    with pytest.raises(AssertionError):
        poisson_delta(-1.,1.,0.)
    with pytest.raises(AssertionError):
        poisson_delta(1.,0.,2.)
