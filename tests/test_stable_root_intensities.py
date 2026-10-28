"""Independent 70-digit checks for continuous extended-mixture NLL differences."""

from decimal import Decimal, localcontext
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from stable_root_intensities import extended_delta


def dec(value):
    return Decimal(np.format_float_scientific(np.longdouble(value), precision=35, unique=False))


@pytest.mark.parametrize("yield_shift,shifts", [
    ("0.01", ["0.000001", "-0.000002", "0.000003"]),
    ("-0.03", ["-0.000001", "0.000002", "-0.000003"]),
    ("4", ["2", "-0.2", "3"]),
])
def test_extended_delta_against_decimal(yield_shift, shifts):
    baseline_yield = np.longdouble("100000000")
    expected_yield = baseline_yield+np.longdouble(yield_shift)
    baseline = np.array(["1.1", "2.7", "15.5"], dtype=np.longdouble)
    intensities = baseline+np.array(shifts, dtype=np.longdouble)
    weights = np.array(["12.5", "0.2", "19"], dtype=np.longdouble)
    with localcontext() as context:
        context.prec = 70
        expected = dec(expected_yield)-dec(baseline_yield)-sum(dec(w)*(dec(a)/dec(b)).ln()
                    for a,b,w in zip(intensities,baseline,weights,strict=True))
        scale = abs(dec(expected_yield)-dec(baseline_yield))+sum(abs(dec(w)*(dec(a)/dec(b)).ln())
                    for a,b,w in zip(intensities,baseline,weights,strict=True))
    actual = extended_delta(expected_yield,baseline_yield,intensities,baseline,weights)
    assert abs(dec(actual)-expected) < Decimal("20")*dec(np.finfo(np.longdouble).eps)*scale


def test_empty_rows_and_invalid_inputs():
    assert extended_delta(12.,10.,[0.,3.],[0.,0.],[0.,0.]) == 2
    with pytest.raises(AssertionError):
        extended_delta(1.,1.,[0.],[1.],[1.])
    with pytest.raises(AssertionError):
        extended_delta(1.,1.,[1.],[0.],[1.])
    with pytest.raises(AssertionError):
        extended_delta(0.,1.,[1.],[1.],[1.])
