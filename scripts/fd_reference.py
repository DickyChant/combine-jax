"""Independent local derivative consistency checks with short stencils."""

import numpy as np


def estimates(deltas, step):
    """Deltas are f(x+k*h)-f(x); use no samples outside [-2*h,2*h]."""
    values = {k:np.longdouble(deltas[k]) for k in (-2,-1,1,2)}
    h = np.longdouble(step)
    assert np.isfinite(h) and h > 0 and all(np.isfinite(v) for v in values.values())
    return dict(central_two=float((values[1]-values[-1])/(2*h)),
                forward_two=float((4*values[1]-values[2])/(2*h)),
                backward_two=float((4*values[-1]-values[-2])/(-2*h)),
                central_four=float((8*(values[1]-values[-1])-(values[2]-values[-2]))/(12*h)))


def first_consistent_triple(probes, tolerance=1e-5):
    """Select using all twelve estimates, independent of any AD derivative."""
    for first in range(len(probes)-2):
        values = [v for p in probes[first:first+3] for v in p["estimates"].values()]
        assert len(values) == 12
        if not np.isfinite(values).all():
            continue
        spread = (max(values)-min(values))/max(min(abs(v) for v in values),1e-8)
        if spread < tolerance:
            return dict(first_probe_index=first, relative_spread=spread)
    return None
