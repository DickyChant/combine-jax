"""ROOT/JAX-independent finite differences for native reference validation."""
import numpy as np


def derivative_column(evaluate, point, index, step, low, high, *, bound_aware=False):
    """Same five-point stencil, with paired differences cancelling offsets."""
    shift=np.zeros(len(point));shift[index]=step
    if bound_aware and min(point[index]-low[index],high[index]-point[index])<2*step:
        direction=1 if high[index]-point[index]>=point[index]-low[index] else -1
        step=min(step,max(high[index]-point[index],point[index]-low[index])/5)
        shift[index]=direction*step;center=evaluate(point)
        return direction*(48*(evaluate(point+shift)-center)-36*(evaluate(point+2*shift)-center)
            +16*(evaluate(point+3*shift)-center)-3*(evaluate(point+4*shift)-center))/(12*step)
    return (8*(evaluate(point+shift)-evaluate(point-shift))
            -(evaluate(point+2*shift)-evaluate(point-2*shift)))/(12*step)

