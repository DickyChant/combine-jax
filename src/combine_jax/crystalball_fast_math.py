"""Native RooDoubleCBFast formulas, shared by NumPy and JAX evaluators.

Preserves the source's finite-range, logarithmic and |t|>1000 branches. The
intermediate fast_pow calls are deliberate: do not silently repair a native
nonfinite result with an algebraically stabilized but different expression.
"""
import math


def fast_pow(base, power, xp):
    positive = base > 0
    regular = xp.exp(power*xp.log(xp.where(positive, base, 1.)))
    return xp.where(positive, regular, xp.where((base == 0) & (power > 0), 0., xp.nan))


def value(x, mean, width, a1, n1, a2, n2, xp):
    t = (x-mean)/width
    left = t <= -a1
    right = (~left) & (t >= a2)
    base1 = xp.where(left, 1.-a1/n1*(a1+t), 1.)
    base2 = xp.where(right, 1.-a2/n2*(a2-t), 1.)
    return xp.where(left, xp.exp(-.5*a1*a1)*fast_pow(base1, -n1, xp),
                    xp.where(right, xp.exp(-.5*a2*a2)*fast_pow(base2, -n2, xp), xp.exp(-.5*t*t)))


def _left_tail(t0, t1, width, alpha, n, full, xp):
    high = xp.minimum(t1, -alpha)
    active = t0 < high
    full = full & (n-1. > 1e-5)
    regular = xp.abs(n-1.) > 1e-5
    # Finite placeholders only in branches the source does not evaluate.
    work_n = xp.where(active & regular & ~full, n, 2.)
    ratio = work_n/xp.abs(alpha)
    factor = fast_pow(ratio, -work_n/(work_n-1.), xp)
    base_high = xp.where(active & regular & ~full, ratio-alpha-high, ratio)
    base_low = xp.where(active & regular & ~full, ratio-alpha-t0, ratio)
    upper = xp.where(t1 > -alpha, ratio, fast_pow(factor*base_high, 1.-work_n, xp))
    lower = xp.where(t0 < -1000., 0., fast_pow(factor*base_low, 1.-work_n, xp))
    ordinary = width*xp.exp(-.5*alpha*alpha)/(work_n-1.)*(upper-lower)
    log_n = xp.where(active & ~regular & ~full, n, 2.)
    log_ratio = log_n/xp.abs(alpha)
    base_high = xp.where(active & ~regular & ~full, log_ratio-xp.abs(alpha)-high, 1.)
    base_low = xp.where(active & ~regular & ~full, log_ratio-xp.abs(alpha)-t0, 1.)
    logarithmic = fast_pow(log_ratio, log_n, xp)*xp.exp(-.5*alpha*alpha)*width*(xp.log(base_low)-xp.log(base_high))
    complete = width*xp.exp(-.5*alpha*alpha)*n/(alpha*xp.where(full, n-1., 1.))
    return xp.where(full, complete, xp.where(active, xp.where(regular, ordinary, logarithmic), 0.))


def integral(low, high, mean, width, a1, n1, a2, n2, xp, erf):
    t0, t1 = (low-mean)/width, (high-mean)/width
    full = (t0 < -1000.) & (t1 > 1000.)
    central_low = xp.maximum(low, mean-a1*width)
    central_high = xp.minimum(high, mean+a2*width)
    central = xp.where(central_low < central_high, math.sqrt(math.pi/2)*width*
        (erf((central_high-mean)/width/math.sqrt(2.))-erf((central_low-mean)/width/math.sqrt(2.))), 0.)
    return (_left_tail(t0,t1,width,a1,n1,full,xp) + central
            + _left_tail(-t1,-t0,width,a2,n2,full,xp))
