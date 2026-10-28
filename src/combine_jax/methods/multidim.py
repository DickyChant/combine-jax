"""Grid scans and profile-error impacts over a converged profile-fit driver."""
import math

import numpy as np
from scipy.optimize import brentq
from scipy.special import chdtrc, chdtri


DELTA68 = .5*float(chdtri(1, .32))


def grid_points(pois, ranges, counts, first=0, last=None):
    """Native alignEdges=1 grid, with the last axis varying fastest."""
    if not pois or len(set(pois)) != len(pois) or len(counts) != len(pois):
        raise ValueError('Distinct grid POIs and one count per axis are required')
    axes = []
    for name, count in zip(pois, counts, strict=True):
        if name not in ranges or type(count) is not int or count < 2:
            raise ValueError('Each aligned grid axis requires bounds and at least two points')
        lo, hi = ranges[name]
        if not math.isfinite(lo+hi) or lo >= hi:
            raise ValueError('Grid bounds must be finite and increasing')
        axes.append(np.linspace(lo, hi, count))
    size = math.prod(counts)
    last = size-1 if last is None else last
    if not 0 <= first <= last < size:
        raise ValueError('Invalid first/last grid point')
    return [dict(zip(pois, (float(axes[d][i]) for d, i in
        enumerate(np.unravel_index(index, tuple(counts)))), strict=True))
        for index in range(first, last+1)]


def scan(driver, pois, points, *, nll0):
    """Standard best-fit and grid rows; never silently clip a negative deltaNLL."""
    if driver.global_point is None:
        driver.global_minimum()
    indices = [driver.names.index(n) for n in pois]
    minimum = np.asarray(driver.global_point)[indices]
    baseline = driver.global_value
    common = dict(nll0=float(nll0), nll=baseline-driver.initial_nll)
    rows = [dict(parameters=dict(zip(pois, map(float, minimum), strict=True)),
                 deltaNLL=0., quantile=-1., **common)]
    for point in points:
        if set(point) != set(pois):
            raise ValueError('Every scan point must fix exactly the requested POIs')
        _, record = driver.profile(point)
        delta = record['value']-baseline
        if delta < -.01:
            raise RuntimeError('Conditional fit improves global baseline by more than 0.01: '+repr(record))
        rows.append(dict(parameters=dict(point), deltaNLL=delta,
                         quantile=float(chdtrc(len(pois), max(0., 2*delta))), **common))
    return rows


def interval(driver, name, *, delta_nll=.5, relative_accuracy=1e-5, absolute_accuracy=1e-6):
    """Profile interval (native Minos/robustFit Up, not a Hessian surrogate).

    Bracket each side from the global estimate, respecting the imported bounds.
    Every conditional fit uses the same global start. Boundary truncation is
    explicit. Brent's method supplies a numerical crossing; its path differs
    from Minos/robustFit and must be validated against native interval outputs.
    """
    if driver.global_point is None:
        driver.global_minimum()
    i = driver.names.index(name)
    center = float(driver.global_point[i])
    info = driver.info.arrays
    scale = float(info['error'][i])
    if scale <= 0:
        scale = .1*max(1., abs(center))
    if not math.isfinite(delta_nll) or delta_nll <= 0:
        raise ValueError('Positive finite interval threshold required')
    target = driver.global_value+delta_nll
    trace = []
    def evaluate(value):
        _, record = driver.profile({name: float(value)})
        trace.append(dict(value=float(value), nll=record['value'], record=len(driver.records)-1))
        return record['value']-target
    ends = []
    for direction, bound in [(-1, float(info['lower'][i])), (1, float(info['upper'][i]))]:
        if center == bound:
            ends.append(dict(value=center, status='at_bound', delta_nll=0.))
            continue
        step = scale
        for _ in range(60):
            edge = min(center+step, bound) if direction > 0 else max(center-step, bound)
            value = evaluate(edge)
            if value >= 0:
                lo, hi = sorted([center, edge])
                def crossing(x):
                    return -delta_nll if x == center else evaluate(x)
                root = float(brentq(crossing, lo, hi, xtol=absolute_accuracy,
                                   rtol=relative_accuracy, maxiter=100))
                residual = evaluate(root)
                if abs(residual) > .005:
                    raise RuntimeError('Profile interval crossing is unresolved: '+str(residual))
                ends.append(dict(value=root, status='crossing', delta_nll=residual+delta_nll))
                break
            if edge == bound:
                ends.append(dict(value=edge, status='bound_truncated', delta_nll=value+delta_nll))
                break
            step *= 2
        else:
            raise RuntimeError('Could not bracket profile interval for '+name)
    return dict(parameter=name, center=center, lower=ends[0], upper=ends[1],
                delta_nll=delta_nll, trace=trace, algorithm='brentq_profile',
                absolute_accuracy=absolute_accuracy, relative_accuracy=relative_accuracy)


def impacts(driver, poi, parameters, *, nll0, delta_nll=.5):
    """Many nuisance impacts sharing one global and one conditional executable."""
    if len(set(parameters)) != len(parameters):
        raise ValueError('Duplicate impact parameters')
    if driver.global_point is None:
        driver.global_minimum()
    global_point = np.asarray(driver.global_point)
    output = {}
    for name in parameters:
        names = [poi] if name == poi else [poi, name]
        ids = [driver.names.index(n) for n in names]
        common = dict(nll0=float(nll0), nll=driver.global_value-driver.initial_nll)
        rows = [dict(parameters={n: float(global_point[i]) for n, i in zip(names, ids)},
                     deltaNLL=0., quantile=-1., **common)]
        error = interval(driver, name, delta_nll=delta_nll)
        for side, quantile in [('lower', -.32 if name == poi else .32), ('upper', .32)]:
            point, record = driver.profile({name: error[side]['value']})
            point = np.asarray(point)
            # Native doSingles/doImpact never assign deltaNLL_ after the global
            # row. Keep the actual profile NLL in the receipt, not this branch.
            delta = 0.
            rows.append(dict(parameters={n: float(point[i]) for n, i in zip(names, ids)},
                             deltaNLL=delta, quantile=quantile, **common))
        output[name] = dict(rows=rows, interval=error)
    return output
