"""Pointwise scan differences and explicitly linear, bracketed grid intervals."""
import numpy as np
import math


def check_global_baseline(fits):
    """A conditional point below the unrestricted fit disproves its minimum.

    Use the existing limit profiler's 64-ULP roundoff allowance; do not shift
    the scan to hide a failed denominator. This necessary check does not certify
    global optimality, even when no lower conditional point has been found.
    """
    if not fits or not math.isfinite(float(fits[0]['value'])):
        return dict(valid=False,reason='Missing or nonfinite unrestricted NLL',lower_points=[])
    global_value=float(fits[0]['value']); lower=[]; nonfinite=[]
    for number, row in enumerate(fits[1:],1):
        value=float(row['value'])
        if not math.isfinite(value):
            nonfinite.append(number);continue
        allowance=64*math.ulp(max(abs(global_value),abs(value),1.))
        if value<global_value-allowance:
            lower.append(dict(fit_index=number,task=row.get('task'),fixed=row.get('fixed',{}),
                value=value,deltaNLL=value-global_value,roundoff_allowance=allowance))
    return dict(valid=not lower and not nonfinite,global_value=global_value,
                lower_points=lower,nonfinite_points=nonfinite,statistic_shift_applied=False,
                criterion='No conditional NLL below unrestricted NLL by more than 64 ULP; necessary, not sufficient for a global minimum')


def checked_points(points):
    values=np.asarray(points,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=2 or len(values)<2 or not np.isfinite(values).all():
        raise ValueError('A finite (x, test statistic) scan with at least two points is required')
    values=values[np.argsort(values[:,0],kind='stable')]
    if np.any(np.diff(values[:,0])<=0):raise ValueError('Scan coordinates must be unique')
    return values


def linear_intervals(points, threshold):
    """All sublevel components on the available grid; never extrapolate a bound.

    These are piecewise-linear estimates, not native profile interval results.
    A missing crossing has a null bound and is explicitly unbracketed. Isolated
    points exactly at the threshold are retained as zero-width components.
    """
    values=checked_points(points)
    if not np.isfinite(threshold):raise ValueError('Threshold must be finite')
    segments=[]
    for (x0,y0),(x1,y1) in zip(values[:-1],values[1:]):
        if min(y0,y1)>threshold:continue
        low,high=float(x0),float(x1)
        if (y0>threshold)!=(y1>threshold):
            crossing=float(x0+(threshold-y0)*(x1-x0)/(y1-y0))
            if y0>threshold:low=crossing
            else:high=crossing
        if segments and low<=segments[-1][1]:segments[-1][1]=max(high,segments[-1][1])
        else:segments.append([low,high])
    result=[]
    for low,high in segments:
        left_open=low==values[0,0] and values[0,1]<threshold
        right_open=high==values[-1,0] and values[-1,1]<threshold
        result.append(dict(lower=None if left_open else low,upper=None if right_open else high,
            lower_bracketed=not left_open,upper_bracketed=not right_open))
    return dict(threshold=float(threshold),method='piecewise linear on stored scan points; no extrapolation',
                intervals=result,grid_range=values[[0,-1],0].tolist())


def compare_scan_points(points, reference):
    """Require identical coordinates and compare raw statistics, without shifts."""
    actual=checked_points(points);expected=checked_points(reference)
    if actual.shape!=expected.shape or not np.array_equal(actual[:,0],expected[:,0]):
        raise ValueError('Pointwise comparison requires exactly the same coordinates')
    difference=actual[:,1]-expected[:,1]
    return dict(point_count=len(actual),max_absolute_difference=float(np.max(np.abs(difference))),
                statistic_shift_applied=False,points=[dict(x=float(x),value=float(y),reference=float(r),
                    difference=float(d)) for (x,y),(_,r),d in zip(actual,expected,difference)])
