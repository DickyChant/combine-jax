"""Minuit's bounded reported-error convention, separate from curvature errors.

See ROOT MnUserTransformation::Int2extError. Numerical comparisons use the
installed ROOT implementation, not a second transcription as the reference.
"""
import numpy as np


REPORTED_ERROR_STATUS={0:'reported',1:'fixed',2:'boundary_transform_unresolved',3:'invalid_input_or_covariance'}
SOURCE_URL='https://root.cern.ch/doc/backup_latest-stable/MnUserTransformation_8cxx_source.html'


def minuit_reported_errors(values,curvature_errors,lower,upper,*,xp=np):
    """Map physical covariance sigmas to Minuit's finite-step reported errors.

    Arrays are float64 vectors. Returns (errors, per-coordinate status). This
    leaves the covariance and sqrt(diagonal) untouched. It is not a profile or
    confidence-interval construction. At a transform's numerical boundary,
    external covariance cannot reliably reconstruct Minuit's internal step;
    return NaN/status 2 instead of inventing one. Equal limits with zero sigma
    are fixed coordinates and return zero/status 1.
    """
    arrays=tuple(xp.asarray(a) for a in (values,curvature_errors,lower,upper))
    values,sigma,lower,upper=arrays
    if values.ndim!=1 or any(a.shape!=values.shape or a.dtype!=np.float64 for a in arrays):
        raise ValueError('Reported errors require matching one-dimensional float64 vectors')
    lo=xp.isfinite(lower);hi=xp.isfinite(upper);both=lo&hi;single=lo^hi
    valid=(xp.isfinite(values)&xp.isfinite(sigma)&(sigma>=0)&(lower<=values)&(values<=upper))
    fixed=(lower==upper)&(sigma==0)&valid
    span=xp.where(both,upper,1.)-xp.where(both,lower,0.)
    safe_span=xp.where((span>0)&xp.isfinite(span),span,1.)
    fraction=(values-xp.where(both,lower,values))/safe_span
    argument=2*fraction-1
    eps2=2*np.sqrt(np.finfo(np.float64).eps)
    double_resolved=(argument*argument<=1-eps2)&(span>0)&xp.isfinite(span)
    q=xp.arcsin(xp.clip(xp.where(both,argument,0.),-1.,1.))
    derivative=.5*safe_span*xp.cos(q)
    internal=sigma/xp.where(derivative>0,derivative,1.)
    # The large-internal-step branch is part of Minuit's reporting convention.
    plus=.5*safe_span*(xp.sin(q+internal)-xp.sin(q))
    minus=.5*safe_span*(xp.sin(q-internal)-xp.sin(q))
    double_error=.5*(xp.where(internal>1.,safe_span,xp.abs(plus))+xp.abs(minus))
    distance=xp.where(lo,values-xp.where(lo,lower,values),xp.where(hi,upper,values)-values)
    distance=xp.where(single,distance,1.)
    # Keep the coordinate resolvable in the sqrt(1+q^2) transformation.
    single_resolved=(distance>eps2)&xp.isfinite(distance)
    u=xp.sqrt(xp.maximum(distance,0.))*xp.sqrt(xp.maximum(distance+2.,0.))
    jac=u/(distance+1.)
    step=sigma/xp.where(jac>0,jac,1.)
    base=xp.hypot(1.,u)
    single_error=.5*(xp.abs(xp.hypot(1.,u+step)-base)+xp.abs(xp.hypot(1.,u-step)-base))
    error=xp.where(both,double_error,xp.where(single,single_error,sigma))
    unresolved=(both&~double_resolved)|(single&~single_resolved)
    status=xp.where(~valid,3,xp.where(fixed,1,xp.where(unresolved,2,xp.where(xp.isfinite(error),0,3))))
    return xp.where(status==1,0.,xp.where(status==0,error,xp.nan)),status
