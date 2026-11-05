"""Native Romberg refinements of a fixed linear basis, folded at import.

Every refinement allowed by the native configuration is retained. This is not
a fixed-grid approximation: floating coefficients still select the native
adaptive stopping level, and AD differentiates the selected native expression.
"""
import numpy as np

from .romberg import extrapolate, RombergConfig


def product_coefficients(coefficients, groups, xp=np):
    """Coefficients of a tensor product of implicit-one linear polynomials."""
    result=xp.ones((*coefficients.shape[:-1],1),dtype=xp.float64);start=0
    for count in groups:
        block=xp.concatenate((xp.ones((*coefficients.shape[:-1],1),dtype=xp.float64),
                              coefficients[...,start:start+count]),axis=-1)
        result=(result[..., :, None]*block[..., None, :]).reshape(*coefficients.shape[:-1],-1)
        start+=count
    if start!=coefficients.shape[-1]:raise ValueError('Polynomial coefficient group size mismatch')
    return result


def build_table(basis, low, high, config=RombergConfig()):
    """Fold trapezoidal estimates for each basis function at every refinement."""
    if not np.isfinite(low+high) or high<=low:raise ValueError('Invalid basis integration range')
    width=high-low
    previous=.5*width*np.sum(basis(np.array([low,high])),axis=1)
    if previous.ndim!=1:raise ValueError('Basis must return a (function, coordinate) array')
    rows=[previous]
    for level in range(2,config.max_steps+1):
        count=1 << (level-2);total=np.zeros_like(previous)
        for start in range(0,count,64):
            x=low+(np.arange(start,min(start+64,count),dtype=np.float64)+.5)*(width/count)
            total+=np.sum(basis(x),axis=1)
        previous=.5*(previous+width*total/count);rows.append(previous)
    result=np.asarray(rows)
    if not np.all(np.isfinite(result)):raise ValueError('Nonfinite fixed numerical basis')
    return result


def integral(coefficients, table, width, config=RombergConfig(), xp=np):
    """Native value and selected (level, mode, converged); framework-neutral data."""
    if table.shape!=(config.max_steps,len(coefficients)):
        raise ValueError('Invalid linear integration table')
    def step(state,coefs,estimates):
        level,history,_,_=state
        raw=xp.dot(estimates[level],coefs);level=level+1
        history=xp.concatenate((history[1:],xp.reshape(raw,(1,))))
        estimate,error=extrapolate(history) if config.extrapolate else (raw,raw-history[-2])
        zero=(level>=config.min_steps_zero)&(raw<config.eps_abs/width)
        if config.fixed_steps:
            done=level==config.fixed_steps;mode=xp.asarray(0,dtype=xp.int32)
        else:
            done=(level>=5)&(xp.abs(error)<=xp.maximum(config.eps_abs,config.eps_rel*xp.abs(estimate)))
            mode=xp.asarray(xp.where(done,int(config.extrapolate),0),dtype=xp.int32)
        return level,history,xp.where(zero,2,mode),zero|done
    state=(xp.asarray(0,dtype=xp.int32),xp.zeros(5,dtype=xp.float64),xp.asarray(0,dtype=xp.int32),xp.asarray(False))
    if xp is np:
        while state[0]<config.max_steps and not state[3]:state=step(state,coefficients,table)
    else:
        import jax
        # Only the discrete refinement decision is frozen. The expression at
        # that refinement is evaluated below with differentiable coefficients.
        decision_coefs=jax.lax.stop_gradient(coefficients)
        decision_table=jax.lax.stop_gradient(table)
        state=jax.lax.while_loop(lambda s:(s[0]<config.max_steps)&~s[3],
            lambda s:step(s,decision_coefs,decision_table),state)
    level,_,mode,done=state
    padded=xp.concatenate((xp.zeros((4,table.shape[1]),dtype=xp.float64),table),axis=0)
    history=padded[xp.arange(5)+level-1]@coefficients
    estimate,_=extrapolate(history)
    return xp.where(mode==2,0.,xp.where(mode==1,estimate,history[-1])),level,mode,done
