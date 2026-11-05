"""Pure JAX native Romberg normalization with streamed derivative sums."""
from functools import partial

import jax
import jax.numpy as jnp

from .romberg import RombergConfig, extrapolate


def _trapezoid(function,p,level,previous):
    low,high=p[-2:];width=high-low
    def endpoints(_):
        return .5*width*jnp.sum(function(p[:-2],jnp.stack((low,high))))
    def interior(_):
        count=jnp.left_shift(1,jnp.maximum(level-2,0))
        def accumulate(start,total):
            index=start*64+jnp.arange(64)
            # Masked samples stay inside the valid domain, including their AD.
            x=low+(jnp.minimum(index,count-1).astype(jnp.float64)+.5)*(width/count)
            return total+jnp.sum(jnp.where(index<count,function(p[:-2],x),0.))
        total=jax.lax.fori_loop(0,(count+63)//64,accumulate,jnp.asarray(0.,dtype=jnp.float64))
        return .5*(previous+width*total/count)
    return jax.lax.cond(level==1,endpoints,interior,operand=None)


def solve(function,p,config=RombergConfig()):
    """Native estimate plus count/mode/convergence diagnostics, all on device."""
    def step(state):
        level,history,_,_,_=state;level=level+1
        raw=_trapezoid(function,p,level,history[-1])
        history=jnp.concatenate((history[1:],raw[None]))
        estimate,error=extrapolate(history) if config.extrapolate else (raw,raw-history[-2])
        zero=(level>=config.min_steps_zero)&(raw<config.eps_abs/(p[-1]-p[-2]))
        if config.fixed_steps:
            converged=level==config.fixed_steps;mode=jnp.asarray(0,jnp.int32);value=raw
        else:
            converged=(level>=5)&(jnp.abs(error)<=jnp.maximum(config.eps_abs,config.eps_rel*jnp.abs(estimate)))
            mode=jnp.where(converged,int(config.extrapolate),0).astype(jnp.int32);value=jnp.where(converged,estimate,raw)
        return level,history,jnp.where(zero,0.,value),jnp.where(zero,2,mode),zero|converged
    initial=(jnp.asarray(0,jnp.int32),jnp.zeros(5,dtype=jnp.float64),jnp.asarray(0.,jnp.float64),
             jnp.asarray(0,jnp.int32),jnp.asarray(False))
    level,_,value,mode,done=jax.lax.while_loop(lambda s:(s[0]<config.max_steps)&~s[4],step,initial)
    return value,level,mode,done


def _at_selected_refinement(function,p,level,mode):
    def step(state):
        i,history=state;i=i+1
        raw=_trapezoid(function,p,i,history[-1])
        return i,jnp.concatenate((history[1:],raw[None]))
    _,history=jax.lax.while_loop(lambda s:s[0]<level,step,(jnp.asarray(0,jnp.int32),jnp.zeros(5,jnp.float64)))
    value,_=extrapolate(history)
    return jnp.where(mode==2,0.,jnp.where(mode==1,value,history[-1]))


@partial(jax.custom_jvp,nondiff_argnums=(0,2))
def _integral(function,p,config):
    return solve(function,p,config)[0]


@_integral.defjvp
def _integral_jvp(function,config,primals,tangents):
    (p,),(p_dot,)=primals,tangents
    value,level,mode,_=solve(function,p,config)
    # Differentiate the selected native numerical expression. Adaptivity is a
    # discrete decision; its local derivative is zero. Accumulate the small
    # parameter Jacobian before contracting the tangent, so reverse AD never
    # needs a per-sample residual tape. Forward-over-reverse Hessians/HVPs remain
    # differentiable through the streamed Jacobian accumulation.
    selected=lambda v:_at_selected_refinement(function,v,jax.lax.stop_gradient(level),jax.lax.stop_gradient(mode))
    derivative=jax.jacfwd(selected)(p)
    return value,jnp.dot(derivative,p_dot)


def integrate(function,parameters,low,high,config=RombergConfig()):
    p=jnp.concatenate((jnp.asarray(parameters,dtype=jnp.float64),jnp.asarray([low,high],dtype=jnp.float64)))
    return _integral(function,p,config)
