"""PyTorch analytic equations matching the certified NumPy/JAX conventions.

The formulas are a direct port of analytic_shapes.py (b380a911e8e659b8), with
PyTorch special functions instead of the JAX-specific imports and fusion barrier.
Kept separate to leave the accepted numerical oracle source unchanged. Both
value and first/second derivatives require independent cross-backend checks.
"""
import math

from .analytic_shapes import ARITIES


def _tail(low,high,alpha,power,xp):
    regular=xp.abs(power-1.)>1e-5;q=xp.where(regular,1.-power,1.)
    log_ratio=xp.log(high/low)
    integral=xp.where(regular,xp.exp(q*xp.log(low))*xp.expm1(q*log_ratio)/q,log_ratio)
    return xp.exp(power*xp.log(power/alpha)-.5*alpha*alpha)*integral


def analytic_operation(op,scalars,b,xp,parameters=None):
    _,kind,mode=op.split('_')
    if kind=='bernstein':
        if parameters is not None:return xp.einsum('hcr,hcr->hr',parameters,b['basis'])
        coefs=xp.where(b['coef_mask'],scalars[b['coefs']],0.)
        return xp.einsum('hc,hcr->hr',coefs,b['basis'])
    erf,erfc=xp.torch.special.erf,xp.torch.special.erfc
    # Eager PyTorch/CUDA graph replay preserves these separate kernels. No
    # expression fusion or torch.compile is used by this adapter.
    barrier=lambda x:x
    if parameters is None:
        coefficients=xp.where(b['coef_mask'],scalars[b['coefs']],0.)
        p=[coefficients[:,i,None] for i in range(ARITIES[kind])]
    else:p=[parameters[:,i,:] for i in range(ARITIES[kind])]
    x=b['data'];low=b['low'][:,None];high=b['high'][:,None];sign=b['sign'][:,None]
    if kind=='gaussian':
        center,sigma=p
        if mode=='value':return xp.exp(-.5*((x-center)/sigma)**2)
        scale=math.sqrt(2.)*xp.abs(sigma)
        result=math.sqrt(math.pi/2)*xp.abs(sigma)*(erf((high-center)/scale)-erf((low-center)/scale))
    elif kind=='exponential':
        slope=sign*p[0]
        if mode=='value':return xp.exp(slope*x)
        width=high-low;z=slope*width
        series=1+z*(.5+z*(1/6+z*(1/24+z*(1/120+z/720))))
        small=width*xp.exp(slope*low)*series;magnitude=xp.abs(slope)
        large=xp.exp(xp.maximum(slope*low,slope*high))*(-xp.expm1(-magnitude*width))/xp.where(magnitude>0,magnitude,1.)
        result=xp.where(xp.abs(z)<1e-4,small,large)
    elif kind=='power':
        exponent=p[0]
        if mode=='value':return xp.power(x,exponent)
        q=exponent+1.;positive=low>0;safe_low=xp.where(positive,low,1.)
        log_ratio=xp.log(high/safe_low);z=q*log_ratio
        series=1+z*(.5+z*(1/6+z*(1/24+z*(1/120+z/720))))
        regular=xp.expm1(z)/xp.where(q!=0,q,1.)
        result=xp.exp(q*xp.log(safe_low))*xp.where(xp.abs(z)<1e-4,log_ratio*series,regular)
        result=xp.where(positive,result,xp.where(q>0,xp.exp(q*xp.log(high))/xp.where(q>0,q,1.),xp.inf))
    elif kind=='doublecb':
        other,mean,width,a1,n1,a2,n2=p;abs1,abs2=xp.abs(a1),xp.abs(a2)
        valid=(width>0)&(a1!=0)&(a2!=0)&(n1>0)&(n2>0)
        if mode=='value':
            t=(sign*(x-other)-mean)/width
            base1=xp.where(t < -a1,n1/abs1-abs1-t,1.)
            base2=xp.where((t>=-a1)&(t>a2),n2/abs2-abs2+t,1.)
            left=xp.exp(n1*xp.log(n1/abs1)-.5*a1*a1-n1*xp.log(base1))
            right=xp.exp(n2*xp.log(n2/abs2)-.5*a2*a2-n2*xp.log(base2))
            result=xp.where(t < -a1,left,xp.where(t>a2,right,xp.exp(-.5*t*t)))
            return xp.where(valid&(base1>0)&(base2>0),result,xp.nan)
        lower=xp.where(sign>0,low-other,other-high);upper=xp.where(sign>0,high-other,other-low)
        t0,t1=(lower-mean)/width,(upper-mean)/width
        c0,c1=xp.clip(t0,-a1,a2),xp.clip(t1,-a1,a2)
        central=xp.where(-a1<a2,math.sqrt(math.pi/2)*(erf(c1/math.sqrt(2.))-erf(c0/math.sqrt(2.))),0.)
        b1,b2=n1/abs1-abs1,n2/abs2-abs2
        l0=xp.where(t0 < -a1,b1-xp.minimum(t1,-a1),1.);l1=xp.where(t0 < -a1,b1-xp.minimum(t0,-a1),1.)
        r0=xp.where(t1>a2,b2+xp.maximum(t0,a2),1.);r1=xp.where(t1>a2,b2+xp.maximum(t1,a2),1.)
        # v11 advertises independent tail intervals even for overlapping signed alphas.
        result=width*(_tail(l0,l1,abs1,n1,xp)+central+_tail(r0,r1,abs2,n2,xp))
        result=xp.where(valid&(l0>0)&(l1>0)&(r0>0)&(r1>0),result,xp.nan)
    elif kind=='erfexp':
        slope,offset,width=p;slope=xp.where(slope==0.,-1e-7,slope);width=xp.where(width<1e-2,1e-2,width)
        def root_erf(z):return barrier(xp.where(xp.abs(z)>1,1-barrier(erfc(z)),erf(z)))
        if mode=='value':return xp.exp(slope*x)*(1+root_erf((x-offset)/width))/2
        def primitive(v):
            return (xp.exp(slope*slope*width*width/4+slope*offset)*root_erf((2*v-slope*width*width-2*offset)/2/width)
                -xp.exp(slope*v)*root_erf((v-offset)/width)-xp.exp(slope*v))/(-2*slope)
        result=primitive(high)-primitive(low)
    else:raise ValueError(f'Unknown analytic shape {kind}')
    return xp.broadcast_to(result,x.shape)

