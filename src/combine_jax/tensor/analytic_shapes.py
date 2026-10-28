"""Batched analytic queries with scalar shape parameters and exact integrals.

No framework objects enter the importer or its arrays. The formulas retain the
certified backend's Combine conventions, including signed DoubleCB tails.
"""
import math

import numpy as np

from ..graph import UnsupportedModelError


ANALYTIC_CLASSES=frozenset(('RooGaussian','RooExponential','RooPower','RooDoubleCB','RooErfExpPdf','RooBernstein'))
ARITIES={'gaussian':2,'exponential':1,'power':1,'doublecb':7,'erfexp':3}
ANALYTIC_OPERATIONS=frozenset(f'analytic_{kind}_{mode}' for kind in (*ARITIES,'bernstein') for mode in ('value','integral'))
GRID_ANALYTIC_OPERATIONS=frozenset(op.replace('analytic_','gridanalytic_',1) for op in ANALYTIC_OPERATIONS)


def pack_analytic_query(name,node,mode,axes,coordinates,nodes,lookup,depends,rows,*,allow_grid=False):
    """Resolve exact analytic geometry; grid parameters use the caller's typed program."""
    cls,inputs,a=node
    if cls not in ANALYTIC_CLASSES or mode not in ('value','integral'):
        raise UnsupportedModelError(f'{name}: unsupported analytic query {cls}/{mode}')
    if cls=='RooExponential' and a['sign']<0:
        raise UnsupportedModelError(f'{name}: negated exponential differs between RooFit and v11 optimized CachingNLL; unsupported')
    if cls=='RooBernstein':
        axis=a['observable_name'];names=a['coefs'];degree=len(names)-1
        if (mode=='integral' and axes!=(axis,)) or axis not in coordinates:
            raise UnsupportedModelError(f'{name}: Bernstein query requires its direct observable')
        if any((n not in lookup and not allow_grid) or depends(n,axis) for n in names):
            raise UnsupportedModelError(f'{name}: observable/profile-dependent Bernstein coefficient')
        low,high=a['reference_range'];width=high-low
        if degree<0 or width<=0 or not math.isfinite(low+high):raise UnsupportedModelError(f'{name}: invalid Bernstein basis')
        x=(np.asarray(coordinates[axis])-low)/width
        if mode=='value':basis=np.array([math.comb(degree,i)*x**i*(1-x)**(degree-i) for i in range(degree+1)])
        else:
            tmin=(nodes[axis][2]['minimum']-low)/width;tmax=(nodes[axis][2]['maximum']-low)/width
            weights=[sum((-1.)**(j-i)*math.comb(degree,j)*math.comb(j,i)*(tmax**(j+1)-tmin**(j+1))/(j+1)
                for j in range(i,degree+1))*width for i in range(degree+1)]
            basis=np.broadcast_to(np.array(weights)[:,None],(degree+1,rows)).copy()
        fields=dict(coefs=np.array([lookup.get(n,0) for n in names],dtype=np.int32),coef_mask=np.ones(len(names),dtype=bool),basis=basis)
        return f'analytic_bernstein_{mode}',fields,tuple(names)
    candidates=(tuple(n for n in a['fields'][:2] if n) if cls=='RooDoubleCB'
        else inputs[:2] if cls in ('RooGaussian','RooExponential') else inputs[:1])
    if mode=='integral':
        if len(axes)!=1 or axes[0] not in candidates:
            raise UnsupportedModelError(f'{name}: analytic integral requires one direct argument, got {axes}')
        axis=axes[0]
    else:
        observed=[n for n in candidates if n in coordinates]
        if not observed or (len(observed)!=1 and not allow_grid):
            raise UnsupportedModelError(f'{name}: analytic value requires one direct data argument, got {observed}')
        axis=axes[0] if len(axes)==1 and axes[0] in observed else observed[0]
    sign=1.
    if cls=='RooDoubleCB':
        fields=a['fields'];other=fields[1] if axis==fields[0] else fields[0]
        names=(other,*fields[2:]);kind='doublecb';sign=1. if axis==fields[0] else -1.
    elif cls in ('RooGaussian','RooExponential'):
        other=inputs[1] if axis==inputs[0] else inputs[0]
        names=(other,*inputs[2:]);kind='gaussian' if cls=='RooGaussian' else 'exponential'
        if cls=='RooExponential':sign=float(a['sign'])
    else:
        names=inputs[1:];kind='power' if cls=='RooPower' else 'erfexp'
    if candidates.count(axis)!=1 or any(n and depends(n,axis) for n in names):
        raise UnsupportedModelError(f'{name}: analytic shape parameters depend on integration/data variable {axis}')
    missing={n for n in names if n and n not in lookup}
    if missing and not allow_grid:
        raise UnsupportedModelError(f'{name}: observable/profile-dependent or missing analytic shape parameters: {sorted(missing)}')
    if nodes[axis][0]!='RooRealVar':raise UnsupportedModelError(f'{name}: analytic axis is not a RooRealVar')
    low,high=(float(nodes[axis][2][n]) for n in ('minimum','maximum'))
    if not math.isfinite(low+high) or high<=low or (kind=='power' and low<0):
        raise UnsupportedModelError(f'{name}: invalid finite analytic integration range [{low}, {high}]')
    if axis not in coordinates or np.shape(coordinates[axis])!=(rows,):
        raise UnsupportedModelError(f'{name}: missing analytic coordinates {axis}')
    fields=dict(coefs=np.array([lookup.get(n,0) for n in names],dtype=np.int32),
        coef_mask=np.array([bool(n) for n in names],dtype=bool),data=np.asarray(coordinates[axis],dtype=np.float64),
        low=np.array(low),high=np.array(high),sign=np.array(sign))
    return f'analytic_{kind}_{mode}',fields,tuple(names)


def _tail(low,high,alpha,power,xp):
    regular=xp.abs(power-1.)>1e-5;q=xp.where(regular,1.-power,1.)
    log_ratio=xp.log(high/low)
    integral=xp.where(regular,xp.exp(q*xp.log(low))*xp.expm1(q*log_ratio)/q,log_ratio)
    return xp.exp(power*xp.log(power/alpha)-.5*alpha*alpha)*integral


def analytic_operation(op,scalars,b,xp=np,parameters=None):
    _,kind,mode=op.split('_')
    if kind=='bernstein':
        if parameters is not None:return xp.einsum('hcr,hcr->hr',parameters,b['basis'])
        coefs=xp.where(b['coef_mask'],scalars[b['coefs']],0.)
        return xp.einsum('hc,hcr->hr',coefs,b['basis'])
    if xp is np:
        from scipy.special import erf,erfc
        barrier=lambda x:x
    else:
        from jax.scipy.special import erf,erfc
        from jax.lax import optimization_barrier as barrier
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


def validate_analytic_fields(op,b,rows):
    if op not in ANALYTIC_OPERATIONS|GRID_ANALYTIC_OPERATIONS:raise ValueError(f'Unknown analytic operation {op}')
    grid=op in GRID_ANALYTIC_OPERATIONS;layout=b['children'] if grid else b['coefs']
    kind=op.split('_')[1];batch=len(layout)
    if kind=='bernstein':
        if b['basis'].shape!=(batch,layout.shape[1],rows):raise ValueError('Invalid Bernstein basis layout')
        return
    if layout.shape!=((batch,ARITIES[kind],rows) if grid else (batch,ARITIES[kind])) or b['data'].shape!=(batch,rows):
        raise ValueError('Invalid analytic parameter/data layout')
    if any(b[n].shape!=(batch,) for n in ('low','high','sign')):
        raise ValueError('Invalid analytic geometry layout')
    if np.any(b['high']<=b['low']) or np.any(np.abs(b['sign'])!=1.) or (kind=='power' and np.any(b['low']<0)):
        raise ValueError('Invalid analytic integration geometry')
