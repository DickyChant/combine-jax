"""RooCBShape formulas with explicit ROOT 6.30/6.40 tail conventions.

Source definitions: ROOT RooCBShape.cxx (v630) and RooFit/Detail/MathFuncs.h
(6.40). The near-n=1 integral changed; neither convention substitutes the
mathematically exact power-law integral for ROOT's advertised branch.
"""


def value(x, mean, sigma, alpha, power, xp, *, legacy):
    t=(x-mean)/sigma
    t=xp.where(alpha<0,-t,t)
    magnitude=xp.abs(alpha)
    in_tail=t < -magnitude
    # Inactive tail arithmetic must stay finite for reverse-mode AD.
    a=xp.where(in_tail,magnitude,1.);n=xp.where(in_tail,power,2.)
    r=n/a;base=xp.where(in_tail,r-a-t,1.)
    if legacy:
        tail=xp.power(r,n)*xp.exp(-.5*a*a)/xp.power(base,n)
    else:
        tail=xp.exp(-.5*a*a)*xp.power(r/base,n)
    return xp.where(in_tail,tail,xp.exp(-.5*t*t))


def integral(low, high, mean, sigma, alpha, power, xp, erf, *, legacy):
    sig=xp.abs(sigma);t0=(low-mean)/sig;t1=(high-mean)/sig
    lower=xp.where(alpha<0,-t1,t0);upper=xp.where(alpha<0,-t0,t1)
    magnitude=xp.abs(alpha);has_tail=lower < -magnitude;tail_only=upper <= -magnitude
    a=xp.where(has_tail,magnitude,1.);n=xp.where(has_tail,power,2.)
    r=n/a;b=r-a
    left=xp.where(has_tail,b-lower,r+1.)
    right=xp.where(has_tail & tail_only,b-upper,r)
    logarithmic=xp.abs(n-1.) < 1e-5
    q=xp.where(logarithmic,1.,1.-n)
    log_left=xp.log(left);log_right=xp.log(right)
    if legacy:
        factor=xp.power(r,n)*xp.exp(-.5*a*a)
        regular=factor*sig/q*(1./xp.power(left,n-1.)-1./xp.power(right,n-1.))
        near=factor*sig*(log_left-log_right)
    else:
        factor=r*xp.exp(-.5*a*a)
        regular=factor*sig/q*(xp.power(r/left,n-1.)-xp.where(tail_only,xp.power(r/right,n-1.),1.))
        near=factor*xp.power(r,n-1.)*sig*(log_left-log_right+.5*(1.-n)*(log_left**2-log_right**2))
    tail=xp.where(has_tail,xp.where(logarithmic,near,regular),0.)
    def approx_erf(z):
        return xp.where(z>5.,1.,xp.where(z < -5.,-1.,erf(z)))
    central_low=xp.where(has_tail,-magnitude,lower)
    central=xp.where(tail_only,0.,sig*1.2533141373*(approx_erf(upper/1.4142135624)-approx_erf(central_low/1.4142135624)))
    result=tail+central
    return xp.where(result==0.,1e-300,result)
