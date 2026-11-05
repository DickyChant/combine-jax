"""Stored spline polynomial evaluation without importing a numerical framework."""


def value(x, attrs, xp):
    knots=xp.asarray(attrs['knots'])
    if attrs['algorithm']=='POLYNOMIAL':
        u=(x-knots[0])/(knots[-1]-knots[0])
        result=xp.asarray(attrs['newton'][-1])
        for index in range(len(attrs['newton'])-2,-1,-1):
            result=result*(u-attrs['scaled_knots'][index])+attrs['newton'][index]
    else:
        index=xp.clip(xp.searchsorted(knots,x,side='right')-1,0,len(knots)-2)
        delta=x-knots[index]
        c0,c1,c2,c3=xp.asarray(attrs['coefficients'])[:,index]
        result=((c3*delta+c2)*delta+c1)*delta+c0
    return xp.where((x>=knots[0])&(x<=knots[-1]),result,xp.nan)
