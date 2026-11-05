"""Framework-neutral literal Combine RooModZPdf formula, including Bernstein factor."""
import math


def value(x,parameters,low,high,xp):
    a,b,c,mean,width=parameters[:5]
    base=xp.exp(a*x+b*x*x)/(xp.power(x*x-mean*mean,c)+xp.power(width*mean,c))
    degree=len(parameters)-5
    if degree==0:return base
    coordinate=(x-low)/(high-low)
    bernstein=1.;coefficient_sum=0.
    for i in range(degree):
        coefficient=parameters[5+i]
        coefficient_sum-=coefficient
        bernstein+=(degree+1.)*math.comb(degree,i)*coordinate**(degree-i)*(1.-coordinate)**i*coefficient
    bernstein+=coefficient_sum*(1.-coordinate)**degree
    return base*bernstein
