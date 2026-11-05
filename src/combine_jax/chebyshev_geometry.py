"""Framework-neutral, exact Chebyshev basis and antiderivative geometry."""
import numpy as np


def basis(degree, x):
    x=np.asarray(x,dtype=np.float64)
    previous,current=np.ones_like(x),x
    terms=[]
    for _ in range(degree):
        terms.append(current)
        previous,current=current,2*x*current-previous
    return np.array(terms,dtype=np.float64).reshape((degree,*x.shape))


def integral_basis(degree, low, high):
    if degree==0:return np.empty(0)
    a=np.r_[1.,basis(degree+1,low)]
    b=np.r_[1.,basis(degree+1,high)]
    return np.array([.5*(high+low)*(high-low),
        *[.5*((b[n+1]-a[n+1])/(n+1)-(b[n-1]-a[n-1])/(n-1)) for n in range(2,degree+1)]])
