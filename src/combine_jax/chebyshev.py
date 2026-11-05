"""RooChebychev's implicit constant and fixed reference-range polynomial."""
import math

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .root_io import stored_member


def arguments(root):
    """Use the semantic members; retain strict extraction for older ROOT APIs."""
    if hasattr(root,'x') and hasattr(root,'coefList'):
        return root.x(),tuple(root.coefList())
    # ROOT 6.30 lacks the later public x()/coefList() accessors.
    proxies = {str(root.getProxy(i).GetName()): root.getProxy(i) for i in range(root.numProxies())}
    if set(proxies) != {'x','coefList'}:
        raise UnsupportedModelError('Unrecognized Chebychev proxy layout without semantic accessors')
    return proxies['x'].arg(),tuple(proxies['coefList'])


def extract_chebyshev(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError('Chebychev forced numerical or named-range normalization')
    x,coefficients = arguments(root)
    if x.ClassName() != 'RooRealVar':
        raise UnsupportedModelError('Chebychev requires a direct real observable')
    stored = stored_member(root, '_refRangeName')
    if stored is not None and (not isinstance(stored,dict) or not isinstance(stored.get('fName'),str)):
        raise UnsupportedModelError('Invalid Chebychev reference range')
    reference = stored['fName'] if stored else ''
    low,high = (float(x.getMin(reference)),float(x.getMax(reference))) if reference else (float(x.getMin()),float(x.getMax()))
    if not math.isfinite(low+high) or high<=low:
        raise UnsupportedModelError('Chebychev requires a finite positive reference interval')
    coefs = tuple(str(c.GetName()) for c in coefficients)
    axis = str(x.GetName())
    return (axis,*coefs),dict(observable_name=axis,coefs=coefs,reference_range=(low,high))


def chebyshev_value(node, context):
    a = node.attrs;low,high = a['reference_range']
    x = (context.value(a['observable_name'])-.5*(high+low))/(.5*(high-low))
    result,previous,current = jnp.ones_like(x),jnp.ones_like(x),x
    for coefficient in context.values(a['coefs']):
        result = result+current*coefficient
        previous,current = current,2*x*current-previous
    return result


def chebyshev_integral(node, context, axes):
    a=node.attrs
    if axes!=(a['observable_name'],) or any(context.depends_on(n,axes[0]) for n in a['coefs']):
        raise UnsupportedModelError(f'{node.name}: Chebychev integral requires independent coefficients and its direct axis')
    low,high=a['reference_range'];half=.5*(high-low);mid=.5*(high+low)
    lo,hi=context.bounds(axes[0]);lo,hi=(lo-mid)/half,(hi-mid)/half
    # This is the exact polynomial antiderivative; all geometry is fixed at import.
    from .chebyshev_geometry import integral_basis
    weights=integral_basis(len(a['coefs']),lo,hi)
    return half*((hi-lo)+sum(w*context.value(n) for w,n in zip(weights,a['coefs'],strict=True)))
