"""RooLandau conversion retaining ROOT's DENLAN/DISLAN conventions."""
import jax.numpy as jnp

from . import landau_math
from .graph import UnsupportedModelError


def extract_landau(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError('Landau forced numerical or named-range normalization')
    if hasattr(root,'getX'):
        fields=tuple(str(getattr(root,n)().GetName()) for n in ('getX','getMean','getSigma'))
    else:
        proxies={str(root.getProxy(i).GetName()):root.getProxy(i) for i in range(root.numProxies())}
        if set(proxies)!={'x','mean','sigma'}:
            raise UnsupportedModelError('Unrecognized Landau proxy layout without semantic accessors')
        fields=tuple(str(proxies[n].arg().GetName()) for n in ('x','mean','sigma'))
    return fields, {}


def landau_value(node, context):
    return landau_math.value(*context.values(node.inputs), jnp)


def landau_integral(node, context, axes):
    if axes!=(node.inputs[0],) or any(context.depends_on(n,axes[0]) for n in node.inputs[1:]):
        raise UnsupportedModelError(f'{node.name}: Landau integral requires independent shape parameters')
    return landau_math.integral(*context.bounds(axes[0]), *context.values(node.inputs[1:]), jnp)
