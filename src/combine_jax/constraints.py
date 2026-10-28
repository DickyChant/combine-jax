"""RooFit Poisson terms and Combine's distinct fast-constraint convention."""

import math

import jax.numpy as jnp
from jax.scipy.special import gammainc, gammaincc, gammaln, xlogy

from .graph import UnsupportedModelError


def extract_poisson(root):
    return (str(root.getX().GetName()), str(root.getMean().GetName())), {
        "no_rounding": bool(root.getNoRounding()),
        "protect_negative": bool(root.getProtectNegativeMean()),
    }


def poisson_log_value(node, context):
    x, mean = context.values(node.inputs)
    if not node.attrs["no_rounding"]:
        x = jnp.floor(x)
    value = xlogy(x, jnp.where(x == 0, 1., mean)) - mean - gammaln(x+1)
    return jnp.where(x < 0, -jnp.inf, jnp.where(mean < 0, jnp.nan, value))


def poisson_integral(node, context, axes):
    if len(axes) != 1 or axes[0] not in node.inputs:
        raise UnsupportedModelError(f"{node.name}: Poisson integral over {axes}")
    variable = axes[0]
    other = node.inputs[1] if variable == node.inputs[0] else node.inputs[0]
    if context.depends_on(other, variable):
        raise UnsupportedModelError(f"{node.name}: Poisson arguments depend on integration variable")
    low, high = context.bounds(variable)
    if variable == node.inputs[1]:
        x = context.value(node.inputs[0])
        if not node.attrs["no_rounding"]:
            x = jnp.floor(x)
        return gammainc(x+1, high) - gammainc(x+1, max(0., low))
    mean = context.value(node.inputs[1])
    low = max(0., low)
    if high < low:
        return jnp.asarray(0.)
    imin, imax = int(low), min(int(high + 1) if math.isfinite(high) else 2**32-1, 2**32-1)
    if imin == 0:
        return gammaincc(imax, mean)
    return jnp.where(imin <= mean,
                     gammaincc(imax, mean) - gammaincc(imin, mean),
                     gammainc(imin, mean) - gammainc(imax, mean))


def fast_poisson_log_value(node, context):
    observed, mean = context.values(node.inputs)
    # Source: SimplePoissonConstraint::getLogValFast. This deliberately retains
    # its large-count Gaussian branch and its sign on log(mean), as written.
    safe_mean = jnp.where(mean > 0, mean, 1.)
    exact = xlogy(observed, jnp.where(jnp.abs(observed) < 1e-10, 1., mean)) - mean - gammaln(observed+1)
    large = .5*jnp.log(safe_mean) - .5*(observed-mean)**2/safe_mean
    empty = jnp.where(jnp.abs(mean) < 1e-10, 0., -mean)
    return jnp.where(jnp.abs(observed) < 1e-10, empty,
                     jnp.where(observed < 1e6, exact, large))
