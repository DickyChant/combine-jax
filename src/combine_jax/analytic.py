"""Finite-range exponential and Bernstein PDF families."""

import math

import jax.numpy as jnp

from .graph import UnsupportedModelError


def extract_exponential(root):
    return (str(root.variable().GetName()), str(root.coefficient().GetName())), {
        "sign": -1. if root.negateCoefficient() else 1.}


def exponential_value(node, context):
    x, slope = context.values(node.inputs)
    return jnp.exp(node.attrs["sign"]*slope*x)


def exponential_integral(node, context, axes):
    if len(axes) != 1 or axes[0] not in node.inputs:
        raise UnsupportedModelError(f"{node.name}: exponential integral over {axes}")
    other = node.inputs[1] if axes[0] == node.inputs[0] else node.inputs[0]
    if context.depends_on(other, axes[0]):
        raise UnsupportedModelError(f"{node.name}: exponential arguments depend on integration variable")
    slope = node.attrs["sign"]*context.value(other)
    low, high = context.bounds(axes[0])
    width = high-low
    if not math.isfinite(low) or not math.isfinite(high) or not math.isfinite(width) or width <= 0:
        raise UnsupportedModelError(f"{node.name}: exponential integral requires a finite positive range")
    z = slope*width
    # The power series supplies the exact limiting derivatives at zero slope.
    series = 1 + z*(.5 + z*(1/6 + z*(1/24 + z*(1/120 + z/720))))
    small = width*jnp.exp(slope*low)*series
    magnitude = jnp.abs(slope)
    large = (jnp.exp(jnp.maximum(slope*low, slope*high)) * -jnp.expm1(-magnitude*width)
             / jnp.where(magnitude > 0, magnitude, 1.))
    return jnp.where(jnp.abs(z) < 1e-4, small, large)


def extract_power(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("Power forced numerical or named-range normalisation")
    return (str(root.base().GetName()), str(root.exponent().GetName())), {}


def power_value(node, context):
    base, exponent = context.values(node.inputs)
    return jnp.power(base, exponent)


def power_integral(node, context, axes):
    if axes != (node.inputs[0],) or context.depends_on(node.inputs[1], axes[0]):
        raise UnsupportedModelError(f"{node.name}: power integral requires an independent direct base")
    low, high = context.bounds(axes[0])
    if low < 0 or high <= low or not math.isfinite(low+high):
        raise UnsupportedModelError(f"{node.name}: power integral requires a finite nonnegative range")
    q = context.value(node.inputs[1])+1
    if low == 0:
        return jnp.where(q > 0, jnp.exp(q*jnp.log(high))/jnp.where(q > 0, q, 1.), jnp.inf)
    log_ratio = math.log(high/low)
    z = q*log_ratio
    series = 1+z*(.5+z*(1/6+z*(1/24+z*(1/120+z/720))))
    regular = jnp.expm1(z)/jnp.where(q != 0, q, 1.)
    return jnp.exp(q*math.log(low))*jnp.where(jnp.abs(z)<1e-4, log_ratio*series, regular)


def extract_bernstein(root):
    if root.x().ClassName() != "RooRealVar":
        raise UnsupportedModelError("Bernstein requires a direct RooRealVar observable")
    root.fillBuffer()
    low, high = float(root.xmin()), float(root.xmax())
    coefficients = tuple(str(c.GetName()) for c in root.coefList())
    if not coefficients or high <= low or not math.isfinite(low+high):
        raise UnsupportedModelError("Invalid Bernstein coefficients or reference range")
    observable = str(root.x().GetName())
    return (observable, *coefficients), {"observable_name": observable, "coefs": coefficients,
                                        "reference_range": (low, high)}


def bernstein_value(node, context):
    coefficients = context.values(node.attrs["coefs"])
    low, high = node.attrs["reference_range"]
    x = (context.value(node.attrs["observable_name"])-low)/(high-low)
    degree = len(coefficients)-1
    if degree == 0:
        return coefficients[0]+jnp.zeros_like(x)
    if degree == 1:
        return (coefficients[1]-coefficients[0])*x+coefficients[0]
    if degree == 2:
        a1 = 2*(coefficients[1]-coefficients[0])
        a2 = coefficients[2]-a1-coefficients[0]
        return (a2*x+a1)*x+coefficients[0]
    result, power = coefficients[0]*(1-x), x
    for i in range(1, degree):
        result = (result+power*math.comb(degree, i)*coefficients[i])*(1-x)
        power = power*x
    return result+power*coefficients[-1]


def bernstein_integral(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: Bernstein integral over {axes}")
    if any(context.depends_on(c, axes[0]) for c in node.attrs["coefs"]):
        raise UnsupportedModelError(f"{node.name}: observable-dependent Bernstein coefficient")
    low, high = node.attrs["reference_range"]
    xmin, xmax = context.bounds(axes[0])
    tmin, tmax = (xmin-low)/(high-low), (xmax-low)/(high-low)
    degree = len(node.attrs["coefs"])-1
    result = jnp.asarray(0.)
    # Follow ROOT's advertised power-basis integral, including its ordering.
    for i, name in enumerate(node.attrs["coefs"]):
        weight = 0.
        for j in range(i, degree+1):
            weight += (-1.)**(j-i)*math.comb(degree, j)*math.comb(j, i)*(tmax**(j+1)-tmin**(j+1))/(j+1)
        result += weight*context.value(name)
    return result*(high-low)
