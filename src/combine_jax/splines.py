"""Exact polynomial lowering of RooSpline1D's stored interpolation tables."""

import jax.numpy as jnp
import numpy as np
from numpy.polynomial import Polynomial

from .graph import UnsupportedModelError
from .histograms import immutable_array
from .root_io import load_root, stored_member


ALGORITHMS = ("LINEAR", "POLYNOMIAL", "CSPLINE", "CSPLINE_PERIODIC", "AKIMA", "AKIMA_PERIODIC")


def extract_spline(node):
    x, y = np.array(stored_member(node, "x_"), dtype=np.float64), np.array(stored_member(node, "y_"), dtype=np.float64)
    algorithm = stored_member(node, "type_")
    minimum = 5 if algorithm.startswith("AKIMA") else (3 if algorithm.startswith("CSPLINE") else 2)
    if algorithm not in ALGORITHMS:
        raise UnsupportedModelError(f"Unknown RooSpline1D interpolation algorithm {algorithm!r}")
    if (x.ndim != 1 or x.shape != y.shape or len(x) < minimum
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)) or np.any(np.diff(x) <= 0)):
        raise UnsupportedModelError(f"Invalid {algorithm} spline table (requires at least {minimum} ordered points)")
    servers = tuple(node.servers())
    if len(servers) != 1:
        raise UnsupportedModelError("RooSpline1D requires one input argument")
    attrs = {"knots": immutable_array(x), "algorithm": algorithm}
    if algorithm == "POLYNOMIAL":
        # Newton interpolation in a scaled coordinate avoids large powers of
        # physical masses. These are exact divided differences of the table.
        u = (x-x[0])/(x[-1]-x[0])
        divided = y.copy()
        for order in range(1, len(x)):
            divided[order:] = (divided[order:]-divided[order-1:-1])/(u[order:]-u[:-order])
        polynomial = Polynomial([divided[-1]])
        for index in range(len(x)-2, -1, -1):
            polynomial = polynomial*Polynomial([-u[index], 1.])+divided[index]
        attrs.update(newton=immutable_array(divided), scaled_knots=immutable_array(u),
                     antiderivative=immutable_array(polynomial.integ().coef))
    else:
        h = np.diff(x)
        slopes = np.diff(y)/h
        if algorithm == "LINEAR":
            coefficients = np.stack((y[:-1], slopes, np.zeros_like(slopes), np.zeros_like(slopes)))
        else:
            root = load_root()
            enum = getattr(root.Math.Interpolation, "k"+algorithm)
            interpolator = root.Math.Interpolator(x.tolist(), y.tolist(), enum)
            derivatives = np.array([interpolator.Deriv(float(knot)) for knot in x])
            if not np.all(np.isfinite(derivatives)):
                raise UnsupportedModelError(f"{algorithm} spline has nonfinite knot derivatives")
            # All four GSL spline/Akima variants are piecewise cubic. Endpoint
            # values and first derivatives uniquely determine each polynomial;
            # this is coefficient extraction, not a fit to sampled values.
            coefficients = np.stack((y[:-1], derivatives[:-1],
                                     (3*slopes-2*derivatives[:-1]-derivatives[1:])/h,
                                     (derivatives[:-1]+derivatives[1:]-2*slopes)/h**2))
        attrs["coefficients"] = immutable_array(coefficients)
    return (str(servers[0].GetName()),), attrs


def spline_value(node, context):
    attrs = node.attrs
    x = context.value(node.inputs[0])
    knots = jnp.asarray(attrs["knots"])
    if attrs["algorithm"] == "POLYNOMIAL":
        u = (x-knots[0])/(knots[-1]-knots[0])
        result = jnp.asarray(attrs["newton"][-1])
        for index in range(len(attrs["newton"])-2, -1, -1):
            result = result*(u-attrs["scaled_knots"][index])+attrs["newton"][index]
    else:
        index = jnp.clip(jnp.searchsorted(knots, x, side="right")-1, 0, len(knots)-2)
        delta = x-knots[index]
        c0, c1, c2, c3 = jnp.asarray(attrs["coefficients"])[:, index]
        result = ((c3*delta+c2)*delta+c1)*delta+c0
    # GSL does not extrapolate even its periodic interpolators.
    return jnp.where((x >= knots[0]) & (x <= knots[-1]), result, jnp.nan)


def spline_integral(node, context, axes):
    if axes != node.inputs:
        raise UnsupportedModelError(f"{node.name}: spline integral requires its direct input argument")
    low, high = context.bounds(axes[0])
    attrs, knots = node.attrs, node.attrs["knots"]
    if low < knots[0] or high > knots[-1]:
        raise UnsupportedModelError(f"{node.name}: spline integration range outside the knot domain")
    if attrs["algorithm"] == "POLYNOMIAL":
        polynomial = Polynomial(attrs["antiderivative"])
        width = knots[-1]-knots[0]
        return jnp.asarray(width*(polynomial((high-knots[0])/width)-polynomial((low-knots[0])/width)))
    left = np.clip(low-knots[:-1], 0., np.diff(knots))
    right = np.clip(high-knots[:-1], 0., np.diff(knots))
    result = sum(np.sum(c*(right**(power+1)-left**(power+1))/(power+1))
                 for power, c in enumerate(attrs["coefficients"]))
    return jnp.asarray(result)
