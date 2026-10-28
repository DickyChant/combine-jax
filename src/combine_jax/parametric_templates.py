"""Floating bin densities with each source class's advertised normalisation."""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import immutable_array
from .root_io import load_root


@lru_cache(maxsize=1)
def parametric_template_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "RooParametricStepFunction.h"
      #include "HiggsAnalysis/CombinedLimit/interface/RooParametricHist2D.h"
      namespace combine_jax_io {
      class StepAccess : public RooParametricStepFunction {
      public:
        StepAccess(RooParametricStepFunction const& p) : RooParametricStepFunction(p, "combine_jax_step") {}
        RooAbsReal const& observable() const { return _x.arg(); }
        RooArgList const& coefficients() const { return _coefList; }
        std::vector<double> edges() const { return {_limits.GetArray(), _limits.GetArray()+_limits.GetSize()}; }
      };
      class Parametric2DAccess : public RooParametricHist2D {
      public:
        Parametric2DAccess(RooParametricHist2D const& p) : RooParametricHist2D(p, "combine_jax_parametric2d") {}
        RooAbsReal const& xvar() const { return x.arg(); }
        RooAbsReal const& yvar() const { return y.arg(); }
        RooArgList const& coefficients() const { return pars; }
        std::vector<double> xedges() const { return bins_x; }
        std::vector<double> yedges() const { return bins_y; }
        std::vector<double> xwidths() const { return widths_x; }
        std::vector<double> ywidths() const { return widths_y; }
      };
      }
    '''):
        raise RuntimeError("Cannot compile parametric-template extraction helpers")
    return root.combine_jax_io


def checked_edges(values):
    edges = immutable_array(values)
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise UnsupportedModelError("Invalid parametric-template bin edges")
    return edges


def extract_step(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("Step function forced numerical or named-range normalisation")
    view = parametric_template_access().StepAccess(root)
    if view.observable().ClassName() != "RooRealVar":
        raise UnsupportedModelError("Step function requires a direct real observable")
    edges = checked_edges(list(view.edges()))
    coefficients = tuple(str(v.GetName()) for v in view.coefficients())
    if len(edges) != root.getnBins()+1 or len(coefficients) != root.getnBins()-1:
        raise UnsupportedModelError("Step function bin/parameter count mismatch")
    axis = str(view.observable().GetName())
    return (axis, *coefficients), {"axes": (axis,), "coefs": coefficients, "edges": edges}


def selected_bin(coefficients, index):
    # Scalar bin parameters need only one table, avoiding a bins-by-events
    # intermediate. Conditional coefficients can also carry event dimensions.
    values = jnp.stack(jnp.broadcast_arrays(*coefficients))
    if values.ndim == 1:
        return values[index]
    shape = jnp.broadcast_shapes(values.shape[1:], index.shape)
    values = values.reshape((values.shape[0],)+(1,)*(len(shape)-values.ndim+1)+values.shape[1:])
    values = jnp.broadcast_to(values, (values.shape[0], *shape))
    return jnp.take_along_axis(values, jnp.broadcast_to(index, shape)[None], axis=0)[0]


def step_value(node, context):
    edges = node.attrs["edges"]
    x = context.value(node.attrs["axes"][0])
    coefficients = context.values(node.attrs["coefs"])
    last = (1-sum(c*w for c,w in zip(coefficients, np.diff(edges)[:-1], strict=True)))/(edges[-1]-edges[-2])
    coefficients.append(jnp.where(last > 0, last, 1e-6))
    index = jnp.clip(jnp.searchsorted(jnp.asarray(edges), x, side="right")-1, 0, len(edges)-2)
    value = selected_bin(coefficients, index)
    return jnp.where((x >= edges[0]) & (x < edges[-1]), value, 0.)


def step_integral(node, context, axes):
    if axes != node.attrs["axes"] or any(context.depends_on(n, axes[0]) for n in node.attrs["coefs"]):
        raise UnsupportedModelError(f"{node.name}: unsupported step integral configuration")
    # ROOT's default integral is one even when the last bin was floored or
    # the observable's default bounds do not cover the complete binning.
    return jnp.asarray(1.)


def extract_parametric_2d(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("ParametricHist2D forced numerical or named-range normalisation")
    view = parametric_template_access().Parametric2DAccess(root)
    variables = (view.xvar(), view.yvar())
    if any(v.ClassName() != "RooRealVar" for v in variables):
        raise UnsupportedModelError("ParametricHist2D requires direct real observables")
    axes = tuple(str(v.GetName()) for v in variables)
    if axes[0] == axes[1]:
        raise UnsupportedModelError("ParametricHist2D requires distinct axes")
    edges = tuple(checked_edges(list(v)) for v in (view.xedges(), view.yedges()))
    widths = tuple(immutable_array(list(v)) for v in (view.xwidths(), view.ywidths()))
    for edge, width in zip(edges, widths, strict=True):
        if len(width) != len(edge)-1 or not np.all(np.isfinite(width)) or np.any(width <= 0):
            raise UnsupportedModelError("Invalid ParametricHist2D stored widths")
    coefficients = tuple(str(v.GetName()) for v in view.coefficients())
    shape = tuple(len(e)-1 for e in edges)
    if len(coefficients) != np.prod(shape):
        raise UnsupportedModelError("ParametricHist2D bin/parameter count mismatch")
    return (*axes, *coefficients), {"axes": axes, "coefs": coefficients, "edges": edges, "widths": widths, "shape": shape}


def parametric_2d_value(node, context):
    indices, valid = [], jnp.asarray(True)
    for axis, edge in zip(node.attrs["axes"], node.attrs["edges"], strict=True):
        x = context.value(axis)
        indices.append(jnp.clip(jnp.searchsorted(jnp.asarray(edge), x, side="right")-1, 0, len(edge)-2))
        valid = valid & (x >= edge[0]) & (x < edge[-1])
    ix, iy = indices
    index = iy*node.attrs["shape"][0]+ix  # x varies fastest in Combine's parameter list.
    volume = jnp.asarray(node.attrs["widths"][0])[ix]*jnp.asarray(node.attrs["widths"][1])[iy]
    value = selected_bin(context.values(node.attrs["coefs"]), index)/volume
    return jnp.where(valid, value, 0.)


def parametric_2d_integral(node, context, axes):
    if set(axes) != set(node.attrs["axes"]) or any(context.depends_on(n, a) for n in node.attrs["coefs"] for a in axes):
        raise UnsupportedModelError(f"{node.name}: ParametricHist2D requires a joint integral with independent bin parameters")
    return sum(context.values(node.attrs["coefs"]))
