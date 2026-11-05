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
      #include "HiggsAnalysis/CombinedLimit/interface/RooParametricHist.h"
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
      class Parametric1DAccess : public RooParametricHist {
      public:
        Parametric1DAccess(RooParametricHist const& p) : RooParametricHist(p, "combine_jax_parametric1d") {}
        RooAbsReal const& xvar() const { return x.arg(); }
        RooArgList const& coefficients() const { return pars; }
        RooArgList const& morphs() const { return _coeffList; }
        std::vector<double> edges() const { return bins; }
        std::vector<double> binWidths() const { return widths; }
        std::vector<std::vector<double>> diffs() const { return _diffs; }
        std::vector<std::vector<double>> sums() const { return _sums; }
        bool morphing() const { return _hasMorphs; }
        double region() const { return _smoothRegion; }
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


def extract_parametric_1d(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("ParametricHist forced numerical or named-range normalisation")
    view = parametric_template_access().Parametric1DAccess(root)
    if view.xvar().ClassName() != "RooRealVar":
        raise UnsupportedModelError("ParametricHist requires a direct real observable")
    axis = str(view.xvar().GetName())
    edges = checked_edges(list(view.edges()))
    widths = immutable_array(list(view.binWidths()))
    coefs = tuple(str(v.GetName()) for v in view.coefficients())
    morphs = tuple(str(v.GetName()) for v in view.morphs())
    if (len(coefs) != len(edges)-1 or widths.shape != (len(coefs),)
            or not np.all(np.isfinite(widths)) or np.any(widths <= 0)):
        raise UnsupportedModelError("Invalid ParametricHist stored bin geometry")
    has_morphs = bool(view.morphing())
    if morphs and not has_morphs:
        raise UnsupportedModelError("Inconsistent ParametricHist morph flag/coefficients")
    # Serialized HIG workspaces can retain _hasMorphs=true with no coefficients.
    # Native evaluateMorphFunction returns its initial scale=1 in that case;
    # neither the tables nor smoothing region are read. Fold that empty product.
    diffs = immutable_array([list(row) for row in view.diffs()] if morphs else np.empty((len(coefs), 0)))
    sums = immutable_array([list(row) for row in view.sums()] if morphs else np.empty((len(coefs), 0)))
    region = float(view.region()) if morphs else 0.
    if (diffs.shape != (len(coefs), len(morphs)) or sums.shape != diffs.shape
            or not np.all(np.isfinite(diffs)) or not np.all(np.isfinite(sums))
            or not np.isfinite(region) or region < 0):
        raise UnsupportedModelError("Invalid ParametricHist morph tables or smoothing region")
    return (axis, *coefs, *morphs), dict(axes=(axis,), coefs=coefs, morphs=morphs,
        edges=(edges,), widths=(widths,), shape=(len(coefs),), diffs=diffs, sums=sums, region=region)


def parametric_1d_bins(node, context):
    a = node.attrs
    values = context.values(a['coefs'])
    scales = [jnp.ones_like(value) for value in values]
    # Preserve the native ordered product and its division by the current bin
    # yield. In particular, do not silently regularize zero morphed bin yields.
    for j, theta in enumerate(context.values(a['morphs'])):
        z = theta / (a['region'] if a['region'] > 0 else 1.)
        smooth = jnp.where(jnp.abs(theta) >= a['region'], jnp.where(theta > 0, 1., -1.),
                           .125*z*(z*z*(3*z*z-10)+15))
        # Multiplying the yield at each step would change the denominator of
        # subsequent factors; native always divides by the unmorphed yield.
        scales = [scale * (1. + (1./value)*(.5*theta)*(diff + smooth*total))
                  for value, scale, diff, total in zip(values, scales, a['diffs'][:, j], a['sums'][:, j], strict=True)]
    return [value*scale for value, scale in zip(values, scales, strict=True)] if a['morphs'] else values


def parametric_1d_value(node, context):
    a = node.attrs
    edges = a['edges'][0]
    x = context.value(a['axes'][0])
    index = jnp.clip(jnp.searchsorted(jnp.asarray(edges), x, side='right')-1, 0, len(edges)-2)
    value = selected_bin(parametric_1d_bins(node, context), index)/jnp.asarray(a['widths'][0])[index]
    return jnp.where((x >= edges[0]) & (x < edges[-1]), value, 0.)


def parametric_1d_integral(node, context, axes):
    if axes != node.attrs['axes'] or any(context.depends_on(n, a)
            for n in (*node.attrs['coefs'], *node.attrs['morphs']) for a in axes):
        raise UnsupportedModelError(f"{node.name}: ParametricHist requires an integral with independent bin/morph parameters")
    # Native's unnamed integral is the full yield even if the default
    # observable bounds cover only part of the stored histogram geometry.
    return sum(parametric_1d_bins(node, context))
