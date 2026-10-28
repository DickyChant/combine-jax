"""Combine vertical interpolation, with distinct PDF and stored-histogram semantics."""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import immutable_array, lookup_bins, smooth_step
from .root_io import load_root


def extract_vertical(root):
    if root.getForceNumInt() or root.quadraticAlgo() < 0:
        raise UnsupportedModelError("VerticalInterpPdf numerical/multiplicative normalisation is not yet supported")
    if root.quadraticRegion() < 0:
        raise UnsupportedModelError("Negative VerticalInterpPdf smoothing region")
    functions = tuple(str(f.GetName()) for f in root.funcList())
    coefs = tuple(str(c.GetName()) for c in root.coefList())
    if len(functions) != 2*len(coefs)+1:
        raise UnsupportedModelError("Invalid VerticalInterpPdf component count")
    return functions + coefs, {"functions": functions, "coefs": coefs,
                              "region": float(root.quadraticRegion()), "algorithm": int(root.quadraticAlgo()),
                              "pdf_floor": float(root.pdfFloorVal()), "integral_floor": float(root.integralFloorVal())}


def vertical_interpolate(values, coefficients, attrs):
    central = values[0]
    result = central
    region, algorithm = attrs["region"], attrs["algorithm"]
    for index, theta in enumerate(coefficients):
        high, low = values[2*index+1:2*index+3]
        linear = theta*jnp.where(theta > 0, high-central, central-low)
        if region == 0:
            shift = linear
        else:
            if algorithm == 0:
                up = theta*(region+theta)/(2*region)
                down = -theta*(region-theta)/(2*region)
                middle = up*high + down*low - theta**2/region*central
            elif algorithm == 1:
                up, down = (region+theta)**2/(4*region), (region-theta)**2/(4*region)
                middle = up*high + down*low - (up+down)*central
            else:
                middle = .5*theta*((high-low)+smooth_step(theta, region)*(high+low-2*central))
            shift = jnp.where(jnp.abs(theta) >= region, linear, middle)
        result = result + shift
    # This floor replaces nonpositive values only; it is not a lower clamp.
    return jnp.where(result > 0, result, attrs["pdf_floor"])


def vertical_value(node, context):
    return vertical_interpolate(context.values(node.attrs["functions"]), context.values(node.attrs["coefs"]), node.attrs)


def vertical_integral(node, context, axes):
    if any(context.depends_on(c, axis) for c in node.attrs["coefs"] for axis in axes):
        raise UnsupportedModelError(f"{node.name}: observable-dependent vertical coefficient")
    result = vertical_interpolate([context.integral(f, axes) for f in node.attrs["functions"]],
                                  context.values(node.attrs["coefs"]), node.attrs)
    # Match v11's advertised analytic integral, including its floor ordering.
    return jnp.where(result > 0, result, node.attrs["integral_floor"])


@lru_cache(maxsize=1)
def fast_vertical_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
      namespace combine_jax_io {
      class FastOldAccess : public FastVerticalInterpHistPdf {
      public:
        FastOldAccess(FastVerticalInterpHistPdf const& h) : FastVerticalInterpHistPdf(h, "combine_jax_oldvertical") {}
        FastHisto const& nominalLog() const { return _cacheNominalLog; }
        FastHisto const& cache() const { return _cache; }
        RooRealVar const& x() const { return dynamic_cast<RooRealVar const&>(_x.arg()); }
        void replaceTemplates(std::vector<TH1*> const& histograms) {
          _cacheNominal=FastHisto(*histograms.at(0)); _cacheNominal.Normalize();
          _cacheNominalLog=_cacheNominal; _cacheNominalLog.Log();
          for(size_t i=0;i<_morphs.size();++i) {
            FastHisto hi(*histograms.at(2*i+1)),lo(*histograms.at(2*i+2));
            hi.Normalize(); lo.Normalize(); syncMorph(_morphs[i],_cacheNominal,lo,hi);
          }
          syncTotal();
        }
      };
      class FastVerticalAccess : public FastVerticalInterpHistPdf2 {
      public:
        FastVerticalAccess(FastVerticalInterpHistPdf2 const& h) : FastVerticalInterpHistPdf2(h, "combine_jax_vertical") {}
        FastHisto const& nominalLog() const { return _cacheNominalLog; }
      };
      }
    '''):
        raise RuntimeError("Cannot compile FastVerticalInterpHistPdf2 extraction helper")
    return root.combine_jax_io


def extract_fast_vertical(root, *, template_convention='native'):
    from .template_conventions import check_convention, corrected_histograms
    check_convention(template_convention)
    access = fast_vertical_access()
    if root.ClassName() == "FastVerticalInterpHistPdf":
        if any(f.ClassName() != "RooHistPdf" for f in root.funcList()):
            raise UnsupportedModelError("Older fast vertical morphing requires fixed RooHistPdf templates")
        view = access.FastOldAccess(root)
    else:
        view = access.FastVerticalAccess(root)
    view.evaluate()  # Populate the copy's transient cache; do not change the workspace.
    if template_convention == 'v10_density' and root.ClassName() == 'FastVerticalInterpHistPdf':
        histograms = corrected_histograms(root, [view.x()])
        expected = [view.cacheNominal().GetEdge(i) for i in range(view.cacheNominal().fullsize()+1)]
        if any([h.GetXaxis().GetBinLowEdge(i+1) for i in range(h.GetNbinsX()+1)] != expected for h in histograms):
            raise UnsupportedModelError('Density convention requires template and observable binning to agree')
        view.replaceTemplates(histograms)
    nominal = view.cacheNominal()
    active, size = int(nominal.size()), int(nominal.fullsize())
    if active <= 0 or active > size or root.smoothRegion() < 0:
        raise UnsupportedModelError("Invalid fast vertical histogram geometry or smoothing region")
    coefs = tuple(str(c.GetName()) for c in root.coefList())
    if len(coefs) != len(view.morphs()):
        raise UnsupportedModelError("Fast vertical coefficients and morphs differ in size")
    if any(m.sum.size() != active or m.diff.size() != active for m in view.morphs()):
        raise UnsupportedModelError("Incompatible fast vertical morph geometry")
    def values(template):
        return immutable_array([template[i] for i in range(template.fullsize())])
    logarithmic = root.smoothAlgo() < 0
    observable = str(view.x().GetName())
    return (observable, *coefs), {
        "observable_name": observable, "coefs": coefs, "region": float(root.smoothRegion()),
        "logarithmic": logarithmic, "active": active,
        "template_convention": template_convention,
        "nominal": values(view.nominalLog() if logarithmic else nominal),
        "tail": immutable_array(values(view.cache())[active:]),
        "sums": immutable_array([values(m.sum)[:active] for m in view.morphs()]),
        "diffs": immutable_array([values(m.diff)[:active] for m in view.morphs()]),
        "edges": immutable_array([nominal.GetEdge(i) for i in range(size+1)]),
        "widths": immutable_array([nominal.GetWidth(i) for i in range(active)]),
    }


def fast_vertical_bins(node, context):
    key = ("fast_vertical_bins", node.name)
    if key not in context.cache:
        attrs = node.attrs
        if any(context.depends_on(c, attrs["observable_name"]) for c in attrs["coefs"]):
            raise UnsupportedModelError(f"{node.name}: observable-dependent vertical coefficient")
        values = jnp.asarray(attrs["nominal"][:attrs["active"]])
        for i, name in enumerate(attrs["coefs"]):
            theta = context.value(name)
            step = smooth_step(theta, attrs["region"]) if attrs["region"] else jnp.where(theta > 0, 1., -1.)
            values = values + .5*theta*(attrs["diffs"][i] + step*attrs["sums"][i])
        values = jnp.exp(values) if attrs["logarithmic"] else jnp.maximum(values, 1e-9)
        total = jnp.sum(values*jnp.asarray(attrs["widths"]))
        values = values / jnp.where(total > 0, total, 1.)
        context.cache[key] = jnp.concatenate((values, jnp.asarray(attrs["tail"])))
    return context.cache[key]


def fast_vertical_value(node, context):
    return lookup_bins(fast_vertical_bins(node, context), node.attrs["edges"],
                       context.value(node.attrs["observable_name"]))


def fast_vertical_integral(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: fast vertical histogram integral over {axes}")
    low, high = context.bounds(axes[0])
    edges = node.attrs["edges"]
    overlaps = np.maximum(0., np.minimum(edges[1:], high)-np.maximum(edges[:-1], low))
    return jnp.sum(fast_vertical_bins(node, context)*jnp.asarray(overlaps))


def fast_vertical_density(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: fast vertical histogram normalisation over {axes}")
    # These classes advertise selfNormalized(), even for cropped active storage.
    return context.value(node.name)
