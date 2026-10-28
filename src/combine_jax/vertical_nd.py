"""Stored two- and three-dimensional Combine vertical histogram morphers."""

from functools import lru_cache
import json

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import immutable_array, smooth_step
from .root_io import load_root


@lru_cache(maxsize=1)
def multidimensional_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "TBufferJSON.h"
      #include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
      namespace combine_jax_io {
      struct DoubleFormatGuard {
        std::string saved{TBufferJSON::GetDoubleFormat()};
        DoubleFormatGuard() { TBufferJSON::SetDoubleFormat("%.17g"); }
        ~DoubleFormatGuard() { TBufferJSON::SetDoubleFormat(saved.c_str()); }
      };
      template<class Pdf, class Hist> class FastNDView : public Pdf {
      public:
        FastNDView(Pdf const& p) : Pdf(p, "combine_jax_ndvertical") {}
        Hist const& nominal() const { return this->_cacheNominal; }
        Hist const& nominalLog() const { return this->_cacheNominalLog; }
        void replaceTemplates(std::vector<TH1*> const& histograms) {
          auto make=[&](TH1* h) {
            if constexpr(std::is_same_v<Hist,FastHisto3D>) return Hist(dynamic_cast<TH3&>(*h),this->_conditional);
            else return Hist(dynamic_cast<TH2&>(*h),this->_conditional);
          };
          auto normalize=[&](Hist& h) { if(this->_conditional) h.NormalizeXSlices(); else h.Normalize(); };
          this->_cacheNominal=make(histograms.at(0)); normalize(this->_cacheNominal);
          this->_cacheNominalLog=this->_cacheNominal; this->_cacheNominalLog.Log();
          for(size_t i=0;i<this->_morphs.size();++i) {
            Hist hi=make(histograms.at(2*i+1)),lo=make(histograms.at(2*i+2));
            normalize(hi); normalize(lo); this->syncMorph(this->_morphs[i],this->_cacheNominal,lo,hi);
          }
          this->syncTotal();
        }
        TString geometryMember(char const* member) const {
          DoubleFormatGuard guard;
          return TBufferJSON::ConvertToJSON(&this->_cacheNominal, TClass::GetClass<Hist>(), 3, member);
        }
      };
      using Fast2DView = FastNDView<FastVerticalInterpHistPdf2D, FastHisto2D>;
      using Fast2DStoredView = FastNDView<FastVerticalInterpHistPdf2D2, FastHisto2D>;
      using Fast3DView = FastNDView<FastVerticalInterpHistPdf3D, FastHisto3D>;
      }
    '''):
        raise RuntimeError("Cannot compile multidimensional vertical histogram extraction helper")
    return root.combine_jax_io


def extract_multidimensional(root, *, template_convention='native'):
    from .template_conventions import check_convention, corrected_histograms
    check_convention(template_convention)
    access = multidimensional_access()
    classes = {"FastVerticalInterpHistPdf2D": access.Fast2DView,
               "FastVerticalInterpHistPdf2D2": access.Fast2DStoredView,
               "FastVerticalInterpHistPdf3D": access.Fast3DView}
    if root.ClassName() != "FastVerticalInterpHistPdf2D2":
        if any(f.ClassName() != "RooHistPdf" for f in root.funcList()):
            raise UnsupportedModelError("Older multidimensional vertical morphing requires fixed RooHistPdf templates")
    view = classes[root.ClassName()](root)
    view.evaluate()
    if template_convention == 'v10_density' and root.ClassName() != 'FastVerticalInterpHistPdf2D2':
        dimensions = 3 if root.ClassName() == 'FastVerticalInterpHistPdf3D' else 2
        histograms = corrected_histograms(root, [getattr(view, key)() for key in 'xyz'[:dimensions]], bool(root.conditional()))
        for key in 'XYZ'[:dimensions]:
            expected = json.loads(str(view.geometryMember(f'binEdges{key}_')))
            for hist in histograms:
                axis = getattr(hist, f'Get{key}axis')()
                if [axis.GetBinLowEdge(i+1) for i in range(axis.GetNbins()+1)] != expected:
                    raise UnsupportedModelError('Density convention requires template and observable binning to agree')
        view.replaceTemplates(histograms)
    nominal = view.nominal()
    dimensions = 3 if root.ClassName() == "FastVerticalInterpHistPdf3D" else 2
    axes = tuple(str(getattr(view, key)().GetName()) for key in "xyz"[:dimensions])
    edges = tuple(immutable_array(json.loads(str(view.geometryMember(f"binEdges{key}_"))))
                  for key in "XYZ"[:dimensions])
    shape = tuple(len(e)-1 for e in edges)
    size = int(nominal.fullsize())
    if (nominal.size() != size or size != int(np.prod(shape)) or size == 0
            or any(np.any(np.diff(e) <= 0) for e in edges) or root.smoothRegion() < 0):
        raise UnsupportedModelError("Invalid or cropped multidimensional vertical histogram geometry")
    coefs = tuple(str(c.GetName()) for c in root.coefList())
    if (len(coefs) != len(view.morphs())
            or any(m.sum.size() != size or m.diff.size() != size for m in view.morphs())):
        raise UnsupportedModelError("Incompatible multidimensional vertical morph storage")

    def values(template):
        return immutable_array([template[i] for i in range(size)]).reshape(shape)

    logarithmic = root.smoothAlgo() < 0
    return (*axes, *coefs), {
        "axes": axes, "edges": edges, "shape": shape, "coefs": coefs,
        "conditional": bool(root.conditional()), "logarithmic": logarithmic,
        "region": float(root.smoothRegion()),
        "template_convention": template_convention,
        "nominal": values(view.nominalLog() if logarithmic else nominal),
        "sums": immutable_array([values(m.sum) for m in view.morphs()]),
        "diffs": immutable_array([values(m.diff) for m in view.morphs()]),
        "widths": immutable_array(json.loads(str(view.geometryMember("binWidths_")))).reshape(shape),
    }


def multidimensional_bins(node, context):
    key = ("multidimensional_vertical_bins", node.name)
    if key not in context.cache:
        attrs = node.attrs
        if any(context.depends_on(c, axis) for c in attrs["coefs"] for axis in attrs["axes"]):
            raise UnsupportedModelError(f"{node.name}: observable-dependent vertical coefficient")
        values = jnp.asarray(attrs["nominal"])
        for i, name in enumerate(attrs["coefs"]):
            theta = context.value(name)
            step = smooth_step(theta, attrs["region"]) if attrs["region"] else jnp.where(theta > 0, 1., -1.)
            values = values + .5*theta*(attrs["diffs"][i] + step*attrs["sums"][i])
        values = jnp.exp(values) if attrs["logarithmic"] else jnp.maximum(values, 1e-9)
        dimensions = tuple(range(1 if attrs["conditional"] else 0, len(attrs["axes"])))
        total = jnp.sum(values*jnp.asarray(attrs["widths"]), axis=dimensions, keepdims=True)
        context.cache[key] = values / jnp.where(total > 0, total, 1.)
    return context.cache[key]


def lookup_grid(values, names, edges, context):
    indices, valid = [], jnp.asarray(True)
    for name, edge in zip(names, edges, strict=True):
        coordinate = context.value(name)
        index = jnp.searchsorted(jnp.asarray(edge), coordinate, side="left")-1
        indices.append(jnp.clip(index, 0, len(edge)-2))
        valid = valid & (index >= 0) & (index < len(edge)-1)
    return jnp.where(valid, values[tuple(indices)], 0.)


def multidimensional_value(node, context):
    return lookup_grid(multidimensional_bins(node, context), node.attrs["axes"], node.attrs["edges"], context)


def multidimensional_integral(node, context, axes):
    attrs = node.attrs
    if not set(axes).issubset(attrs["axes"]):
        raise UnsupportedModelError(f"{node.name}: multidimensional histogram integral over {axes}")
    values = multidimensional_bins(node, context)
    # Integrate the geometric volume, including x for a conditionally normalised
    # template. Stored normalization weights intentionally omit that x width.
    for dimension in reversed(range(len(attrs["axes"]))):
        name, edges = attrs["axes"][dimension], attrs["edges"][dimension]
        if name in axes:
            low, high = context.bounds(name)
            overlap = np.maximum(0., np.minimum(edges[1:], high)-np.maximum(edges[:-1], low))
            broadcast = [1]*values.ndim
            broadcast[dimension] = len(overlap)
            values = jnp.sum(values*jnp.asarray(overlap).reshape(broadcast), axis=dimension)
    remaining = [(name, edge) for name, edge in zip(attrs["axes"], attrs["edges"], strict=True) if name not in axes]
    return (lookup_grid(values, tuple(n for n, _ in remaining), tuple(e for _, e in remaining), context)
            if remaining else values)


def multidimensional_density(node, context, axes):
    if not set(axes).issubset(node.attrs["axes"]):
        raise UnsupportedModelError(f"{node.name}: multidimensional histogram normalisation over {axes}")
    # Match RooAbsPdf::selfNormalized, including conditional templates.
    return context.value(node.name)
