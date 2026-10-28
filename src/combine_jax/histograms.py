"""Exact stored-template extraction and vertical CMSHistFunc interpolation."""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .root_io import load_root


@lru_cache(maxsize=1)
def hist_access():
    root = load_root()
    # A derived copy legally exposes protected state without mutating the input.
    # It reads stored arrays, never a sampled parameter-dependent function.
    if not root.gInterpreter.Declare(r'''
        #include "HiggsAnalysis/CombinedLimit/interface/CMSHistFunc.h"
        namespace combine_jax_io {
        class HistAccess : public CMSHistFunc {
        public:
          HistAccess(CMSHistFunc const& h) : CMSHistFunc(h, "combine_jax_export_copy") {}
          RooArgList const& vertical() const { return vmorphs_; }
          unsigned horizontalCount() const { return hmorphs_.size(); }
          unsigned externalCount() const { return external_morph_.size(); }
          unsigned strategy() const { return morph_strategy_; }
          int verticalType() const { return vtype_; }
          double smoothRegion() const { return vsmooth_par_; }
          bool rebin() const { return rebin_; }
          unsigned active() const { return cache_.size(); }
          unsigned outputActive() const { return cache().size(); }
          std::vector<unsigned> rebinScheme() const { return rebin_scheme_; }
          std::vector<double> edges(bool output=false) const {
            auto const& hist = output ? cache() : cache_;
            std::vector<double> result;
            for(unsigned i=0; i<=hist.fullsize(); ++i) result.push_back(hist.GetEdge(i));
            return result;
          }
          std::vector<double> cacheValues(bool output=false) const {
            auto const& hist = output ? cache() : cache_;
            std::vector<double> result;
            for(unsigned i=0; i<hist.fullsize(); ++i) result.push_back(hist[i]);
            return result;
          }
          std::vector<std::vector<double>> storage() const {
            std::vector<std::vector<double>> result;
            for(auto const& h : storage_) {
              std::vector<double> values;
              for(unsigned i=0; i<h.fullsize(); ++i) values.push_back(h[i]);
              result.push_back(values);
            }
            return result;
          }
        };
        }
    '''):
        raise RuntimeError("Cannot compile CMSHistFunc extraction helper")
    return root.combine_jax_io.HistAccess


def immutable_array(values):
    result = np.asarray(values, dtype=np.float64)
    result.flags.writeable = False
    return result


def extract_histfunc(root):
    view = hist_access()(root)
    if view.horizontalCount():
        raise UnsupportedModelError("CMSHistFunc horizontal morphing is not yet supported")
    if view.externalCount():
        raise UnsupportedModelError("CMSHistFunc external morphs are not yet supported")
    if view.strategy() != 0:
        raise UnsupportedModelError("CMSHistFunc nonzero morph strategy is not yet supported")
    if view.verticalType() not in (0, 1) or view.smoothRegion() <= 0:
        raise UnsupportedModelError("Unsupported CMSHistFunc vertical interpolation settings")
    observable = str(root.getXVar().GetName())
    morphs = tuple(str(v.GetName()) for v in view.vertical())
    templates = immutable_array([list(values) for values in view.storage()])
    if templates.ndim != 2 or templates.shape[0] != 1 + 2*len(morphs):
        raise UnsupportedModelError("Incomplete CMSHistFunc storage")
    attrs = {
        "observable_name": observable, "morphs": morphs, "templates": templates,
        "edges": immutable_array(list(view.edges())), "active": int(view.active()),
        "cache": immutable_array(list(view.cacheValues())),
        "errors": immutable_array([root.errors()[i] for i in range(root.errors().fullsize())]),
        "vertical_type": int(view.verticalType()), "smooth_region": float(view.smoothRegion()),
    }
    if view.rebin():
        indices = tuple(int(i) for i in view.rebinScheme())
        output_active = int(view.outputActive())
        if len(indices) != templates.shape[1] or any(i < 0 or i >= output_active for i in indices):
            raise UnsupportedModelError("Invalid CMSHistFunc stored rebinning map")
        attrs.update(source_active=attrs["active"], source_edges=attrs["edges"],
                     rebin_indices=indices, active=output_active,
                     edges=immutable_array(list(view.edges(True))),
                     cache=immutable_array(list(view.cacheValues(True))))
    return (observable, *morphs), attrs


def smooth_step(theta, region):
    x = theta / region
    polynomial = x * (x*x * (3*x*x - 10) + 15) / 8
    return jnp.where(jnp.abs(theta) >= region, jnp.where(theta > 0, 1., -1.), polynomial)


def hist_bins(node, context):
    key = ("histogram_bins", node.name)
    if key in context.cache:
        return context.cache[key]
    a = node.attrs
    active = a.get("source_active", a["active"])
    templates = jnp.asarray(a["templates"][:, :active])
    nominal = templates[0]
    log_mode = a["vertical_type"] == 1
    if log_mode:
        bins = jnp.where(nominal > 0, jnp.log(jnp.where(nominal > 0, nominal, 1.)), -999.)
    else:
        bins = nominal
    for i, name in enumerate(a["morphs"]):
        low, high = templates[2*i + 1], templates[2*i + 2]
        if log_mode:
            def log_ratio(values):
                valid = (values > 0) & (nominal > 0)
                ratio = jnp.where(valid, values, 1.) / jnp.where(valid, nominal, 1.)
                return jnp.where(valid, jnp.log(ratio), 0.)
            low, high = log_ratio(low), log_ratio(high)
        else:
            low, high = low - nominal, high - nominal
        theta = context.value(name)
        bins = bins + .5 * theta * ((high-low) + smooth_step(theta, a["smooth_region"])*(high+low))
    if log_mode:
        bins = jnp.exp(bins)
        # CMSHistFunc preserves the unweighted active-bin sum in log mode.
        bins = bins * (jnp.sum(nominal) / jnp.sum(bins))
    bins = jnp.maximum(bins, 1e-9)
    if "rebin_indices" in a:
        # Combine assigns each entire fine bin using its stored midpoint map.
        # Flooring and log-mode normalization happen before width-weighted sums.
        indices = jnp.asarray(a["rebin_indices"][:active])
        widths = jnp.diff(jnp.asarray(a["source_edges"]))[:active]
        result = jnp.zeros(len(a["cache"])).at[indices].add(bins*widths)
        result = result / jnp.diff(jnp.asarray(a["edges"]))
    else:
        result = jnp.concatenate((bins, jnp.asarray(a["cache"][active:])))
    context.cache[key] = result
    return result


def hist_value(node, context):
    coordinates = context.value(node.attrs["observable_name"])
    return lookup_bins(hist_bins(node, context), node.attrs["edges"], coordinates)


def lookup_bins(values, edges, coordinates):
    edges = jnp.asarray(edges)
    # FastHisto uses lower_bound: bins are (low, high], including the upper edge.
    index = jnp.searchsorted(edges, coordinates, side="left") - 1
    return jnp.where((index >= 0) & (index < len(values)), values[jnp.clip(index, 0, len(values)-1)], 0.)


def hist_integral(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: CMSHistFunc integral over {axes}")
    active = node.attrs["active"]
    # CMSHistFunc::analyticalIntegral uses its full active geometry, not rangeName.
    return jnp.sum(hist_bins(node, context)[:active] * jnp.diff(jnp.asarray(node.attrs["edges"]))[:active])
