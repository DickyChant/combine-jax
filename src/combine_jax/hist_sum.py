"""CMSHistSum's stored morphs, process errors and bin-statistical shifts.

Unlike CMSHistErrorPropagator, this object owns transformed templates and its
Poisson shifts use compcache (before log exponentiation and clipping).
"""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import immutable_array, lookup_bins, smooth_step
from .root_io import load_root


@lru_cache(maxsize=1)
def sum_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/CMSHistSum.h"
      namespace combine_jax_io {
      class HistSumAccess : public CMSHistSum {
      public:
        HistSumAccess(CMSHistSum const& h) : CMSHistSum(h, "combine_jax_sum_copy") {}
        RooArgList const& morphs() const { return morphpars_; }
        RooArgList const& parameters() const { return binpars_; }
        auto const& types() const { return bintypes_; }
        auto const& processes() const { return process_fields_; }
        auto const& fields() const { return vmorph_fields_; }
        std::vector<int> modes() const { return std::vector<int>(vtype_.begin(),vtype_.end()); }
        auto const& regions() const { return vsmooth_par_; }
        unsigned externalCount() const { return external_morphs_.size(); }
        unsigned active() const { return cache_.size(); }
        std::vector<double> edges() const {
          std::vector<double> out;
          for(unsigned i=0; i<=cache_.fullsize(); ++i) out.push_back(cache_.GetEdge(i));
          return out;
        }
        std::vector<double> cacheValues() const {
          std::vector<double> out;
          for(unsigned i=0; i<cache_.fullsize(); ++i) out.push_back(cache_[i]);
          return out;
        }
        std::vector<std::vector<double>> arrays(bool errors=false) const {
          std::vector<std::vector<double>> out;
          for(auto const& h : errors ? binerrors_ : storage_) {
            std::vector<double> row;
            for(unsigned i=0; i<h.fullsize(); ++i) row.push_back(h[i]);
            out.push_back(row);
          }
          return out;
        }
      };
      }
    '''):
        raise RuntimeError("Cannot compile CMSHistSum extraction helper")
    return root.combine_jax_io.HistSumAccess


def extract_hist_sum(root):
    view = sum_access()(root)
    if view.externalCount():
        raise UnsupportedModelError("CMSHistSum external morphs are not supported")
    observable = str(root.getXVar().GetName())
    coefs = tuple(str(p.GetName()) for p in root.coefList())
    morphs = tuple(str(p.GetName()) for p in view.morphs())
    processes = tuple(int(i) for i in view.processes())
    fields = tuple(int(i) for i in view.fields())
    modes = tuple(int(i) for i in view.modes())
    regions = tuple(float(i) for i in view.regions())
    storage = immutable_array([list(row) for row in view.arrays()])
    errors = immutable_array([list(row) for row in view.arrays(True)])
    edges = immutable_array(list(view.edges()))
    cache = immutable_array(list(view.cacheValues()))
    active = int(view.active())
    count = len(coefs)
    if (not count or len(processes) != count or len(modes) != count or len(regions) != count
            or len(fields) != count*len(morphs) or any(m not in (0, 1) for m in modes)
            or any(not np.isfinite(r) or r < 0 for r in regions)
            or storage.ndim != 2 or errors.shape != (count, len(cache))
            or storage.shape[1] != len(cache) or edges.shape != (len(cache)+1,)
            or not 0 < active <= len(cache) or np.any(np.diff(edges) <= 0)
            or any(i < 0 or i >= len(storage) for i in processes)
            or any(i != -1 and (i < 0 or i+1 >= len(storage)) for i in fields)
            or not all(np.all(np.isfinite(a)) for a in (storage, errors, edges, cache))):
        raise UnsupportedModelError("Invalid CMSHistSum stored geometry/morph arrays")
    types = tuple(tuple(int(t) for t in row) for row in view.types())
    if len(types) not in (0, active):
        raise UnsupportedModelError("CMSHistSum bin types do not cover active bins")
    parameters = iter(str(p.GetName()) for p in view.parameters())
    bin_parameters = []
    for row in types:
        if (not row or any(t not in range(5) for t in row)
                or (row[0] in (0, 1) and len(row) != 1)
                or (row[0] not in (0, 1) and len(row) != count)):
            raise UnsupportedModelError(f"Invalid CMSHistSum bin types {row}")
        try:
            bin_parameters.append(tuple(next(parameters) if t in (1, 2, 3) else None for t in row))
        except StopIteration as exc:
            raise UnsupportedModelError("Missing CMSHistSum bin parameter") from exc
    if next(parameters, None) is not None:
        raise UnsupportedModelError("Unused CMSHistSum bin parameter")
    inputs = (observable, *coefs, *morphs,
              *(n for row in bin_parameters for n in row if n is not None))
    return inputs, dict(observable_name=observable, coefs=coefs, morphs=morphs,
        process_fields=processes, morph_fields=fields, vertical_types=modes,
        smooth_regions=regions, storage=storage, errors=errors, edges=edges,
        cache=cache, active=active, bin_types=types, bin_parameters=tuple(bin_parameters))


def sum_moments(node, context):
    key = ("sum_moments", node.name)
    if key in context.cache:
        return context.cache[key]
    a = node.attrs
    if any(context.depends_on(n, a['observable_name']) for n in (*a['coefs'], *a['morphs'])):
        raise UnsupportedModelError(f"{node.name}: observable-dependent CMSHistSum parameters")
    active = a['active']
    storage = jnp.asarray(a['storage'][:, :active])
    values, raw = [], []
    for p, index in enumerate(a['process_fields']):
        nominal = storage[index]
        log = a['vertical_types'][p] == 1
        bins = jnp.where(nominal > 0, jnp.log(jnp.where(nominal > 0, nominal, 1.)), -999.) if log else nominal
        region = a['smooth_regions'][p]
        for m, name in enumerate(a['morphs']):
            field = a['morph_fields'][p*len(a['morphs'])+m]
            if field < 0:
                continue
            theta = context.value(name)
            alpha = smooth_step(theta, region) if region else jnp.where(theta > 0, 1., -1.)
            bins = bins + .5*theta*(storage[field+1] + alpha*storage[field])
        raw.append(bins)
        if log:
            bins = jnp.exp(bins)
            bins = bins*(jnp.sum(nominal)/jnp.sum(bins))
        values.append(jnp.maximum(bins, 1e-9))
    coefficients = jnp.stack(context.values(a['coefs']))
    values, raw = jnp.stack(values), jnp.stack(raw)
    errors = jnp.asarray(a['errors'][:, :active])
    total = jnp.sum(coefficients[:, None]*values, axis=0)
    variance = jnp.sum((coefficients[:, None]*errors)**2, axis=0)
    sigma = jnp.where(variance > 0, jnp.sqrt(jnp.where(variance > 0, variance, 1.)), 0.)
    context.cache[key] = (total, sigma, raw, errors, coefficients)
    return context.cache[key]


def sum_bins(node, context):
    key = ("sum_bins", node.name)
    if key in context.cache:
        return context.cache[key]
    total, sigma, raw, errors, coefficients = sum_moments(node, context)
    for b, (types, params) in enumerate(zip(node.attrs['bin_types'], node.attrs['bin_parameters'], strict=True)):
        shift = jnp.asarray(0.)
        if types[0] == 1:
            shift = sigma[b]*context.value(params[0])
        elif types[0] != 0:
            for p, (kind, name) in enumerate(zip(types, params, strict=True)):
                if kind == 2:
                    shift += coefficients[p]*raw[p, b]*(context.value(name)-1.)
                elif kind == 3:
                    shift += coefficients[p]*errors[p, b]*context.value(name)
        total = total.at[b].add(shift)
    result = jnp.concatenate((jnp.maximum(total, 1e-9), jnp.asarray(node.attrs['cache'][node.attrs['active']:])))
    context.cache[key] = result
    return result


def sum_value(node, context):
    return lookup_bins(sum_bins(node, context), node.attrs['edges'], context.value(node.attrs['observable_name']))


def sum_integral(node, context, axes):
    if axes != (node.attrs['observable_name'],):
        raise UnsupportedModelError(f"{node.name}: CMSHistSum integral over {axes}")
    active = node.attrs['active']
    return jnp.sum(sum_bins(node, context)[:active]*jnp.diff(jnp.asarray(node.attrs['edges']))[:active])
