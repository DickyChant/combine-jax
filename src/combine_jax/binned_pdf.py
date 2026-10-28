"""Piecewise-constant RooHistPdf tables, preserving RooFit's own bin convention."""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import immutable_array
from .root_io import load_root


@lru_cache(maxsize=1)
def hist_pdf_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "RooHistPdf.h"
      namespace combine_jax_io {
      class HistPdfAccess : public RooHistPdf {
      public:
        HistPdfAccess(RooHistPdf const& h) : RooHistPdf(h, "combine_jax_histpdf") {}
        RooArgSet const& histObservables() const { return _histObsList; }
      };
      }
    '''):
        raise RuntimeError("Cannot compile RooHistPdf extraction helper")
    return root.combine_jax_io.HistPdfAccess


def extract_hist_pdf(root):
    if root.getInterpolationOrder() != 0 or root.getCdfBoundaries() or root.haveUnitNorm():
        raise UnsupportedModelError("RooHistPdf interpolation, CDF boundaries and unit-norm mode are not yet supported")
    view = hist_pdf_access()(root)
    pdf_obs, hist_obs = tuple(root.variables()), tuple(view.histObservables())
    if len(pdf_obs) != len(hist_obs) or not pdf_obs:
        raise UnsupportedModelError("Invalid RooHistPdf observable mapping")
    if any(o.ClassName() != "RooRealVar" for o in (*pdf_obs, *hist_obs)):
        raise UnsupportedModelError("RooHistPdf requires direct real observable mappings")
    data = view.dataHist()
    axes, edges = [], []
    row = data.get()
    for pdf_axis, hist_axis in zip(pdf_obs, hist_obs, strict=True):
        variable = row.find(hist_axis.GetName())
        if not variable:
            raise UnsupportedModelError("RooHistPdf observable missing from data")
        binning = variable.getBinning()
        axes.append(str(pdf_axis.GetName()))
        edges.append(immutable_array([binning.binLow(i) for i in range(binning.numBins())]
                                     + [binning.highBound()]))
    shape = tuple(len(e)-1 for e in edges)
    if data.numEntries() != int(np.prod(shape)):
        raise UnsupportedModelError("RooHistPdf data do not form a complete real grid")
    contents, seen = np.zeros(shape), np.zeros(shape, dtype=bool)
    for i in range(data.numEntries()):
        row = data.get(i)
        cell = tuple(int(np.searchsorted(e, row.getRealValue(h.GetName()), side="right")-1)
                     for e, h in zip(edges, hist_obs, strict=True))
        if any(j < 0 or j >= n for j, n in zip(cell, shape, strict=True)) or seen[cell]:
            raise UnsupportedModelError("RooHistPdf data have invalid or duplicate cells")
        contents[cell] = float(data.weight()) / float(data.binVolume())
        seen[cell] = True
    return tuple(axes), {"axes": tuple(axes), "edges": tuple(edges), "shape": shape,
                         "values": immutable_array(contents), "sum_weights": float(data.sumEntries())}


def hist_pdf_value(node, context):
    indices, valid = [], jnp.asarray(True)
    for name, edges in zip(node.attrs["axes"], node.attrs["edges"], strict=True):
        x = context.value(name)
        index = jnp.searchsorted(jnp.asarray(edges), x, side="right")-1
        indices.append(jnp.clip(index, 0, len(edges)-2))
        valid = valid & (x >= edges[0]) & (x <= edges[-1])
    values = jnp.asarray(node.attrs["values"])[tuple(indices)]
    return jnp.where(valid, jnp.maximum(values, 0.), 0.)


def hist_pdf_integral(node, context, axes):
    if set(axes) != set(node.attrs["axes"]):
        raise UnsupportedModelError(f"{node.name}: partial RooHistPdf integral over {axes}")
    for name, edges in zip(node.attrs["axes"], node.attrs["edges"], strict=True):
        low, high = context.bounds(name)
        if low > edges[0] or high < edges[-1]:
            raise UnsupportedModelError(f"{node.name}: restricted RooHistPdf integral range")
    # ROOT integrates the stored weights, including negative bins which its
    # point evaluator clips to zero. Keep these two operations distinct.
    return jnp.asarray(node.attrs["sum_weights"])
