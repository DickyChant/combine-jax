"""Combine's fixed-degree Bernstein-to-power-basis PDF implementation."""

from functools import lru_cache
import math
import re

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .histograms import immutable_array
from .root_io import load_root


@lru_cache(maxsize=1)
def bernstein_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/RooBernsteinFast.h"
      namespace combine_jax_io {
      template<int N> class BernsteinAccess : public RooBernsteinFast<N> {
      public:
        BernsteinAccess(RooBernsteinFast<N> const& p) : RooBernsteinFast<N>(p, "combine_jax_bernstein") {}
        RooAbsReal const& observable() const { return this->_x.arg(); }
        RooArgList const& coefficients() const { return this->_coefList; }
        std::vector<std::vector<double>> matrix() const {
          std::vector<std::vector<double>> result(N+1, std::vector<double>(N+1));
          for (int i=0; i<=N; ++i) for (int j=0; j<=N; ++j) result[i][j] = this->_cmatrix(i,j);
          return result;
        }
      };
      }
    '''):
        raise RuntimeError("Cannot compile RooBernsteinFast extraction helper")
    return root.combine_jax_io.BernsteinAccess


def extract_fast_bernstein(root):
    degree = int(re.fullmatch(r"RooBernsteinFast<(\d+)>", str(root.ClassName()))[1])
    view = bernstein_access()[degree](root)
    observable = view.observable()
    if observable.ClassName() != "RooRealVar":
        raise UnsupportedModelError("Fast Bernstein requires a direct RooRealVar observable")
    coefficients = tuple(str(c.GetName()) for c in view.coefficients())
    if len(coefficients) != degree:
        raise UnsupportedModelError("Fast Bernstein requires exactly N explicit coefficients")
    low, high = float(observable.getMin()), float(observable.getMax())
    if high <= low or not math.isfinite(low+high):
        raise UnsupportedModelError("Fast Bernstein requires a finite positive range")
    name = str(observable.GetName())
    return (name, *coefficients), {"observable_name": name, "coefs": coefficients, "range": (low, high),
                                  "matrix": immutable_array([list(row) for row in view.matrix()])}


def fast_bernstein_powers(node, context):
    key = ("bernstein_power_coefficients", node.name)
    if key not in context.cache:
        if any(context.depends_on(c, node.attrs["observable_name"]) for c in node.attrs["coefs"]):
            raise UnsupportedModelError(f"{node.name}: observable-dependent fast Bernstein coefficient")
        coefficients = [jnp.asarray(1.), *context.values(node.attrs["coefs"])]
        context.cache[key] = [sum(weight*c for weight, c in zip(row, coefficients, strict=True))
                              for row in node.attrs["matrix"]]
    return context.cache[key]


def fast_bernstein_value(node, context):
    low, high = node.attrs["range"]
    x = (context.value(node.attrs["observable_name"])-low)/(high-low)
    return sum(coefficient*x**degree for degree, coefficient in enumerate(fast_bernstein_powers(node, context)))


def fast_bernstein_integral(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: fast Bernstein integral over {axes}")
    low, high = node.attrs["range"]
    return (high-low)*sum(coefficient/(degree+1) for degree, coefficient in enumerate(fast_bernstein_powers(node, context)))
