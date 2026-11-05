"""Combine HMuMu analytic shape extraction."""
from functools import lru_cache
import math

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .root_io import load_root
from . import twoexp_math


@lru_cache(maxsize=1)
def hmumu_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/HMuMuRooPdfs.h"
      namespace combine_jax_io {
      class TwoExpAccess : public RooSumTwoExpPdf {
      public:
        TwoExpAccess(RooSumTwoExpPdf const& p) : RooSumTwoExpPdf(p,"combine_jax_twoexp") {}
        std::vector<std::string> fields() const {
          return {x.arg().GetName(),a1.arg().GetName(),a2.arg().GetName(),f.arg().GetName(),
                  m.absArg() ? m.arg().GetName() : ""};
        }
        bool useOffset() const { return offset; }
        double low() const { return x.min(); }
        double high() const { return x.max(); }
      };
      class ModZAccess : public RooModZPdf {
      public:
        ModZAccess(RooModZPdf const& p) : RooModZPdf(p,"combine_jax_modz") {}
        std::vector<std::string> fields() const {
          return {x.arg().GetName(),a.arg().GetName(),
                  b.absArg() ? b.arg().GetName() : "", c.absArg() ? c.arg().GetName() : "",
                  m.absArg() ? m.arg().GetName() : "", w.absArg() ? w.arg().GetName() : ""};
        }
        RooArgList const& coefficients() const { return bernCoef; }
        double low() const { return x.min(); }
        double high() const { return x.max(); }
      };
      }
    '''):
        raise RuntimeError('Cannot compile HMuMu extraction helpers')
    return root.combine_jax_io


def extract_twoexp(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError('SumTwoExp forced numerical or named-range normalization')
    view = hmumu_access().TwoExpAccess(root)
    fields = tuple(str(n) or None for n in view.fields())
    low, high = float(view.low()), float(view.high())
    if not math.isfinite(low+high) or high <= low:
        raise UnsupportedModelError('SumTwoExp requires a finite positive normalization interval')
    return tuple(n for n in fields if n), dict(fields=fields, offset=bool(view.useOffset()), reference_range=(low, high))


def twoexp_parameters(node, context):
    fields = node.attrs['fields']
    mean = context.value(fields[4]) if fields[4] else jnp.asarray(91.2)
    if not node.attrs['offset']: mean = jnp.asarray(0.)
    return (*context.values(fields[1:4]), mean)


def twoexp_value(node, context):
    return twoexp_math.value(context.value(node.inputs[0]), *twoexp_parameters(node, context),
                             *node.attrs['reference_range'], jnp)


def twoexp_integral(node, context, axes):
    if axes != (node.inputs[0],) or any(context.depends_on(n, axes[0]) for n in node.inputs[1:]):
        raise UnsupportedModelError(f'{node.name}: SumTwoExp integral requires independent shape parameters')
    low, high = context.bounds(axes[0])
    if (low, high) == node.attrs['reference_range']:
        return jnp.asarray(1.)
    return twoexp_math.integral(low, high, *node.attrs['reference_range'], *twoexp_parameters(node, context), jnp)


def extract_modz(root):
    from .romberg import extract_config
    if root.normRange():
        raise UnsupportedModelError('ModZ named normalization range is not translated')
    view=hmumu_access().ModZAccess(root)
    fields=tuple(str(n) or None for n in view.fields())
    coefficients=tuple(str(n.GetName()) for n in view.coefficients())
    low,high=float(view.low()),float(view.high())
    if not math.isfinite(low+high) or high<=low:
        raise UnsupportedModelError('ModZ requires finite positive observable bounds')
    return tuple(n for n in fields if n)+coefficients,dict(fields=fields,coefficients=coefficients,
        reference_range=(low,high),integration_config=extract_config(root))


def modz_parameters(node,context):
    defaults=(0.,0.,2.,91.2,2.5)
    parameters=[context.value(n) if n else jnp.asarray(v) for n,v in zip(node.attrs['fields'][1:],defaults,strict=True)]
    parameters.extend(context.values(node.attrs['coefficients']))
    return jnp.stack(parameters)


def modz_value(node,context):
    from .modz_math import value
    return value(context.value(node.inputs[0]),modz_parameters(node,context),*node.attrs['reference_range'],jnp)


def modz_integral(node,context,axes):
    from .modz_math import value
    from .romberg import RombergConfig
    from .romberg_jax import integrate
    if axes!=(node.inputs[0],) or any(context.depends_on(n,axes[0]) for n in node.inputs[1:]):
        raise UnsupportedModelError(f'{node.name}: ModZ requires an independent direct integration axis')
    parameters=modz_parameters(node,context)
    if parameters.ndim!=1:
        raise UnsupportedModelError(f'{node.name}: conditional ModZ normalization needs batched lowering')
    function=lambda p,x:value(x,p,*node.attrs['reference_range'],jnp)
    return integrate(function,parameters,*context.bounds(axes[0]),RombergConfig(**node.attrs['integration_config']))
