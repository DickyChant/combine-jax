"""Combine double Crystal Ball and error-function/exponential shapes."""

from functools import lru_cache

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, erfc

from .graph import UnsupportedModelError
from .root_io import load_root


@lru_cache(maxsize=1)
def resonance_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/HZZ2L2QRooPdfs.h"
      #include "HiggsAnalysis/CombinedLimit/interface/HWWLVJRooPdfs.h"
      namespace combine_jax_io {
      class DoubleCBAccess : public RooDoubleCB {
      public:
        DoubleCBAccess(RooDoubleCB const& p) : RooDoubleCB(p, "combine_jax_doublecb") {}
        std::vector<std::string> fields() const {
          return {x.arg().GetName(), xp.absArg() ? xp.arg().GetName() : "", mean.arg().GetName(),
                  width.arg().GetName(), alpha1.arg().GetName(), n1.arg().GetName(),
                  alpha2.arg().GetName(), n2.arg().GetName()};
        }
        bool valid() const { return width>0 && alpha1!=0 && alpha2!=0 && n1>0 && n2>0; }
      };
      class ErfExpAccess : public RooErfExpPdf {
      public:
        ErfExpAccess(RooErfExpPdf const& p) : RooErfExpPdf(p, "combine_jax_erfexp") {}
        std::vector<std::string> fields() const {
          return {x.arg().GetName(), c.arg().GetName(), offset.arg().GetName(), width.arg().GetName()};
        }
      };
      }
    '''):
        raise RuntimeError("Cannot compile resonance PDF extraction helpers")
    return root.combine_jax_io


def extract_doublecb(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("DoubleCB forced numerical or named-range normalisation")
    view = resonance_access().DoubleCBAccess(root)
    if not view.valid():
        raise UnsupportedModelError("DoubleCB requires positive width/n1/n2 and nonzero alpha1/2 at export")
    fields = tuple(str(n) or None for n in view.fields())
    return tuple(n for n in fields if n), {"fields": fields}


def doublecb_values(node, context):
    return tuple(context.value(n) if n else jnp.asarray(0.) for n in node.attrs["fields"])


def doublecb_value(node, context):
    x, xp, mean, width, a1, n1, a2, n2 = doublecb_values(node, context)
    t = (x-xp-mean)/width
    abs1, abs2 = jnp.abs(a1), jnp.abs(a2)
    base1 = jnp.where(t < -a1, n1/abs1-abs1-t, 1.)
    base2 = jnp.where((t >= -a1) & (t > a2), n2/abs2-abs2+t, 1.)
    left = jnp.exp(n1*jnp.log(n1/abs1)-.5*a1*a1-n1*jnp.log(base1))
    right = jnp.exp(n2*jnp.log(n2/abs2)-.5*a2*a2-n2*jnp.log(base2))
    result = jnp.where(t < -a1, left, jnp.where(t > a2, right, jnp.exp(-.5*t*t)))
    valid = (width > 0) & (a1 != 0) & (a2 != 0) & (n1 > 0) & (n2 > 0) & (base1 > 0) & (base2 > 0)
    return jnp.where(valid, result, jnp.nan)


def tail_integral(low, high, alpha, power):
    # Preserve v11's logarithmic branch throughout |n-1| <= 1e-5, not
    # merely at n=1. Inactive branches use finite denominators for AD.
    regular = jnp.abs(power-1.) > 1e-5
    q = jnp.where(regular, 1-power, 1.)
    log_low, log_ratio = jnp.log(low), jnp.log(high/low)
    integral = jnp.where(regular, jnp.exp(q*log_low)*jnp.expm1(q*log_ratio)/q, log_ratio)
    return jnp.exp(power*jnp.log(power/alpha)-.5*alpha*alpha)*integral


def doublecb_integral(node, context, axes):
    fields = node.attrs["fields"]
    if len(axes) != 1 or axes[0] not in fields[:2]:
        raise UnsupportedModelError(f"{node.name}: DoubleCB integral requires x or xp")
    axis = axes[0]
    if any(context.depends_on(n, axis) for n in fields if n and n != axis):
        raise UnsupportedModelError(f"{node.name}: DoubleCB shape parameters depend on integration variable")
    if fields.count(axis) != 1:
        raise UnsupportedModelError(f"{node.name}: repeated DoubleCB integration variable")
    x, xp, mean, width, a1, n1, a2, n2 = (context.value(n) if n and n != axis else jnp.asarray(0.) for n in fields)
    low, high = context.bounds(axis)
    low, high = (low-xp, high-xp) if axis == fields[0] else (x-high, x-low)
    t0, t1 = (low-mean)/width, (high-mean)/width
    c0, c1 = jnp.clip(t0, -a1, a2), jnp.clip(t1, -a1, a2)
    central = jnp.where(-a1 < a2, jnp.sqrt(jnp.pi/2)*(erf(c1/jnp.sqrt(2.))-erf(c0/jnp.sqrt(2.))), 0.)
    abs1, abs2 = jnp.abs(a1), jnp.abs(a2)
    b1, b2 = n1/abs1-abs1, n2/abs2-abs2
    l0, l1 = (jnp.where(t0 < -a1, b1-jnp.minimum(t, -a1), 1.) for t in (t1, t0))
    r0, r1 = (jnp.where(t1 > a2, b2+jnp.maximum(t, a2), 1.) for t in (t0, t1))
    # Keep v11's independent left/right integration intervals even when
    # signed alphas make them overlap; this is its advertised integral.
    left = tail_integral(l0, l1, abs1, n1)
    right = tail_integral(r0, r1, abs2, n2)
    valid = ((width > 0) & (a1 != 0) & (a2 != 0) & (n1 > 0) & (n2 > 0)
             & (l0 > 0) & (l1 > 0) & (r0 > 0) & (r1 > 0))
    return jnp.where(valid, width*(left+central+right), jnp.nan)


def extract_erfexp(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("ErfExp forced numerical or named-range normalisation")
    return tuple(str(n) for n in resonance_access().ErfExpAccess(root).fields()), {}


def erfexp_parameters(node, context):
    slope, offset, width = context.values(node.inputs[1:])
    # The v11 class substitutes this slope in both the raw value and integral.
    return jnp.where(slope == 0., -1e-7, slope), offset, jnp.where(width < 1e-2, 1e-2, width)


def root_erf(x):
    # ROOT's Cephes implementation forms erf from erfc outside [-1, 1].
    # Retain its rounding before subsequent 1+erf cancellation; a few ulps
    # in erf otherwise become sizeable relative errors in these tail PDFs.
    tail = 1-jax.lax.optimization_barrier(erfc(x))
    return jax.lax.optimization_barrier(jnp.where(jnp.abs(x) > 1, tail, erf(x)))


def erfexp_value(node, context):
    slope, offset, width = erfexp_parameters(node, context)
    x = context.value(node.inputs[0])
    return jnp.exp(slope*x)*(1+root_erf((x-offset)/width))/2


def erfexp_integral(node, context, axes):
    if axes != (node.inputs[0],):
        raise UnsupportedModelError(f"{node.name}: ErfExp integral requires its direct x argument")
    if any(context.depends_on(n, axes[0]) for n in node.inputs[1:]):
        raise UnsupportedModelError(f"{node.name}: ErfExp parameters depend on integration variable")
    slope, offset, width = erfexp_parameters(node, context)
    low, high = context.bounds(axes[0])
    def primitive(x):
        return (jnp.exp(slope*slope*width*width/4+slope*offset)*root_erf((2*x-slope*width*width-2*offset)/2/width)
                - jnp.exp(slope*x)*root_erf((x-offset)/width)-jnp.exp(slope*x))/(-2*slope)
    return primitive(high)-primitive(low)
