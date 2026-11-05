"""Factorised RooProdPdf densities with explicit conditional normalisation sets."""

import math
from functools import lru_cache

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .root_io import stored_member, load_root
from .product_axes import simple_product_axes


@lru_cache(maxsize=1)
def simple_product_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/SimpleProdPdf.h"
      namespace combine_jax_io {
      class SimpleProdAccess : public SimpleProdPdf {
      public:
        SimpleProdAccess(SimpleProdPdf const& p) : SimpleProdPdf(p, "combine_jax_simpleprod") {}
        std::vector<int> settings() const { return _pdfSettings; }
      };
      }
    '''):
        raise RuntimeError('Cannot compile SimpleProdPdf extraction helper')
    return root.combine_jax_io.SimpleProdAccess


def extract_simple_product(root):
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError('SimpleProdPdf forced numerical or named-range normalization')
    inputs = tuple(str(child.GetName()) for child in root.pdfList())
    settings = tuple(int(v) for v in simple_product_access()(root).settings())
    if len(inputs) != len(settings):
        raise UnsupportedModelError('SimpleProdPdf settings/factor count mismatch')
    # The vector-settings constructor stores the factor indices, not the
    # passed flags. Read the persisted settings verbatim; only ==1 is special.
    return inputs, {'settings': settings}


def simple_product_integral(node, context, axes):
    selected = simple_product_axes(node.inputs, node.attrs['settings'], axes, context.depends_on)
    # This is the class's advertised factorized integral, including overlapping
    # factors and axes absent from all children; do not substitute a joint integral.
    return math.prod(context.integral(name, chosen) for name, chosen in zip(node.inputs, selected, strict=True))


def extract_pdf_product(root):
    if (not root.selfNormalized() or root.getForceNumInt() or root.normRange()
            or stored_member(root, "_cutOff") != 0
            or stored_member(root, "_refRangeName") is not None
            or stored_member(root, "_defNormSet")["_list"]):
        raise UnsupportedModelError("Product cutoff, reference/default normalisation or nonstandard normalisation mode")
    inputs, specifications = [], []
    for child in root.pdfList():
        inputs.append(str(child.GetName()))
        specified = root.findPdfNSet(child)
        kind = str(specified.GetName()) if specified and len(specified) else "all"
        if kind not in ("all", "nset", "cset"):
            raise UnsupportedModelError(f"Unknown product normalisation set kind {kind!r}")
        specifications.append((kind, tuple(str(v.GetName()) for v in specified) if specified else ()))
    attrs={"normalisations": tuple(specifications)}
    from .joint_spline_product import extract
    joint=extract(root,specifications)
    if joint is not None:attrs['joint_spline_cheby']=joint
    from .joint_numeric_product import extract as extract_numeric
    joint=extract_numeric(root,specifications)
    if joint is not None:attrs['joint_numeric_product']=joint
    return tuple(inputs), attrs


def pdf_product_value(node, context):
    return math.prod(context.value(name) for name in node.inputs)


def product_axes(node, context, axes):
    key = ("product_axes", node.name, axes)
    if key in context.cache:
        return context.cache[key]
    dependencies, normalisations, owners = [], [], {}
    for index, (name, (kind, specified)) in enumerate(zip(node.inputs, node.attrs["normalisations"], strict=True)):
        dependent = {a for a in axes if context.depends_on(name, a)}
        selected = {a for a in specified if context.depends_on(name, a)}
        if kind == "cset":
            selected = dependent - selected
        # RooProdPdf treats an empty effective specification as an ordinary
        # factor. In particular, removing every requested axis with cset does
        # not drop that factor: it can create an overlapping joint integral.
        normalised = dependent & selected if selected else dependent
        for axis in normalised:
            if axis in owners:
                raise UnsupportedModelError(f"{node.name}: overlapping product normalisation over {axis}")
            owners[axis] = index
        dependencies.append(dependent)
        normalisations.append(tuple(sorted(normalised)))
    # Conditional chains can be integrated in reverse dependency order. A
    # cycle instead needs a joint integral of the coupled factors.
    incoming = [{owners[a] for a in dependent-set(normalised) if a in owners}
                if normalised else set()
                for dependent, normalised in zip(dependencies, normalisations, strict=True)]
    pending = set(range(len(node.inputs)))
    while pending:
        ready = {i for i in pending if not incoming[i] & pending}
        if not ready:
            raise UnsupportedModelError(f"{node.name}: cyclic conditional product requires a joint integral")
        pending -= ready
    result = tuple(normalisations)
    context.cache[key] = result
    return result


def pdf_product_density(node, context, axes):
    if not axes:
        return pdf_product_value(node, context)
    if 'joint_spline_cheby' in node.attrs:
        from .joint_spline_product import density
        return density(node,context,axes)
    if 'joint_numeric_product' in node.attrs:
        from .joint_numeric_product import density
        return density(node,context,axes)
    normalisations = product_axes(node, context, axes)
    # RooProdPdf omits factors with no requested normalisation observables,
    # including constraints and factors declared conditional on all axes.
    return math.prod(context.density(name, normalised)
                     for name, normalised in zip(node.inputs, normalisations, strict=True) if normalised)


def pdf_product_integral(node, context, axes):
    joint=node.attrs.get('joint_spline_cheby') or node.attrs.get('joint_numeric_product')
    if joint and axes==(joint['axis'],):
        # RooProdPdf advertises this factorized raw integral. Its normalized
        # density separately uses the generated joint-product denominator.
        # Nested self-normalized RooProdPdf factors contribute one here.
        # The enclosing density still uses the full joint raw-product norm.
        units=joint.get('unit_integral_factors',())
        return math.prod(context.integral(name,axes) for name in node.inputs if name not in units)
    used, result = set(), jnp.asarray(1.)
    for name in node.inputs:
        dependent = tuple(a for a in axes if context.depends_on(name, a))
        if used.intersection(dependent):
            raise UnsupportedModelError(f"{node.name}: raw product integral with overlapping dependencies")
        used.update(dependent)
        result = result * context.integral(name, dependent)
    return result * math.prod(context.bounds(a)[1]-context.bounds(a)[0] for a in set(axes)-used)
