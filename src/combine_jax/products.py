"""Factorised RooProdPdf densities with explicit conditional normalisation sets."""

import math

import jax.numpy as jnp

from .graph import UnsupportedModelError
from .root_io import stored_member


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
    return tuple(inputs), {"normalisations": tuple(specifications)}


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
    normalisations = product_axes(node, context, axes)
    # RooProdPdf omits factors with no requested normalisation observables,
    # including constraints and factors declared conditional on all axes.
    return math.prod(context.density(name, normalised)
                     for name, normalised in zip(node.inputs, normalisations, strict=True) if normalised)


def pdf_product_integral(node, context, axes):
    used, result = set(), jnp.asarray(1.)
    for name in node.inputs:
        dependent = tuple(a for a in axes if context.depends_on(name, a))
        if used.intersection(dependent):
            raise UnsupportedModelError(f"{node.name}: raw product integral with overlapping dependencies")
        used.update(dependent)
        result = result * context.integral(name, dependent)
    return result * math.prod(context.bounds(a)[1]-context.bounds(a)[0] for a in set(axes)-used)
