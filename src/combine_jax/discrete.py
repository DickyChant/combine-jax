"""Frozen RooMultiPdf selections and v11 channel correction bookkeeping."""

import math

from .graph import UnsupportedModelError


def extract_category(root):
    return (), {"index": int(root.getCurrentIndex()), "label": str(root.getCurrentLabel()),
                "states": tuple((str(t.first), int(t.second)) for t in root)}


def extract_multipdf(root):
    index = int(root.getCurrentIndex())
    count = int(root.getNumPdfs())
    if not 0 <= index < count:
        raise UnsupportedModelError(f"MultiPdf index {index} is outside its {count} alternatives")
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError("MultiPdf forced numerical or named-range normalisation")
    correction = float(root.getCorrection())
    if not math.isfinite(correction):
        raise UnsupportedModelError("Nonfinite MultiPdf correction")
    selected = str(root.getCurrentPdf().GetName())
    category = str(root.indexCategory().arg().GetName())
    return (selected, category), {"selected": selected, "category": category, "index": index,
                                  "alternatives": tuple(str(root.getPdf(i).GetName()) for i in range(count)),
                                  "correction": correction}


def active_dependencies(nodes, roots):
    """Follow exported inputs; inactive MultiPdf alternatives are provenance only."""
    active, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name in active:
            continue
        active.add(name)
        pending.extend(nodes[name].inputs)
    return active


def channel_correction(nodes, name):
    node = nodes[name]
    # CachingAddNLL recursively flattens RooAddPdf children, but does not
    # recurse into arbitrary PDF products or the alternatives of a MultiPdf.
    if node.class_name == "RooMultiPdf":
        return node.attrs["correction"]
    if node.class_name == "RooAddPdf":
        return sum(channel_correction(nodes, child) for child in node.attrs["pdfs"])
    if node.class_name == "RooRealSumPdf":
        return sum(nodes[f].attrs["correction"] for f in node.attrs["functions"] if nodes[f].class_name == "RooMultiPdf")
    return 0.
