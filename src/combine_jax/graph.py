"""Exact-class graph traversal and stable continuous parameter metadata."""

from collections import Counter
from dataclasses import asdict, dataclass


class UnsupportedModelError(ValueError):
    """A named node or configuration has no certified translation."""


@dataclass(frozen=True)
class Parameter:
    name: str
    value: float
    minimum: float
    maximum: float
    error: float
    role: str


def names(collection):
    return {str(item.GetName()) for item in collection} if collection else set()


def _observable_geometry(node):
    """Exact import-relevant geometry for a data-bound real observable clone."""
    if node.ClassName() != "RooRealVar":
        return None
    binnings = []
    for name in sorted({"", *(str(n) for n in node.getBinningNames())}):
        binning = node.getBinning(name)
        if binning.isParameterized():
            return None
        binnings.append((name, tuple(float(binning.binLow(i)) for i in range(binning.numBins()))
                         + (float(binning.highBound()),)))
    return bool(node.isConstant()), tuple(binnings)


def walk_roots(roots, *, observables=()):
    """Return dependency-first nodes, checking identity across all roots.

    RooFit can own separate copies of a dataset coordinate in each channel.
    They bind the same named data column. Only explicitly declared real
    observables with identical static ranges/binnings can share a graph key;
    parameters and other same-named objects still require pointer identity.
    """
    import ROOT

    nodes, identities, visiting, geometry = {}, {}, set(), {}
    observables = set(observables)

    def visit(node):
        name = str(node.GetName())
        address = ROOT.addressof(node)
        if name in identities and identities[name] != address:
            if name in observables and name in nodes:
                if name not in geometry:
                    geometry[name] = _observable_geometry(nodes[name])
                expected = geometry[name]
                if expected is not None and _observable_geometry(node) == expected:
                    return
            raise ValueError(f"Distinct graph nodes share the name {name!r}; "
                             "only equivalent data observables may alias")
        if name in visiting:
            raise ValueError(f"Cycle at {name} ({node.ClassName()})")
        if name in nodes:
            return
        identities[name] = address
        visiting.add(name)
        for child in node.servers():
            visit(child)
        visiting.remove(name)
        nodes[name] = node

    for root in roots:
        visit(root)
    return nodes


def walk(root, *, observables=()):
    """Return dependency-first unique nodes; detect cycles and name collisions."""
    return walk_roots((root,), observables=observables)


def parameters(pdf, observables=(), globals_=(), pois=()):
    observables = set(observables)
    excluded = observables | set(globals_)
    result = []
    for name, node in walk(pdf, observables=observables).items():
        if node.ClassName() == "RooRealVar" and name not in excluded and not node.isConstant():
            result.append(Parameter(name, float(node.getVal()), float(node.getMin()),
                                    float(node.getMax()), float(node.getError()),
                                    "poi" if name in pois else "nuisance"))
    return tuple(sorted(result, key=lambda p: p.name))


def inspect_workspace(workspace, model="ModelConfig", data="data_obs"):
    config = workspace.obj(model)
    dataset = workspace.data(data)
    if not config or not dataset:
        raise ValueError(f"Missing {model} or dataset {data}")
    pdf = config.GetPdf()
    observable_names = names(config.GetObservables())
    nodes = walk(pdf, observables=observable_names)
    categories = {}
    for name, node in nodes.items():
        if node.InheritsFrom("RooAbsCategory"):
            categories[name] = {str(item.first): int(item.second) for item in node}
    channels = {}
    if pdf.InheritsFrom("RooSimultaneous"):
        for state in pdf.indexCat():
            label = str(state.first)
            child = pdf.getPdf(label)
            if child:
                channels[label] = dict(Counter(str(n.ClassName()) for n in
                    walk(child, observables=observable_names).values()))
    return {
        "workspace": str(workspace.GetName()), "pdf": str(pdf.GetName()),
        "pdf_class": str(pdf.ClassName()), "node_count": len(nodes),
        "classes": dict(sorted(Counter(str(n.ClassName()) for n in nodes.values()).items())),
        "observables": sorted(names(config.GetObservables())),
        "global_observables": sorted(names(config.GetGlobalObservables())),
        "parameters": [asdict(p) for p in parameters(pdf, names(config.GetObservables()),
                        names(config.GetGlobalObservables()), names(config.GetParametersOfInterest()))],
        "categories": categories, "channels": channels,
        "data": {"name": data, "class": str(dataset.ClassName()),
                 "entries": int(dataset.numEntries()), "sum_weights": float(dataset.sumEntries()),
                 "weighted": bool(dataset.isWeighted())},
    }
