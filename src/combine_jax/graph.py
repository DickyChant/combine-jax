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


def walk(root):
    """Return dependency-first unique nodes; detect cycles and name collisions."""
    import ROOT

    nodes, identities, visiting = {}, {}, set()

    def visit(node):
        name = str(node.GetName())
        address = ROOT.addressof(node)
        if name in identities and identities[name] != address:
            raise ValueError(f"Distinct graph nodes share the name {name!r}")
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

    visit(root)
    return nodes


def parameters(pdf, observables=(), globals_=(), pois=()):
    excluded = set(observables) | set(globals_)
    result = []
    for name, node in walk(pdf).items():
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
    nodes = walk(pdf)
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
                channels[label] = dict(Counter(str(n.ClassName()) for n in walk(child).values()))
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
