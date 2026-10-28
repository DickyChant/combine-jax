"""ROOT extraction registry and a pure JAX evaluation context."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Mapping

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from .graph import UnsupportedModelError, walk


@dataclass(frozen=True)
class Node:
    name: str
    class_name: str
    inputs: tuple[str, ...]
    attrs: Mapping


@dataclass(frozen=True)
class Translator:
    extract: Callable
    value: Callable
    integral: Callable | None = None
    density: Callable | None = None
    log_value: Callable | None = None


class Registry:
    def __init__(self):
        self.translators = {}

    def register(self, class_name, extract, value, *, integral=None, density=None, log_value=None):
        if class_name in self.translators:
            raise ValueError(f"Duplicate translator for {class_name}")
        self.translators[class_name] = Translator(extract, value, integral, density, log_value)

    def export(self, roots, observables=()):
        import ROOT

        root_nodes = {}
        for root in roots:
            for name, node in walk(root).items():
                if name in root_nodes and ROOT.addressof(root_nodes[name]) != ROOT.addressof(node):
                    raise ValueError(f"Distinct graph nodes share the name {name!r}")
                root_nodes[name] = node
        unsupported = {}
        for name, node in root_nodes.items():
            if node.ClassName() not in self.translators:
                unsupported.setdefault(str(node.ClassName()), []).append(name)
        if unsupported:
            detail = "; ".join(f"{cls}: {', '.join(ns[:5])} ({len(ns)} nodes)"
                               for cls, ns in sorted(unsupported.items()))
            raise UnsupportedModelError("Untranslated classes: " + detail)
        result = {}
        for name, root in root_nodes.items():
            try:
                inputs, attrs = self.translators[root.ClassName()].extract(root)
            except UnsupportedModelError as exc:
                raise UnsupportedModelError(f"{name} ({root.ClassName()}): {exc}") from exc
            if name in observables:
                attrs = {**attrs, "observable": True}
            result[name] = Node(name, str(root.ClassName()), tuple(inputs), MappingProxyType(attrs))
        return MappingProxyType(result)


class Evaluation:
    """One trace/evaluation; memoisation is scoped to this parameter/data binding."""

    def __init__(self, nodes, registry, parameter_names, theta, data=None, overrides=None):
        self.nodes = nodes
        self.registry = registry
        self.parameters = dict(zip(parameter_names, theta, strict=True))
        self.parameters.update(overrides or {})
        self.data = data if data is not None else {}
        self.cache = {}

    def value(self, name):
        key = ("value", name)
        if key not in self.cache:
            node = self.nodes[name]
            self.cache[key] = self.registry.translators[node.class_name].value(node, self)
        return self.cache[key]

    def integral(self, name, axes):
        axes = tuple(sorted(axes))
        if not axes:
            return self.value(name)
        node = self.nodes[name]
        translator = self.registry.translators[node.class_name]
        if translator.integral is None:
            raise UnsupportedModelError(f"{name} ({node.class_name}): no integral over {axes}")
        key = ("integral", name, axes)
        if key not in self.cache:
            self.cache[key] = translator.integral(node, self, axes)
        return self.cache[key]

    def density(self, name, axes):
        axes = tuple(sorted(axes))
        key = ("density", name, axes)
        if key not in self.cache:
            node = self.nodes[name]
            translator = self.registry.translators[node.class_name]
            self.cache[key] = (translator.density(node, self, axes) if translator.density
                               else self.value(name) / self.integral(name, axes))
        return self.cache[key]

    def values(self, inputs):
        return [self.value(name) for name in inputs]

    def log_value(self, name):
        node = self.nodes[name]
        translator = self.registry.translators[node.class_name]
        return (translator.log_value(node, self) if translator.log_value
                else jnp.log(self.value(name)))

    def bounds(self, name):
        node = self.nodes[name]
        if node.class_name != "RooRealVar":
            raise UnsupportedModelError(f"Integral variable {name} must be a RooRealVar")
        return node.attrs["minimum"], node.attrs["maximum"]

    def depends_on(self, name, variable):
        return name == variable or any(self.depends_on(child, variable)
                                       for child in self.nodes[name].inputs)


def leaf_value(node, context):
    if node.name in context.parameters:
        return context.parameters[node.name]
    if node.attrs.get("observable"):
        if node.name not in context.data:
            raise ValueError(f"Unbound observable {node.name}")
        return jnp.asarray(context.data[node.name])
    return jnp.asarray(node.attrs["value"], dtype=jnp.float64)
