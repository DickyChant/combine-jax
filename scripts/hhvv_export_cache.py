"""Trusted local benchmark cache; not a public or portable file format."""

from pathlib import Path
import pickle
from types import MappingProxyType

from combine_jax.data import DataBinding
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Node
from combine_jax.translators import default_registry


def save_export(likelihood, path, source):
    payload = dict(source=source,
                   nodes={n: (v.class_name, v.inputs, dict(v.attrs)) for n, v in likelihood.nodes.items()},
                   parameters=likelihood.parameters, constraints=likelihood.constraints,
                   profiles=likelihood.profiles, constraint_axes=likelihood.constraint_axes,
                   mask_configuration=likelihood.mask_configuration, offset=likelihood.offset,
                   channels=[(c.name, c.pdf, c.observables, dict(c.data.coordinates), c.data.weights,
                              c.data.storage_class, c.masked) for c in likelihood.channels])
    with Path(path).open("wb") as handle:
        pickle.dump(payload, handle, protocol=5)


def load_export(path, source):
    # Call only for a cache written locally by this script; pickle is executable.
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if payload["source"] != source:
        raise ValueError("Benchmark cache source/configuration does not match")
    nodes = {n: Node(n, cls, inputs, MappingProxyType(attrs)) for n, (cls, inputs, attrs) in payload["nodes"].items()}
    channels = []
    for name, pdf, observables, coordinates, weights, storage, masked in payload["channels"]:
        for array in (*coordinates.values(), weights):
            array.flags.writeable = False
        channels.append(Channel(name, pdf, observables, DataBinding(MappingProxyType(coordinates), weights, storage), masked))
    return Likelihood(MappingProxyType(nodes), default_registry(), payload["parameters"], channels,
                      payload["constraints"], payload["offset"], profiles=payload["profiles"],
                      constraint_axes=payload["constraint_axes"], mask_configuration=payload["mask_configuration"])
