"""Canonical identity of channel coordinates and weights consumed by a model."""
import hashlib
import json

import numpy as np


def binding_identity(bindings):
    digest = hashlib.sha256()
    rows = 0
    for label, binding in sorted(bindings.items()):
        columns = {**dict(binding.coordinates), '__weight__': binding.weights}
        digest.update(json.dumps([label, sorted(columns)], separators=(',', ':')).encode())
        for name, values in sorted(columns.items()):
            array = np.asarray(values, dtype='<f8')
            digest.update(json.dumps([name, list(array.shape)], separators=(',', ':')).encode())
            digest.update(array.tobytes(order='C'))
        rows += len(binding.weights)
    return dict(sha256=digest.hexdigest(), channels=len(bindings), rows=rows,
                scope='Sorted channel names and observable names; original row order; float64 coordinates and weights, including zero weights.')


def workspace_dataset_identity(workspace, data):
    from .data import bind_channels
    config = workspace.obj('ModelConfig')
    dataset = workspace.data(data) if isinstance(data, str) else data
    if not dataset:
        raise ValueError('Requested dataset is missing')
    pdf = config.GetPdf()
    category = pdf.indexCat().Clone()
    channels = {}
    for index in range(category.numTypes()):
        category.setBin(index)
        name = str(category.getLabel())
        channel = pdf.getPdf(name)
        if channel:
            channels[name] = sorted(str(v.GetName()) for v in channel.getObservables(config.GetObservables())
                                    if v.InheritsFrom('RooAbsReal'))
    return binding_identity(bind_channels(dataset, str(pdf.indexCat().GetName()), channels))
