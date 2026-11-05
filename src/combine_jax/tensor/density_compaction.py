"""Remove unreferenced density slots after folding, without numerical changes.

This operates on framework-neutral bundles. It retains every runtime read and
write, including padding, masked operands and all discrete alternatives. It is
an explicit storage transformation, not a new folding or liveness policy.
"""
from copy import deepcopy

import numpy as np

from .bundle import TensorBundle


def density_index_buffers(bundle):
    """Names of buffers indexing the density arena, not other index domains."""
    names = {'fast_outputs'}
    for batch in bundle.metadata['program']:
        names.add(batch['buffers']['outputs'])
        if 'children' in batch['buffers']:
            names.add(batch['buffers']['children'])
    for batch in bundle.metadata['nll_program']:
        names.add(batch['buffers']['densities'])
    # fast_inputs index query storage; NLL outputs index channels; coefs/rates
    # index scalars. Grid metadata contains row counts and labels, not offsets.
    return sorted(names)


def compact_density_storage(bundle):
    """Return an immutable bundle with a monotonic one-to-one slot remapping.

    Unchanged buffers are shared; neither the input nor any floating value is
    modified. Full validation before and after catches unsupported operations,
    bad indices, and overlapping output scatters. Empty arenas are allowed when
    all channels have folded. An already compact bundle is returned unchanged.
    """
    from .binned_channels import validate_binned_channels

    validate_binned_channels(bundle)
    source = bundle.arrays
    size = len(source['initial_values'])
    if size > np.iinfo(np.int32).max:
        raise ValueError('Density arena exceeds the int32 index domain')
    names = density_index_buffers(bundle)
    live = np.zeros(size, dtype=bool)
    for name in names:
        live[source[name].reshape(-1)] = True
    retained = int(np.count_nonzero(live))
    if retained == size:
        return bundle
    mapping = np.empty(size, dtype=np.int32)
    np.cumsum(live, dtype=np.int32, out=mapping)
    mapping -= 1
    arrays = dict(source)
    arrays['initial_values'] = source['initial_values'][live]
    for name in names:
        arrays[name] = mapping[source[name]]
    del mapping, live
    for name in ['initial_values', *names]:
        arrays[name].flags.writeable = False
    metadata = deepcopy(bundle.metadata)
    metadata['density_storage_compaction'] = dict(
        algorithm='retain_all_runtime_reads_and_writes_monotonic_v1',
        source_slots=size, retained_slots=retained, removed_slots=size-retained,
        removed_float64_bytes=8*(size-retained))
    result = TensorBundle._from_frozen(metadata, arrays)
    validate_binned_channels(result)
    return result


def compact_model_density_storage(model):
    """Apply exact density compaction while preserving all other model metadata."""
    from .model import unpack_model

    components = unpack_model(model)
    if 'binned' not in components:
        return model
    original = components['binned']
    compact = compact_density_storage(original)
    if compact is original:
        return model
    metadata = deepcopy(model.metadata)
    metadata['components']['binned'] = compact.metadata
    arrays = {**model.arrays, **{'binned__'+n: a for n, a in compact.arrays.items()}}
    result = TensorBundle._from_frozen(metadata, arrays)
    unpack_model(result)
    return result
