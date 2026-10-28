"""Load only the neutral fit-state arrays needed by the native CLI boundary.

The native executable reads the ROOT workspace. Holding its tensor likelihood
buffers in the monitoring parent wastes memory and inflates the combined guard.
This view is not a tensor likelihood and must not be passed to a backend.
"""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from combine_jax.tensor.bundle import json_hash
from combine_jax.tensor.parameter_info import model_parameter_info
from tensor_benchmark_common import sha


SCALAR_FIELDS={'parameter_initial','parameter_fixed','parameter_indices',
               'parameter_slots','initial_values'}
PROFILE_FIELDS={'templates__profile_minimum','templates__profile_maximum'}


def load_native_fit_model(directory):
    directory=Path(directory)
    document=json.loads((directory/'model.json').read_text())
    metadata=document['metadata']
    if metadata.get('schema')!='combine-jax.tensor.v1' or json_hash(metadata)!=document['metadata_sha256']:
        raise ValueError('Native fit metadata checksum/schema mismatch')
    archive=directory/'arrays.npz'
    if sha(archive)!=document['arrays_sha256']:
        raise ValueError('Native fit array archive checksum mismatch')
    required={'scalars__'+name for name in SCALAR_FIELDS}
    if metadata['components'].get('templates',{}).get('profile_names'):
        required|=PROFILE_FIELDS
    with np.load(archive,allow_pickle=False) as saved:
        if not required<=set(saved.files):
            raise ValueError('Native fit arrays are missing')
        names=required|{name for name in saved.files if name.startswith('parameter_info__')}
        arrays={name:saved[name] for name in sorted(names)}
        total_count=len(saved.files)
    for array in arrays.values():
        if array.dtype.kind not in 'bif' or array.dtype.hasobject:
            raise ValueError('Native fit state must be numeric')
        array.flags.writeable=False
    model=SimpleNamespace(metadata=metadata,arrays=arrays)
    model_parameter_info(model)
    evidence=dict(script_sha256=sha(__file__),arrays_sha256=document['arrays_sha256'],
        model_sha256=sha(directory/'model.json'),selected_arrays=sorted(arrays),
        selected_array_bytes=sum(a.nbytes for a in arrays.values()),archive_array_count=total_count,
        scope='Complete archive checksum verified; only fit-state arrays decompressed. Native likelihood comes from the ROOT workspace. Conservative combined RSS guard unchanged.')
    return model,evidence
