"""Versioned JSON + numeric NPZ transport, without framework objects or pickle."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@dataclass(frozen=True)
class TensorBundle:
    metadata: dict
    arrays: object

    def __post_init__(self):
        metadata = json.loads(json.dumps(self.metadata, allow_nan=False))
        if metadata.get('schema') != 'combine-jax.tensor.v1':
            raise ValueError('Unknown tensor schema')
        arrays = {}
        for name, source in self.arrays.items():
            if not name or '/' in name or '\\' in name:
                raise ValueError(f'Invalid buffer name {name!r}')
            array = np.array(source, copy=True, order='C')
            if array.dtype.kind not in 'bif' or array.dtype.hasobject:
                raise ValueError(f'Buffer {name} is not a numeric array')
            array.flags.writeable = False
            arrays[name] = array
        object.__setattr__(self, 'metadata', metadata)
        object.__setattr__(self, 'arrays', MappingProxyType(arrays))

    @classmethod
    def _from_frozen(cls, metadata, arrays):
        """Internal composition of already-owned immutable bundle buffers.

        The public constructor still defensively copies external arrays. Import
        composition and unpacking share existing frozen buffers; they never
        mutate their numerical contents or restore write access.
        """
        metadata=json.loads(json.dumps(metadata,allow_nan=False))
        if metadata.get('schema')!='combine-jax.tensor.v1':
            raise ValueError('Unknown tensor schema')
        frozen={}
        for name,array in arrays.items():
            if not name or '/' in name or '\\' in name:
                raise ValueError(f'Invalid buffer name {name!r}')
            if (not isinstance(array,np.ndarray) or array.dtype.kind not in 'bif'
                    or array.dtype.hasobject or array.flags.writeable or not array.flags.c_contiguous):
                raise ValueError(f'Internal buffer {name} must already be frozen and contiguous')
            frozen[name]=array
        result=object.__new__(cls)
        object.__setattr__(result,'metadata',metadata)
        object.__setattr__(result,'arrays',MappingProxyType(frozen))
        return result

    @property
    def structure_key(self):
        """Code/layout identity; excludes buffer contents, labels and provenance.

        Shapes remain part of the key because ordinary XLA executables specialize
        on shapes. Padding/bucketing across model sizes is a subsequent policy.
        """
        return json_hash(dict(schema=self.metadata['schema'], kind=self.metadata['kind'],
            schedule=self.metadata['schedule'],
            buffers={n:dict(shape=list(a.shape), dtype=a.dtype.str) for n,a in sorted(self.arrays.items())}))

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)
        path = directory/'arrays.npz'
        np.savez_compressed(path, **self.arrays)
        document = dict(metadata=self.metadata, structure_key=self.structure_key,
                        metadata_sha256=json_hash(self.metadata),
                        arrays_sha256=file_hash(path))
        (directory/'model.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        document = json.loads((directory/'model.json').read_bytes())
        if json_hash(document['metadata']) != document['metadata_sha256']:
            raise ValueError('Tensor metadata checksum mismatch')
        path = directory/'arrays.npz'
        if file_hash(path) != document['arrays_sha256']:
            raise ValueError('Tensor array archive checksum mismatch')
        # NPZ members are newly owned arrays. Freezing them here avoids keeping
        # a second complete model while the public constructor defensively
        # copies external buffers. Normalize one member at a time to C order;
        # the internal factory still validates numeric dtype, names and schema.
        buffers = {}
        with np.load(path, allow_pickle=False) as archive:
            for name in archive.files:
                array = archive[name]
                if not array.flags.c_contiguous:
                    array = np.ascontiguousarray(array)
                array.flags.writeable = False
                buffers[name] = array
        result = cls._from_frozen(document['metadata'], buffers)
        if result.structure_key != document['structure_key']:
            raise ValueError('Tensor structure key mismatch')
        return result
