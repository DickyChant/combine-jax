"""Exact observed-row compaction for extended shape terms, separate from normalization."""
import numpy as np

from ..graph import UnsupportedModelError


def padded_size(count):
    return 1 << (int(count)-1).bit_length() if count else 0


def prepare_shape_data(channels,*,compact=True):
    """Drop zero weights and pad with duplicate coordinates carrying zero weight.

    No template, observable range, normalization grid or profile input is changed.
    Only the extended RooAddPdf consumer uses these rows. Original numeric inputs
    are validated before selection, so invalid zero-weight data cannot be hidden.
    """
    output=[];layouts=[];arrays={}
    for i,ch in enumerate(channels):
        if ch.get('masked',False):
            output.append(ch);layouts.append(dict(name=ch['name'],masked=True));continue
        weights=np.asarray(ch['weights'],dtype=np.float64)
        if weights.ndim!=1 or not np.all(np.isfinite(weights)) or np.any(weights<0):
            raise UnsupportedModelError(f'{ch["name"]}: invalid weights')
        coordinates=ch['coordinates'];names=ch['observables']
        if any(n not in coordinates or np.shape(coordinates[n])!=weights.shape or not np.all(np.isfinite(coordinates[n])) for n in names):
            raise UnsupportedModelError(f'{ch["name"]}: invalid data coordinates')
        selected=np.flatnonzero(weights>0) if compact else np.arange(len(weights))
        count=len(selected);size=padded_size(count) if compact else count
        indices=np.full(size,int(selected[0]) if count else 0,dtype=np.int32);indices[:count]=selected
        mask=np.arange(size)<count
        packed_weights=np.zeros(size);packed_weights[:count]=weights[selected]
        packed_coordinates={n:np.asarray(coordinates[n],dtype=np.float64)[indices] for n in names}
        index_key=f'data_source_indices_{i}';mask_key=f'data_source_mask_{i}'
        arrays[index_key]=indices;arrays[mask_key]=mask
        output.append({**ch,'weights':packed_weights,'coordinates':packed_coordinates})
        layouts.append(dict(name=ch['name'],masked=False,original_rows=len(weights),positive_rows=int(np.count_nonzero(weights)),
            retained_rows=count,packed_rows=size,source_indices=index_key,source_mask=mask_key,
            observed_sum=float(np.sum(weights))))
    return output,layouts,arrays


def validate_shape_data(metadata,arrays):
    if 'data_layouts' not in metadata:return  # Earlier serialized artifacts remain valid.
    compact=metadata['compact_shape_data'];layouts=metadata['data_layouts']
    if type(compact) is not bool or [q['name'] for q in layouts]!=metadata['names']:
        raise ValueError('Invalid shape data layout names/policy')
    for q in layouts:
        if q['masked']:continue
        n,p,k,size=(q[v] for v in ('original_rows','positive_rows','retained_rows','packed_rows'))
        if any(type(v) is not int or v<0 for v in (n,p,k,size)) or p>n or k!=(p if compact else n) or size!=(padded_size(k) if compact else k):
            raise ValueError('Invalid shape data row counts')
        indices,mask=arrays[q['source_indices']],arrays[q['source_mask']]
        if indices.dtype!=np.int32 or indices.shape!=(size,) or np.any(indices<0) or np.any(indices>=n):
            raise ValueError('Invalid shape data source indices')
        if mask.dtype!=bool or mask.shape!=(size,) or not np.array_equal(mask,np.arange(size)<k):
            raise ValueError('Invalid shape data padding mask')
        if k and (np.any(np.diff(indices[:k])<=0) or np.any(indices[k:]!=indices[0])):
            raise ValueError('Invalid shape data selection/padding')
        if not np.isfinite(q['observed_sum']) or q['observed_sum']<0:raise ValueError('Invalid observed sum')
