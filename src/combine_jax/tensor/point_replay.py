"""Exact parameter bindings for replaying fixed-choice points after profile import."""
import numpy as np

from .parameter_info import validate_parameter_info


def extend_inactive_parameters(info, source, active_mask):
    """Keep every original coordinate; only add proven-inactive initial values.

    ``active_mask`` must describe the full NLL at the explicitly selected
    discrete choices. This does not validate other choices or invent values for
    newly active parameters. All original ranges, free coordinates, sampling and
    finite-difference steps remain unchanged. No clipping or resampling occurs.
    """
    validate_parameter_info(info)
    names=info.metadata['names'];old=source['parameter_names'].tolist();a=info.arrays
    active=np.asarray(active_mask)
    if active.dtype!=bool or active.shape!=(len(names),):
        raise ValueError('A boolean full-objective active mask is required')
    if len(set(old))!=len(old) or not set(old)<=set(names):
        raise ValueError('Original parameter bindings are missing or duplicated')
    target=np.array([names.index(n) for n in old],dtype=np.int32)
    extra=np.array([i for i,n in enumerate(names) if n not in set(old)],dtype=np.int32)
    if np.any(active[extra]):
        raise ValueError('New active parameters cannot be filled from initial values')
    values=np.asarray(source['points'])
    if (values.dtype!=np.float64 or values.ndim!=2 or values.shape[1]!=len(old)
            or len(values)<1 or not np.isfinite(values).all()):
        raise ValueError('Original points must be a finite float64 matrix')
    if not np.array_equal(values[0],a['initial'][target]):
        raise ValueError('Original nominal coordinates differ from the profile import')
    for key,field in [('lower','lower'),('upper','upper')]:
        if not np.array_equal(source[key],a[field][target]):
            raise ValueError('Original physical bounds differ')
    if not np.array_equal(source['free_indices'],np.flatnonzero(source['lower']<source['upper'])):
        raise ValueError('Original free-parameter bindings differ')
    if np.any(values<a['lower'][target]) or np.any(values>a['upper'][target]):
        raise ValueError('Original points exceed bounds; never clip them')
    scales=np.where(a['error']>0,a['error'],1.)
    fields=dict(scales=scales.copy(),fd_steps=2e-4*scales)
    for key in fields:
        if (source[key].shape!=(len(old),) or not np.isfinite(source[key]).all()
                or np.any(source[key]<=0)):
            raise ValueError('Invalid original '+key)
        fields[key][target]=source[key]
    extended=np.broadcast_to(a['initial'],(len(values),len(names))).copy()
    extended[:,target]=values
    result=dict(parameter_names=np.asarray(names),points=extended,**fields,
        lower=a['lower'],upper=a['upper'],free_indices=np.flatnonzero(a['lower']<a['upper']).astype(np.int32))
    lineage=dict(mapping='Original columns copied bit-for-bit by name; only inactive new coordinates use imported initial values',
        original_to_profile_indices=target.tolist(),
        additional_inactive_parameters=[dict(name=names[i],initial=float(a['initial'][i])) for i in extra],
        resampled=False,clipped=False,other_discrete_choices_validated=False,source_native_values_reused=False)
    return result,lineage
