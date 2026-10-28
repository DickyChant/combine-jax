"""Padded uniform-mixture channels and a complete counting-model tensor objective."""
import math
import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle
from .scalar import validate_program


def pack_counting_channels(nodes,channels,scalars):
    """Channels are plain records with name/pdf/observables/weights/masked fields.

    Every supplied channel must lower successfully. Callers selecting a subset
    must keep it explicitly separate from a complete model acceptance claim.
    """
    validate_program(scalars)
    lookup = {n:i for i,n in enumerate(scalars.metadata['output_names'])}
    dynamic_slots = set(scalars.arrays['parameter_slots'].tolist())
    for batch in scalars.metadata['schedule']:
        dynamic_slots.update(scalars.arrays[batch['outputs']].tolist())
    rows = []
    for channel in channels:
        label = channel['name']
        if channel.get('masked',False):
            rows.append(dict(name=label,indices=[],weight=0.,shape=0.,enabled=False))
            continue
        cls,_,attrs = nodes[channel['pdf']]
        if cls != 'RooAddPdf' or not attrs['extended']:
            raise UnsupportedModelError(f'{label}: tensor counting channel requires an extended RooAddPdf')
        axes = tuple(channel['observables'])
        for pdf in attrs['pdfs']:
            if nodes[pdf][0] != 'RooUniform' or not set(axes) <= set(nodes[pdf][1]):
                raise UnsupportedModelError(f'{label}: {pdf} is not a uniform component over every channel observable')
        if len(attrs['pdfs']) != len(attrs['coefs']) or not attrs['pdfs']:
            raise UnsupportedModelError(f'{label}: invalid extended component list')
        volume = 1.
        for axis in axes:
            cls,_,a = nodes[axis]
            if cls != 'RooRealVar':
                raise UnsupportedModelError(f'{label}: observable must be a direct RooRealVar')
            width = a['maximum']-a['minimum']
            if not math.isfinite(width) or width <= 0:
                raise UnsupportedModelError(f'{label}: nonfinite/degenerate observable range')
            volume *= width
        weights = np.asarray(channel['weights'],dtype=np.float64)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)) or np.any(weights < 0):
            raise UnsupportedModelError(f'{label}: invalid data weights')
        missing = set(attrs['coefs'])-lookup.keys()
        if missing:
            raise UnsupportedModelError(f'{label}: missing scalar rate bindings {sorted(missing)}')
        weight = float(np.sum(weights))
        rows.append(dict(name=label,indices=[lookup[n] for n in attrs['coefs']],weight=weight,
                         shape=weight*math.log(volume),enabled=True))
    if len({r['name'] for r in rows}) != len(rows):
        raise ValueError('Duplicate channel name')
    fixed_values = np.zeros(len(rows),dtype=np.float64)
    dynamic_rows, dynamic_positions, folded = [], [], []
    output_slots = scalars.arrays['output_slots']
    for i,row in enumerate(rows):
        if not row['enabled']:
            folded.append(row['name'])
        elif all(int(output_slots[j]) not in dynamic_slots for j in row['indices']):
            expected = sum(scalars.arrays['initial_values'][output_slots[j]] for j in row['indices'])
            observed = row['weight']
            if expected <= 0:
                raise UnsupportedModelError(f"{row['name']}: invalid fixed expected count")
            fixed_values[i] = row['shape']+expected-observed-observed*(math.log(expected)-math.log(observed if observed > 0 else 1.))
            folded.append(row['name'])
        else:
            dynamic_rows.append(row)
            dynamic_positions.append(i)
    width = max((len(r['indices']) for r in dynamic_rows),default=0)
    indices = np.zeros((len(dynamic_rows),width),dtype=np.int32)
    mask = np.zeros(indices.shape,dtype=bool)
    for i,r in enumerate(dynamic_rows):
        indices[i,:len(r['indices'])] = r['indices']
        mask[i,:len(r['indices'])] = True
    arrays = dict(rate_indices=indices,rate_mask=mask,
        observed=np.array([r['weight'] for r in dynamic_rows],dtype=np.float64),
        shape_offset=np.array([r['shape'] for r in dynamic_rows],dtype=np.float64),
        dynamic_positions=np.array(dynamic_positions,dtype=np.int32),fixed_values=fixed_values)
    return TensorBundle(dict(schema='combine-jax.tensor.v1',kind='counting_channels',
        schedule=[dict(operation='extended_uniform_mixture')],names=[r['name'] for r in rows],
        scalar_names=scalars.metadata['output_names'],folded_channels=folded,
        scope='All supplied channels; observable volume and data weights retained, masks frozen at import.'),arrays)


def validate_counting(bundle):
    m,a = bundle.metadata,bundle.arrays
    if m['kind'] != 'counting_channels' or m['schedule'] != [dict(operation='extended_uniform_mixture')]:
        raise ValueError('Unknown counting channel program')
    total,n = len(m['names']),len(a['dynamic_positions'])
    if len(set(m['names'])) != total or a['rate_indices'].ndim != 2 or len(a['rate_indices']) != n:
        raise ValueError('Invalid counting channel layout')
    if a['rate_indices'].dtype != np.int32 or a['rate_mask'].dtype != bool or a['rate_mask'].shape != a['rate_indices'].shape:
        raise ValueError('Invalid rate gather/mask')
    if np.any(a['rate_indices'] < 0) or np.any(a['rate_indices'] >= len(m['scalar_names'])):
        raise ValueError('Rate index outside scalar vector')
    if a['fixed_values'].shape != (total,) or a['fixed_values'].dtype != np.float64:
        raise ValueError('Invalid fixed channel buffer')
    positions = a['dynamic_positions']
    if positions.ndim != 1 or positions.dtype != np.int32 or len(set(positions.tolist())) != n or np.any(positions < 0) or np.any(positions >= total):
        raise ValueError('Invalid dynamic channel output positions')
    for name in ('observed','shape_offset'):
        if a[name].shape != (n,) or a[name].dtype != np.float64:
            raise ValueError(f'Invalid counting buffer {name}')


def channel_values(scalar_values,arrays,xp=np):
    rates = scalar_values[arrays['rate_indices']]
    expected = xp.sum(xp.where(arrays['rate_mask'],rates,0.),axis=1)
    observed = arrays['observed']
    safe_observed = xp.where(observed > 0,observed,1.)
    nll = arrays['shape_offset']+expected-observed-observed*(xp.log(expected)-xp.log(safe_observed))
    return nll


def evaluate_counting(bundle,scalar_values):
    validate_counting(bundle)
    if np.shape(scalar_values) != (len(bundle.metadata['scalar_names']),):
        raise ValueError('Scalar vector shape mismatch')
    output = bundle.arrays['fixed_values'].copy()
    output[bundle.arrays['dynamic_positions']] = channel_values(np.asarray(scalar_values),bundle.arrays)
    return output


def make_counting_function(bundle):
    import jax.numpy as jnp
    validate_counting(bundle)
    size = len(bundle.metadata['scalar_names'])
    def evaluate(scalar_values,arrays):
        if scalar_values.shape != (size,):
            raise ValueError('Scalar vector shape mismatch')
        return arrays['fixed_values'].at[arrays['dynamic_positions']].set(channel_values(scalar_values,arrays,jnp))
    return evaluate


def make_counting_nll(scalars,constraints,channels):
    """One JAX objective with no host accumulation; all buffers remain arguments."""
    import jax.numpy as jnp
    from .constraints import make_constraint_function
    from .jax_scalar import make_function
    if constraints.metadata['profile_names']:
        raise UnsupportedModelError('A complete counting objective cannot omit a required profile solver')
    for bundle in (constraints,channels):
        if bundle.metadata['scalar_names'] != scalars.metadata['output_names']:
            raise ValueError('Component scalar bindings differ')
    scalar_function = make_function(scalars)
    constraint_function = make_constraint_function(constraints)
    counting_function = make_counting_function(channels)
    def nll(theta,arrays):
        values = scalar_function(theta,arrays['scalars'])
        penalties = constraint_function(values,jnp.empty((0,),dtype=jnp.float64),arrays['constraints'])
        channel_nll = counting_function(values,arrays['channels'])
        return jnp.sum(penalties)+jnp.sum(channel_nll)+arrays['offset']
    return nll
