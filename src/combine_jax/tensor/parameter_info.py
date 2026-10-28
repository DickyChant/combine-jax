"""Framework-neutral physical parameter bounds and native step/error metadata."""
import numpy as np

from .bundle import TensorBundle


COORDINATE_FIELDS = ('reference','free_indices','scale','lower','upper','physical_lower','physical_upper')


def pack_parameter_info(parameters):
    """Preserve the extractor's ordered parameters without inventing fit errors."""
    parameters=tuple(parameters)
    bundle=TensorBundle(dict(schema='combine-jax.tensor.v1',kind='parameter_info',schedule=[],
        names=[p.name for p in parameters],roles=[p.role for p in parameters]),
        dict(initial=np.array([p.value for p in parameters],dtype=np.float64),
             lower=np.array([p.minimum for p in parameters],dtype=np.float64),
             upper=np.array([p.maximum for p in parameters],dtype=np.float64),
             error=np.array([p.error for p in parameters],dtype=np.float64)))
    validate_parameter_info(bundle)
    return bundle


def validate_parameter_info(bundle):
    m,a=bundle.metadata,bundle.arrays
    if m['kind']!='parameter_info' or m['schedule']!=[]:raise ValueError('Expected parameter information')
    names=m['names'];n=len(names)
    if len(set(names))!=n or any(not isinstance(s,str) or not s for s in names):raise ValueError('Invalid parameter names')
    if len(m['roles'])!=n or any(not isinstance(s,str) or not s for s in m['roles']):raise ValueError('Invalid parameter roles')
    expected={'initial','lower','upper','error'}
    policy=m.get('fit_policy')
    if policy is not None:
        from .fit_policy import POLICY
        if policy.get('policy')!=POLICY or policy.get('runtime_flags')!={'OPTIMIZE_BOUNDS':1,'UNBOUND_GAUSSIANS':0}:
            raise ValueError('Unsupported fit policy')
        expected|={'stored_lower','stored_upper'}
    if set(a)!=expected:raise ValueError('Invalid parameter fields')
    for name,value in a.items():
        if value.dtype!=np.float64 or value.shape!=(n,) or np.any(np.isnan(value)):
            raise ValueError(f'Invalid parameter field {name}')
    if not np.isfinite(a['initial']).all() or not np.isfinite(a['error']).all() or np.any(a['error']<0):
        raise ValueError('Invalid parameter initial values/errors')
    if np.any(a['lower']>a['initial']) or np.any(a['upper']<a['initial']):raise ValueError('Initial parameter outside bounds')
    if policy is not None:
        if np.any(a['stored_lower']>a['initial']) or np.any(a['stored_upper']<a['initial']):
            raise ValueError('Initial parameter outside stored bounds')
        optimized=set(policy['nuisance_bounds_removed'])
        fixed=np.array([name in policy['fixed_real_values'] for name in names],dtype=bool)
        lower=a['stored_lower'].copy();upper=a['stored_upper'].copy()
        for i,name in enumerate(names):
            if name in optimized and not fixed[i]:lower[i],upper[i]=-np.inf,np.inf
        if not np.array_equal(lower,a['lower']) or not np.array_equal(upper,a['upper']):
            raise ValueError('Effective bounds disagree with native policy')


def fit_coordinates(info,*,initial=None,fixed=None):
    """Validate physical starts/fixes and construct scaled free-coordinate arrays.

    Native errors are retained separately. A nonpositive error uses 0.1 times
    max(abs(initial),1) as an explicit optimization scale, not an uncertainty.
    """
    validate_parameter_info(info);a=info.arrays;names=info.metadata['names'];n=len(names)
    point=np.array(a['initial'] if initial is None else initial,dtype=np.float64,copy=True)
    if point.shape!=(n,) or not np.isfinite(point).all():raise ValueError('Invalid fit initial vector')
    fixed=dict(fixed or {})
    if fixed.keys()-set(names):raise ValueError(f'Unknown fixed parameters: {sorted(fixed.keys()-set(names))}')
    immutable=a['lower']==a['upper']
    for i in np.flatnonzero(immutable):
        if point[i]!=a['initial'][i] or (names[i] in fixed and fixed[names[i]]!=a['initial'][i]):
            raise ValueError(f'Cannot change imported fixed parameter {names[i]}')
        fixed[names[i]]=float(a['initial'][i])
    for name,value in fixed.items():point[names.index(name)]=value
    if not np.isfinite(point).all() or np.any(point<a['lower']) or np.any(point>a['upper']):raise ValueError('Fit initial/fixed value outside bounds')
    free=np.array([i for i,name in enumerate(names) if name not in fixed],dtype=np.int32)
    raw_scale=np.where(a['error']>0,a['error'],.1*np.maximum(np.abs(a['initial']),1.))
    scale=raw_scale[free]
    return dict(reference=point,free_indices=free,scale=scale,lower=(a['lower'][free]-point[free])/scale,
        upper=(a['upper'][free]-point[free])/scale,physical_lower=a['lower'][free],
        physical_upper=a['upper'][free],fixed=fixed,
        scale_policy='Native positive error, otherwise 0.1*max(abs(imported initial),1); scaling is not a fitted uncertainty.')


def attach_parameter_info(model,info):
    """Add neutral fit metadata while preserving the likelihood's numeric fields."""
    validate_parameter_info(info)
    if model.metadata['parameter_names']!=info.metadata['names']:
        raise ValueError('Model and fit parameter order differ')
    if not np.array_equal(model.arrays['scalars__parameter_initial'],info.arrays['initial']):
        raise ValueError('Model and fit initial values differ')
    if not np.array_equal(model.arrays['scalars__parameter_fixed'],info.arrays['lower']==info.arrays['upper']):
        raise ValueError('Model and fit fixed parameters differ')
    if 'parameter_info' in model.metadata or any(n.startswith('parameter_info__') for n in model.arrays):
        raise ValueError('Model already has parameter information')
    return TensorBundle._from_frozen({**model.metadata,'parameter_info':info.metadata},
        {**model.arrays,**{'parameter_info__'+n:a for n,a in info.arrays.items()}})


def model_parameter_info(model):
    if 'parameter_info' not in model.metadata:raise ValueError('Model lacks parameter information')
    info=TensorBundle(model.metadata['parameter_info'],
        {n[len('parameter_info__'):]:a for n,a in model.arrays.items() if n.startswith('parameter_info__')})
    validate_parameter_info(info)
    if model.metadata['parameter_names']!=info.metadata['names'] or not np.array_equal(
        model.arrays['scalars__parameter_initial'],info.arrays['initial']) or not np.array_equal(
        model.arrays['scalars__parameter_fixed'],info.arrays['lower']==info.arrays['upper']):
        raise ValueError('Model and parameter information disagree')
    return info
