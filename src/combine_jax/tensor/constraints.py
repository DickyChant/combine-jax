"""Framework-neutral constraint stacks, with fixed normalizations folded at import."""
from collections import defaultdict
import math

import numpy as np
from scipy import special as scipy_special

from ..graph import UnsupportedModelError
from .bundle import TensorBundle
from .scalar import validate_program


def constraint_batch(kind, mode, observation_dynamic, values, buffers, xp=np, special=scipy_special):
    """Negative log values for a vector of one semantic constraint type."""
    if kind == 'fixed':
        return buffers['fixed_values']
    x, mean = values[:, 0], values[:, 1]
    if kind == 'gaussian':
        sigma = values[:, 2]
        result = .5*((x-mean)/sigma)**2
    else:
        if observation_dynamic:
            if kind == 'poisson':
                x = xp.where(buffers['no_rounding'], x, xp.floor(x))
            factorial = special.gammaln(x+1)
        else:
            x, factorial = buffers['observed'], buffers['log_factorial']
        if kind == 'poisson':
            logvalue = special.xlogy(x, xp.where(x == 0, 1., mean))-mean-factorial
            result = -xp.where(x < 0, -xp.inf, xp.where(mean < 0, xp.nan, logvalue))
        elif kind == 'fast_poisson':
            safe_mean = xp.where(mean > 0, mean, 1.)
            exact = special.xlogy(x, xp.where(xp.abs(x) < 1e-10, 1., mean))-mean-factorial
            large = .5*xp.log(safe_mean)-.5*(x-mean)**2/safe_mean
            empty = xp.where(xp.abs(mean) < 1e-10, 0., -mean)
            result = -xp.where(xp.abs(x) < 1e-10, empty, xp.where(x < 1e6, exact, large))
        else:
            raise ValueError(f'Unknown constraint kernel {kind}')
    if mode == 'none':
        return result
    if mode == 'fixed':
        return result+buffers['log_normalization']
    integral = normalization_integral(mode,x,mean,values[:,2] if kind == 'gaussian' else None,buffers,xp,special)
    return result+xp.log(integral)


def normalization_integral(mode,x,mean,sigma,buffers,xp=np,special=scipy_special):
    low, high = buffers['bounds'][:, 0], buffers['bounds'][:, 1]
    if mode == 'gaussian':
        center = xp.where(buffers['axis_is_first'], mean, x)
        sigma = xp.abs(sigma)
        integral = xp.sqrt(xp.pi/2)*sigma*(special.erf((high-center)/(xp.sqrt(2.)*sigma))-
                                         special.erf((low-center)/(xp.sqrt(2.)*sigma)))
    elif mode == 'poisson_mean':
        integral = special.gammainc(x+1, high)-special.gammainc(x+1, xp.maximum(0.,low))
    elif mode == 'poisson_observed':
        imin, imax = buffers['integer_bounds'][:, 0], buffers['integer_bounds'][:, 1]
        # Match RooPoisson's integer-range and tail-selection conventions.
        safe_imin = xp.where(imin == 0, 1., imin)
        lower_tail = special.gammaincc(imax,mean)-special.gammaincc(safe_imin,mean)
        upper_tail = special.gammainc(safe_imin,mean)-special.gammainc(imax,mean)
        integral = xp.where(imin == 0, special.gammaincc(imax,mean),
                            xp.where(imin <= mean, lower_tail, upper_tail))
        integral = xp.where(high < xp.maximum(0.,low), 0., integral)
    else:
        raise ValueError(f'Unknown constraint normalization {mode}')
    return integral


def pack_constraints(nodes, names, axes, scalars, *, profile_names=(), fixed_normalization=None):
    """Deduplicate constraints and gather scalar/profile arguments by global identity.

    fixed_normalization(name, axes) returns a native raw integral, and is called
    only when the integral is independent of floating/profiled coordinates.
    Without a native reader the same analytic formula is evaluated with SciPy.
    """
    validate_program(scalars)
    names = tuple(dict.fromkeys(names))
    profile_names = tuple(profile_names)
    scalar_names = scalars.metadata['output_names']
    if len(set(profile_names)) != len(profile_names) or set(profile_names) & set(scalar_names):
        raise ValueError('Ambiguous profile bindings')
    lookup = {n:i for i,n in enumerate([*scalar_names,*profile_names])}
    produced = set(scalars.arrays['parameter_slots'].tolist())
    for batch in scalars.metadata['schedule']:
        produced.update(scalars.arrays[batch['outputs']].tolist())
    dynamic = {n for n,s in zip(scalar_names,scalars.arrays['output_slots'],strict=True) if int(s) in produced}
    dynamic.update(profile_names)
    constants = {n:float(scalars.arrays['initial_values'][s])
                 for n,s in zip(scalar_names,scalars.arrays['output_slots'],strict=True) if n not in dynamic}
    def depends(name, variable):
        return name == variable or any(depends(c,variable) for c in nodes[name][1])
    groups = defaultdict(list)
    folded_normalizations, folded_constraints = [], []
    for position,name in enumerate(names):
        cls, inputs, attrs = nodes[name]
        if cls not in ('SimpleGaussianConstraint','RooGaussian','RooPoisson','SimplePoissonConstraint'):
            raise UnsupportedModelError(f'{name} ({cls}): tensor constraint type not supported')
        kind = ('gaussian' if cls in ('SimpleGaussianConstraint','RooGaussian') else
                'poisson' if cls == 'RooPoisson' else 'fast_poisson')
        if len(inputs) != (3 if kind == 'gaussian' else 2):
            raise ValueError(f'{name}: invalid constraint arity')
        missing = set(inputs)-lookup.keys()
        if missing:
            raise UnsupportedModelError(f'{name}: scalar/profile bindings missing for {sorted(missing)}')
        norm_axes = tuple(axes.get(name,()))
        mode, bounds, first = 'none', (0.,1.), False
        if norm_axes:
            if cls.startswith('Simple'):
                raise UnsupportedModelError(f'{name}: fast constraint must have no normalization axes')
            if len(norm_axes) != 1 or norm_axes[0] not in inputs[:2]:
                raise UnsupportedModelError(f'{name}: unsupported constraint normalization axes {norm_axes}')
            variable = norm_axes[0]
            if nodes[variable][0] != 'RooRealVar':
                raise UnsupportedModelError(f'{name}: integral axis must be a direct RooRealVar')
            first = variable == inputs[0]
            other = inputs[1] if first else inputs[0]
            retained = [other, *([inputs[2]] if kind == 'gaussian' else [])]
            if any(depends(n,variable) for n in retained):
                raise UnsupportedModelError(f'{name}: arguments depend on integration variable')
            bounds = nodes[variable][2]['minimum'], nodes[variable][2]['maximum']
            mode = 'gaussian' if kind == 'gaussian' else 'poisson_observed' if first else 'poisson_mean'
            fixed_norm = not any(n in dynamic for n in retained)
            if mode == 'poisson_mean' and not fixed_norm:
                raise UnsupportedModelError(f'{name}: dynamic observed-value derivative of incomplete gamma requires a later operator')
        else:
            fixed_norm = False
        observation_dynamic = inputs[0] in dynamic if kind != 'gaussian' else False
        observed = constants.get(inputs[0],0.)
        no_rounding = bool(attrs.get('no_rounding',True))
        if kind == 'poisson' and not no_rounding:
            observed = float(np.floor(observed))
        lo, hi = max(0.,bounds[0]), bounds[1]
        integer_bounds = (int(lo) if math.isfinite(lo) else 2**32-1,
                          min(int(hi+1) if math.isfinite(hi) else 2**32-1, 2**32-1))
        record = dict(name=name, position=position, indices=[lookup[n] for n in inputs],
            no_rounding=no_rounding, observed=observed,
            log_factorial=float(scipy_special.gammaln(observed+1)) if kind != 'gaussian' else 0.,
            bounds=bounds, integer_bounds=integer_bounds, axis_is_first=first,
            log_normalization=0., fixed_values=0.)
        temporary = {k:np.asarray([v]) for k,v in record.items() if k not in ('name','position','indices')}
        if fixed_norm:
            if fixed_normalization is not None:
                integral = float(fixed_normalization(name,norm_axes))
                log_integral = float(np.log(integral)) if integral > 0 else float('nan')
            else:
                # The integrated argument is irrelevant; choose a valid dummy.
                vals = np.array([[constants.get(n,1.) for n in inputs]])
                x = vals[:,0]
                if kind == 'poisson' and not no_rounding:
                    x = np.floor(x)
                integral = normalization_integral(mode,x,vals[:,1],vals[:,2] if kind == 'gaussian' else None,temporary)
                log_integral = float(np.log(integral)[0])
            if not math.isfinite(log_integral):
                raise UnsupportedModelError(f'{name}: nonfinite fixed normalization')
            record['log_normalization'] = log_integral
            mode = 'fixed'
            folded_normalizations.append(name)
        if not any(n in dynamic for n in inputs):
            vals = np.array([[constants[n] for n in inputs]])
            temporary = {k:np.asarray([v]) for k,v in record.items() if k not in ('name','position','indices')}
            value = float(constraint_batch(kind,mode,False,vals,temporary)[0])
            if not math.isfinite(value):
                raise UnsupportedModelError(f'{name}: nonfinite fixed constraint')
            record['fixed_values'] = value
            kind,mode,observation_dynamic = 'fixed','none',False
            folded_constraints.append(name)
        groups[(kind,mode,observation_dynamic)].append(record)
    arrays = {'output_template':np.zeros(len(names),dtype=np.float64)}
    schedule = []
    for i,((kind,mode,observation_dynamic),records) in enumerate(sorted(groups.items())):
        buffers = {}
        # Fixed groups may have different original arities; their inputs are unused.
        if kind != 'fixed':
            arrays[f'input_{i}'] = np.asarray([r['indices'] for r in records],dtype=np.int32)
        arrays[f'output_{i}'] = np.asarray([r['position'] for r in records],dtype=np.int32)
        for key in ('no_rounding','observed','log_factorial','bounds','integer_bounds',
                    'axis_is_first','log_normalization','fixed_values'):
            array = np.asarray([r[key] for r in records],dtype=bool if key in ('no_rounding','axis_is_first') else np.float64)
            arrays[f'{key}_{i}'] = array
            buffers[key] = f'{key}_{i}'
        schedule.append(dict(operation=kind,normalization=mode,observation_dynamic=observation_dynamic,
            inputs=None if kind == 'fixed' else f'input_{i}', outputs=f'output_{i}', buffers=buffers))
    return TensorBundle(dict(schema='combine-jax.tensor.v1',kind='constraint_program',schedule=schedule,
        names=list(names),scalar_names=list(scalar_names),profile_names=list(profile_names),
        folded_normalizations=folded_normalizations,folded_constraints=folded_constraints,
        normalization_source='native integral reader' if fixed_normalization else 'analytic SciPy evaluation',
        scope='All requested constraints exactly once; profile values must come from the same model evaluation.'),arrays)


def validate_constraints(bundle):
    m,a = bundle.metadata,bundle.arrays
    if m['kind'] != 'constraint_program' or len(set(m['names'])) != len(m['names']):
        raise ValueError('Invalid constraint program identity')
    if a['output_template'].shape != (len(m['names']),) or a['output_template'].dtype != np.float64:
        raise ValueError('Invalid constraint output buffer')
    count = len(m['scalar_names'])+len(m['profile_names'])
    seen = []
    for batch in m['schedule']:
        kind,mode = batch['operation'],batch['normalization']
        allowed = {'fixed':{'none'},'gaussian':{'none','fixed','gaussian'},
                   'poisson':{'none','fixed','poisson_observed','poisson_mean'},'fast_poisson':{'none'}}
        if kind not in allowed or mode not in allowed[kind]:
            raise ValueError('Unknown constraint operation or normalization')
        out = a[batch['outputs']]
        if out.dtype != np.int32 or out.ndim != 1:
            raise ValueError('Invalid constraint output indices')
        seen.extend(out.tolist())
        if kind != 'fixed':
            indices = a[batch['inputs']]
            if indices.dtype != np.int32 or indices.shape != (len(out),3 if kind == 'gaussian' else 2):
                raise ValueError('Invalid constraint inputs')
            if np.any(indices < 0) or np.any(indices >= count):
                raise ValueError('Constraint input index outside scalar/profile vector')
        for key,name in batch['buffers'].items():
            shape = (len(out),2) if key in ('bounds','integer_bounds') else (len(out),)
            dtype = bool if key in ('no_rounding','axis_is_first') else np.float64
            if a[name].shape != shape or a[name].dtype != dtype:
                raise ValueError(f'Invalid constraint buffer {key}')
    if sorted(seen) != list(range(len(m['names']))):
        raise ValueError('Constraints must be covered exactly once')


def evaluate_constraints(bundle, scalar_values, profile_values=()):
    validate_constraints(bundle)
    if np.shape(scalar_values) != (len(bundle.metadata['scalar_names']),) or np.shape(profile_values) != (len(bundle.metadata['profile_names']),):
        raise ValueError('Constraint scalar/profile input shape mismatch')
    combined = np.concatenate((scalar_values,profile_values))
    output = np.zeros_like(bundle.arrays['output_template'])
    for batch in bundle.metadata['schedule']:
        values = None if batch['inputs'] is None else combined[bundle.arrays[batch['inputs']]]
        buffers = {k:bundle.arrays[v] for k,v in batch['buffers'].items()}
        output[bundle.arrays[batch['outputs']]] = constraint_batch(batch['operation'],batch['normalization'],
            batch['observation_dynamic'],values,buffers)
    return output


def make_constraint_function(bundle):
    import jax.numpy as jnp
    import jax.scipy.special as special
    validate_constraints(bundle)
    schedule = tuple(dict(b) for b in bundle.metadata['schedule'])
    sizes = len(bundle.metadata['scalar_names']),len(bundle.metadata['profile_names'])
    def evaluate(scalar_values,profile_values,arrays):
        if scalar_values.shape != (sizes[0],) or profile_values.shape != (sizes[1],):
            raise ValueError('Constraint scalar/profile input shape mismatch')
        combined = jnp.concatenate((scalar_values,profile_values))
        output = jnp.zeros_like(arrays['output_template'])
        for batch in schedule:
            values = None if batch['inputs'] is None else combined[arrays[batch['inputs']]]
            buffers = {k:arrays[v] for k,v in batch['buffers'].items()}
            result = constraint_batch(batch['operation'],batch['normalization'],batch['observation_dynamic'],values,buffers,jnp,special)
            output = output.at[arrays[batch['outputs']]].set(result)
        return output
    return evaluate
