"""Device-resident short-combination profiling of independent MultiPdf shapes."""
COUNTERS=('iteration','function_evaluations','gradient_evaluations','line_search_evaluations',
          'roundoff_accepts','hessian_evaluations')


def recenter(coordinates,point):
    free=coordinates['free_indices']
    return {**coordinates,'reference':point,
        'lower':(coordinates['physical_lower']-point[free])/coordinates['scale'],
        'upper':(coordinates['physical_upper']-point[free])/coordinates['scale']}


def condition_shape_scales(coordinates,arrays):
    """Avoid stale tiny nominal errors preventing any BFGS secant update.

    For discrete shape coordinates only, floor the optimization scale at 1% of
    a finite native interval (0.1*max(|nominal|,1) if unbounded). This changes
    neither the physical starting point nor the bounds, EDM threshold or NLL.
    Scales are not fitted uncertainties. Common parameters retain their policy.
    """
    import jax.numpy as jnp
    free=coordinates['free_indices'];point=coordinates['reference']
    width=coordinates['physical_upper']-coordinates['physical_lower']
    floor=jnp.where(jnp.isfinite(width),.01*width,.1*jnp.maximum(jnp.abs(point[free]),1.))
    scale=jnp.where(arrays['discrete_profile__parameter_union'][free],
        jnp.maximum(coordinates['scale'],floor),coordinates['scale'])
    return recenter({**coordinates,'scale':scale},point)


def make_discrete_fit_function(model,*,discrete_tolerance=.001,maximum_discrete_iterations=15,**fit_options):
    """Return fit(full_coordinates, short_coordinates, arrays), for one outer JIT.

    Match the native short-combination ordering and reset only the parameters
    active at the beginning of each discrete pass. The common-parameter fit and
    candidate fits retain EDM stopping; neither parameters nor choices leave the
    device. Nonconverged candidate fits are explicit failures, never silently
    omitted from the envelope. The caller must enforce the 8 GiB static guard.
    """
    import jax
    import jax.numpy as jnp
    from .discrete import active_parameters,make_discrete_channels_function
    from .model import make_model_function
    from .fitting import make_fit_function
    from .profiling import PROFILE_FIELDS
    if 'discrete_profile' not in model.metadata:raise ValueError('Model lacks discrete profile maps')
    if discrete_tolerance<=0 or maximum_discrete_iterations<1:raise ValueError('Invalid discrete convergence policy')
    if 'covariance' in fit_options:raise ValueError('Discrete envelope does not define a covariance across index switches')
    full=make_model_function(model,binned_gradient='structural_sequential')
    small=make_discrete_channels_function(model)
    def objective(theta,payload):
        point=jnp.where(payload['active'],theta,payload['reference'])
        return full(point,payload['arrays'])
    def short_objective(theta,payload):
        point=jnp.where(payload['active'],theta,payload['reference'])
        return jnp.sum(jnp.where(payload['channels'],small(point,payload['arrays']),0.))
    full_fit=make_fit_function(objective,covariance=False,**fit_options)
    short_fit=make_fit_function(short_objective,covariance=False,**fit_options)
    def continuous(coordinates,arrays,indices):
        active=active_parameters(indices,arrays)
        data={**arrays,'binned__discrete_indices':indices}
        result=full_fit(coordinates,dict(arrays=data,active=active,reference=coordinates['reference']))
        result['values']=jnp.where(active,result['values'],coordinates['reference'])
        return {name:result[name] for name in PROFILE_FIELDS}
    def fit(coordinates,short_coordinates,arrays):
        coordinates=condition_shape_scales(coordinates,arrays)
        short_coordinates=condition_shape_scales(short_coordinates,arrays)
        initial_indices=arrays['binned__discrete_indices']
        initial=continuous(coordinates,arrays,initial_indices)
        union=arrays['discrete_profile__parameter_union']
        table=arrays['discrete_profile__alternative_parameters']
        membership=arrays['discrete_profile__channel_categories']
        sizes=arrays['discrete_profile__category_sizes']
        offsets=arrays['discrete_profile__orthogonal_offsets']
        category_ids=jnp.arange(len(initial_indices))
        counts={name:initial[name] for name in COUNTERS}
        state=dict(result=initial,indices=initial_indices,delta=jnp.float64(jnp.inf),
            discrete_iteration=jnp.int32(0),candidate_failures=jnp.int32(0),counts=counts,
            continuous_fits=jnp.int32(1),short_fits=jnp.int32(0),
            candidate_status=jnp.zeros(len(offsets),dtype=jnp.int64),
            candidate_iterations=jnp.zeros(len(offsets),dtype=jnp.int32),
            candidate_edm=jnp.full(len(offsets),jnp.nan,dtype=jnp.float64),
            candidate_indices=jnp.zeros_like(offsets))
        def condition(s):
            return ((s['discrete_iteration']<maximum_discrete_iterations)&(s['delta']>=discrete_tolerance)
                &(s['result']['status']==0)&(s['candidate_failures']==0))
        def iteration(s):
            base=s['result']['values'];anchor=s['indices']
            reset=active_parameters(anchor,arrays)&union
            baseline=jnp.sum(small(base,{**arrays,'binned__discrete_indices':anchor}))
            start=dict(point=base,best_point=base,best_value=s['result']['value'],
                previous=anchor,best_indices=anchor,
                counts={name:value+(1 if name=='function_evaluations' else 0) for name,value in s['counts'].items()},
                failures=s['candidate_failures'])
            def candidate(carry,item):
                number,offset=item
                indices=(anchor+offset)%sizes
                point=jnp.where((number>0)&reset,base,carry['point'])
                changed=jnp.max(jnp.where(indices!=carry['previous'],category_ids,-1))
                channels=(changed<0)|membership[:,jnp.maximum(changed,0)]
                enabled=jnp.any(membership&channels[:,None],axis=0)
                active=jnp.any(table[category_ids,indices]&enabled[:,None],axis=0)&union
                data={**arrays,'binned__discrete_indices':indices}
                result=short_fit(recenter(short_coordinates,point),
                    dict(arrays=data,active=active,reference=point,channels=channels))
                point=jnp.where(active,result['values'],point)
                # Import rejects shape sharing outside these independent channels
                # and constrained shape parameters, so native masking offsets are
                # constant in every changed parameter. Full NLL ranks the candidates.
                value=s['result']['value']+jnp.sum(small(point,data))-baseline
                accept=(result['status']==0)&jnp.isfinite(value)&(value<carry['best_value'])
                return dict(point=point,best_point=jnp.where(accept,point,carry['best_point']),
                    best_value=jnp.where(accept,value,carry['best_value']),previous=indices,
                    best_indices=jnp.where(accept,indices,carry['best_indices']),
                    counts={name:carry['counts'][name]+result[name]+(1 if name=='function_evaluations' else 0) for name in COUNTERS},
                    failures=carry['failures']+((result['status']!=0)|~jnp.isfinite(value)).astype(jnp.int32)),dict(
                        status=result['status'],iterations=result['iteration'],edm=result['corrected_estimated_edm'],indices=indices)
            best,trace=jax.lax.scan(candidate,start,(jnp.arange(len(offsets)),offsets))
            result=continuous(recenter(coordinates,best['best_point']),arrays,best['best_indices'])
            return dict(result=result,indices=best['best_indices'],
                delta=jnp.abs(result['value']-s['result']['value']),
                discrete_iteration=s['discrete_iteration']+1,candidate_failures=best['failures'],
                counts={name:best['counts'][name]+result[name] for name in COUNTERS},
                continuous_fits=s['continuous_fits']+1,short_fits=s['short_fits']+len(offsets),
                candidate_status=trace['status'],candidate_iterations=trace['iterations'],
                candidate_edm=trace['edm'],candidate_indices=trace['indices'])
        state=jax.lax.while_loop(condition,iteration,state)
        result={**state['result'],**state['counts']}
        valid=(state['candidate_failures']==0)&(state['delta']<discrete_tolerance)
        result['status']=jnp.where((result['status']==0)&~valid,6,result['status'])
        result.update(discrete_indices=state['indices'],discrete_iteration=state['discrete_iteration'],
            discrete_delta=state['delta'],candidate_failures=state['candidate_failures'],
            continuous_fits=state['continuous_fits'],short_fits=state['short_fits'])
        result.update({name:state[name] for name in ('candidate_status','candidate_iterations','candidate_edm','candidate_indices')})
        return result
    return fit
