"""Device-resident native-order discrete fits with shared-shape mask offsets.

The ordered replay and dependency-aware batching are independent of the
continuous optimizer. Raw physical NLL and accumulated native mask offsets
are reported separately. Reaching the outer cap is never called convergence.
The production importer still requires native shared-fixture validation.
"""


def make_batched_discrete_fit_function(model, *, _full_function=None, _channel_function=None,
                                      candidate_batch_size=8, discrete_tolerance=.001,
                                      maximum_discrete_iterations=15, **fit_options):
    import jax
    import jax.numpy as jnp
    from .discrete import active_parameters
    from .discrete_batch import candidate_plan, make_batched_candidates, replay_native_candidates
    from .discrete_fit import COUNTERS, condition_shape_scales, recenter
    from .fitting import make_fit_function
    from .profiling import PROFILE_FIELDS
    if 'discrete_profile' not in model.metadata:
        raise ValueError('Explicit native discrete maps required')
    required=('native_order','channel_parameters')
    if any('discrete_profile__'+name not in model.arrays for name in required):
        raise ValueError('Native category order and complete channel dependencies required')
    if discrete_tolerance<=0 or maximum_discrete_iterations<1 or 'covariance' in fit_options:
        raise ValueError('Invalid discrete stopping/covariance policy')
    from .model import make_model_function
    from .discrete import make_discrete_channels_function
    full=(_full_function if _full_function is not None else
          make_model_function(model,binned_gradient='structural_sequential'))
    channels=(_channel_function if _channel_function is not None else make_discrete_channels_function(model))

    def full_objective(theta,payload):
        return full(jnp.where(payload['active'],theta,payload['reference']),payload['arrays'])

    def short_objective(theta,payload):
        point=jnp.where(payload['active'],theta,payload['reference'])
        return jnp.sum(jnp.where(payload['channels'],channels(point,payload['arrays']),0.))

    full_fit=make_fit_function(full_objective,covariance=False,**fit_options)
    short_fit=make_fit_function(short_objective,covariance=False,**fit_options)

    def continuous(coordinates,arrays,indices):
        active=active_parameters(indices,arrays)
        result=full_fit(coordinates,dict(arrays={**arrays,'binned__discrete_indices':indices},
                       active=active,reference=coordinates['reference']))
        result['values']=jnp.where(active,result['values'],coordinates['reference'])
        return {name:result[name] for name in (*PROFILE_FIELDS,'inverse_uncertainty')}

    def trial(number,point,indices,active,enabled,payload):
        result=short_fit(recenter(payload['short_coordinates'],point),dict(
            arrays={**payload['arrays'],'binned__discrete_indices':indices},
            active=active,reference=point,channels=enabled))
        result['values']=jnp.where(active,result['values'],point)
        return {name:result[name] for name in ('values','status','corrected_estimated_edm',*COUNTERS)}

    batch=make_batched_candidates(trial,batch_size=candidate_batch_size)

    def channel_values(point,indices,arrays):
        return channels(point,{**arrays,'binned__discrete_indices':indices})

    def fit(coordinates,short_coordinates,arrays):
        coordinates=condition_shape_scales(coordinates,arrays)
        short_coordinates=condition_shape_scales(short_coordinates,arrays)
        indices=arrays['binned__discrete_indices']
        initial=continuous(coordinates,arrays,indices)
        count=len(arrays['discrete_profile__orthogonal_offsets'])
        state=dict(result=initial,indices=indices,offset=jnp.float64(0),delta=jnp.float64(jnp.inf),
            iteration=jnp.int32(0),failures=jnp.int32(0),waves=jnp.int32(0),
            counts={name:initial[name] for name in COUNTERS},
            native_values=jnp.full(maximum_discrete_iterations,jnp.nan),
            raw_values=jnp.full(maximum_discrete_iterations,jnp.nan),
            offsets=jnp.full(maximum_discrete_iterations,jnp.nan),
            index_history=jnp.zeros((maximum_discrete_iterations,len(indices)),dtype=indices.dtype),
            candidate_status=jnp.zeros(count,dtype=initial['status'].dtype))

        def condition(s):
            return ((s['iteration']<maximum_discrete_iterations)&(s['delta']>=discrete_tolerance)
                    &(s['result']['status']==0)&(s['failures']==0))

        def step(s):
            base=s['result']['values'];native_before=s['result']['value']+s['offset']
            plan=candidate_plan(s['indices'],arrays['discrete_profile__orthogonal_offsets'],
                arrays['discrete_profile__category_sizes'],arrays['discrete_profile__native_order'],
                arrays['discrete_profile__alternative_parameters'],arrays['discrete_profile__channel_categories'],
                arrays['discrete_profile__channel_parameters'])
            trials=batch(base,plan,dict(arrays=arrays,short_coordinates=short_coordinates))
            replay=replay_native_candidates(base,s['indices'],native_before,plan,trials['results'],channel_values,arrays)
            result=continuous(recenter(coordinates,replay['best_point']),arrays,replay['best_indices'])
            offset=s['offset']+replay['offset_delta'];native_after=result['value']+offset
            failures=jnp.count_nonzero(trials['results']['status']).astype(jnp.int32)
            failures+=((~trials['complete'])|~jnp.isfinite(native_after)).astype(jnp.int32)
            return dict(result=result,indices=replay['best_indices'],offset=offset,
                delta=jnp.abs(native_after-native_before),iteration=s['iteration']+1,
                failures=s['failures']+failures,waves=s['waves']+trials['waves'],
                counts={name:s['counts'][name]+jnp.sum(trials['results'][name],dtype=s['counts'][name].dtype)+result[name]
                        for name in COUNTERS},
                native_values=s['native_values'].at[s['iteration']].set(native_after),
                raw_values=s['raw_values'].at[s['iteration']].set(result['value']),
                offsets=s['offsets'].at[s['iteration']].set(offset),
                index_history=s['index_history'].at[s['iteration']].set(replay['best_indices']),
                candidate_status=trials['results']['status'])

        state=jax.lax.while_loop(condition,step,state)
        result={**state['result'],**state['counts']}
        valid=(state['failures']==0)&(state['delta']<discrete_tolerance)
        result['status']=jnp.where((result['status']==0)&~valid,6,result['status'])
        result['converged']=result['status']==0
        result.update(discrete_indices=state['indices'],discrete_iteration=state['iteration'],
            discrete_delta=state['delta'],candidate_failures=state['failures'],candidate_waves=state['waves'],
            candidate_status=state['candidate_status'],masking_offset=state['offset'],
            native_rank_value=result['value']+state['offset'],raw_value=result['value'],
            discrete_raw_values=state['raw_values'],discrete_native_values=state['native_values'],
            discrete_masking_offsets=state['offsets'],discrete_index_history=state['index_history'],
            continuous_fits=state['iteration']+1,short_fits=state['iteration']*count,
            ranking_channel_evaluations=state['iteration']*(2*count+1))
        return result
    return fit
