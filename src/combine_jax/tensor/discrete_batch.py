"""Dependency-aware candidate batching and ordered native masking replay.

These primitives do not select a minimizer or change the production discrete
driver. The importer must provide exact channel dependencies and native category
order. A candidate solver may evaluate only its enabled channels, but must read
no coordinates outside the declared read mask. Full-model integration and
native shared-parameter fixture validation are separate gates.
"""
import numpy as np


def orthogonal_offsets(sizes, native_order):
    """Framework-neutral v10 short combinations in registered category order."""
    sizes = np.asarray(sizes)
    order = np.asarray(native_order)
    if (sizes.ndim != 1 or sizes.dtype.kind not in 'iu' or np.any(sizes < 1)
            or order.shape != sizes.shape or order.dtype.kind not in 'iu'
            or sorted(order.tolist()) != list(range(len(sizes)))):
        raise ValueError('Positive category sizes and a complete native-order permutation required')
    offsets = [np.zeros(len(sizes), dtype=np.int32)]
    for category in order:
        for step in range(1, sizes[category]):
            item = np.zeros(len(sizes), dtype=np.int32)
            item[category] = step
            offsets.append(item)
    return np.stack(offsets)


def candidate_plan(anchor, offsets, sizes, native_order, alternative_parameters,
                   channel_categories, channel_parameters):
    """Build the device schedule for one pass, including retained trial state.

    ``channel_parameters`` contains all floating shape coordinates reachable
    from each channel (including its inactive alternatives). The native freeze
    rule enables the globally selected alternatives, intersected with the
    parameters of enabled channels. Only initially active coordinates reset.
    ``sources[i,p]`` is zero for the initial value, or the previous writer's
    candidate number plus one. Dependencies concern coordinates the fit reads;
    untouched coordinates are reconstructed during the ordered replay.
    """
    import jax
    import jax.numpy as jnp

    indices = (anchor[None, :] + offsets) % sizes[None, :]
    previous = jnp.concatenate((anchor[None, :], indices[:-1]), axis=0)
    changed = indices != previous
    rank = jnp.argsort(native_order)
    last_rank = jnp.max(jnp.where(changed, rank[None, :], -1), axis=1)
    last = native_order[jnp.maximum(last_rank, 0)]
    channels = (last_rank[:, None] < 0) | channel_categories[:, last].T
    globally_active = jnp.any(alternative_parameters[
        jnp.arange(len(anchor))[None, :], indices], axis=1)
    reads = (channels.astype(jnp.int32) @ channel_parameters.astype(jnp.int32)) > 0
    writes = globally_active & reads
    reset = globally_active[0]

    def source_step(last_writer, item):
        number, mask = item
        return jnp.where(mask & ~reset, number + 1, last_writer), last_writer

    numbers = jnp.arange(len(offsets), dtype=jnp.int32)
    _, sources = jax.lax.scan(source_step, jnp.zeros_like(reset, dtype=jnp.int32),
                              (numbers, writes))
    # A scatter avoids a candidates x candidates x parameters boolean tensor.
    dependencies = jnp.zeros((len(offsets), len(offsets) + 1), dtype=jnp.int32)
    dependencies = dependencies.at[numbers[:, None], sources].max(reads.astype(jnp.int32))
    dependencies = dependencies[:, 1:] > 0
    return dict(indices=indices, channels=channels, reads=reads, writes=writes,
                reset=reset, sources=sources, dependencies=dependencies)


def make_batched_candidates(solve_candidate, *, batch_size=8):
    """Return a JIT-compatible dependency-wave executor with bounded vmap width.

    ``solve_candidate(number, point, indices, active, channels, payload)`` returns
    a pytree containing at least ``values`` and ``status``. It must be safe for
    padded duplicate lanes: they are discarded and are not counted as trials.
    This function batches fit work only; acceptance always needs ordered replay.
    """
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    import jax
    import jax.numpy as jnp

    def run(base, plan, payload):
        count = len(plan['indices'])
        width = min(batch_size, count)
        # Infer result shapes without executing a fit or copying model arrays.
        sample = jax.eval_shape(solve_candidate, jnp.int32(0), base,
                                plan['indices'][0], plan['writes'][0],
                                plan['channels'][0], payload)
        results = jax.tree.map(lambda x: jnp.zeros((count,) + x.shape, x.dtype), sample)
        state = dict(results=results, done=jnp.zeros(count, dtype=bool),
                     waves=jnp.int32(0), wave=jnp.full(count, -1, dtype=jnp.int32),
                     source_values=jnp.zeros((count + 1, len(base)), base.dtype).at[0].set(base))

        def step(s):
            ready = ~s['done'] & ~jnp.any(plan['dependencies'] & ~s['done'][None, :], axis=1)
            chosen = jnp.nonzero(ready, size=width, fill_value=0)[0].astype(jnp.int32)
            valid = jnp.arange(width) < jnp.minimum(jnp.sum(ready), width)
            points = s['source_values'][plan['sources'][chosen], jnp.arange(len(base))[None, :]]
            points = jnp.where(plan['reads'][chosen], points, base[None, :])
            result = jax.vmap(solve_candidate, in_axes=(0, 0, 0, 0, 0, None))(
                chosen, points, plan['indices'][chosen], plan['writes'][chosen],
                plan['channels'][chosen], payload)

            # Scatter sequentially so padded copies of candidate zero cannot
            # overwrite its valid result or race a real result at that index.
            def store(s, lane):
                number = chosen[lane].astype(jnp.int32)
                def keep(s):
                    return dict(results=jax.tree.map(lambda a, b: a.at[number].set(b[lane]),
                                                      s['results'], result),
                                done=s['done'].at[number].set(True), waves=s['waves'],
                                wave=s['wave'].at[number].set(s['waves']),
                                source_values=s['source_values'].at[number + 1].set(result['values'][lane]))
                return jax.lax.cond(valid[lane], keep, lambda s: s, s), None

            s, _ = jax.lax.scan(store, s, jnp.arange(width, dtype=jnp.int32))
            return {**s, 'waves': s['waves'] + 1}

        state = jax.lax.while_loop(lambda s: ~jnp.all(s['done']) & (s['waves'] < count), step, state)
        return dict(results=state['results'], waves=state['waves'], wave=state['wave'],
                    complete=jnp.all(state['done']))
    return run


def replay_native_candidates(base, anchor, base_value, plan, results, channel_values, payload):
    """Replay strict acceptance and v10 mask offsets after batched candidate fits.

    ``channel_values`` includes penalties and returns all discrete-channel NLL
    terms; omitted terms must be constant in every shape coordinate. A shared
    coordinate may change a masked channel, so ranking the raw full NLL alone
    is invalid. The returned final offset survives restoring the best snapshot;
    it must also be included in the outer discrete convergence test. ``base_value``
    includes the offset at entry; ``offset_delta`` is the change during this pass.
    """
    import jax
    import jax.numpy as jnp

    baseline = jnp.sum(channel_values(base, anchor, payload))
    state = dict(point=base, best_point=base, best_indices=anchor,
                 best_value=base_value, offset_delta=jnp.zeros_like(base_value))

    def step(s, number):
        point = jnp.where((number > 0) & plan['reset'], base, s['point'])
        indices = plan['indices'][number]
        endpoint = jnp.where(plan['writes'][number], results['values'][number], point)
        before = channel_values(point, indices, payload)
        after = channel_values(endpoint, indices, payload)
        local_change = jnp.sum(jnp.where(plan['channels'][number], after - before, 0.))
        value = base_value + (jnp.sum(before) - baseline) + s['offset_delta'] + local_change
        offset = s['offset_delta'] - jnp.sum(jnp.where(plan['channels'][number], 0., after - before))
        accept = (results['status'][number] == 0) & jnp.isfinite(value) & (value < s['best_value'])
        state = dict(point=endpoint, best_point=jnp.where(accept, endpoint, s['best_point']),
                     best_indices=jnp.where(accept, indices, s['best_indices']),
                     best_value=jnp.where(accept, value, s['best_value']), offset_delta=offset)
        return state, dict(value=value, offset_delta=offset, accepted=accept,
                           start=point, end=endpoint)

    state, trace = jax.lax.scan(step, state, jnp.arange(len(plan['indices'])))
    return {**state, 'trace': trace}
