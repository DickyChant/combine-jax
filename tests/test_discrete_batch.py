"""Small stateful fixtures for batching; not a native minimizer agreement claim."""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.discrete_batch import (
    candidate_plan, make_batched_candidates, orthogonal_offsets, replay_native_candidates,
)


def fixture(anchor):
    # Categories 0 and 1 share an initially inactive coordinate in their second
    # alternatives. Category 2 is independent. Native order is deliberately
    # different from the model's category storage order.
    table = np.zeros((3, 2, 5), dtype=bool)
    table[0, 0, 0] = table[1, 0, 1] = True
    table[0, 1, 2] = table[1, 1, 2] = True
    table[2, 0, 3] = table[2, 1, 4] = True
    order = np.array([1, 0, 2], dtype=np.int32)
    sizes = np.array([2, 2, 2], dtype=np.int32)
    return candidate_plan(jnp.array(anchor), jnp.array(orthogonal_offsets(sizes, order)),
                          jnp.array(sizes), jnp.array(order), jnp.array(table),
                          jnp.eye(3, dtype=bool), jnp.array(table.any(axis=1)))


def terms(point, indices, unused):
    a, b, shared, c, d = point
    return jnp.array([jnp.where(indices[0] == 0, (a - .2)**2, (shared - 1.)**2 - .7),
                      jnp.where(indices[1] == 0, (b + .3)**2, (shared + 2.)**2 - .4),
                      jnp.where(indices[2] == 0, c**2, (d - .5)**2 - .2)])


def trial(number, point, indices, active, channels, unused):
    # Deliberately retain some starting-point dependence. This exercises the
    # ordering contract independently of any numerical optimizer's convergence.
    gradient = jax.grad(lambda x: jnp.sum(jnp.where(channels, terms(x, indices, None), 0.)))(point)
    return dict(values=jnp.where(active, point - .2 * gradient, point),
                status=jnp.int32(0), number=number)


def literal_reference(base, anchor, plan):
    """Python reproduction of reset + enable/disable-mask calls, without batching."""
    plan = jax.device_get(plan)
    point = np.array(base)
    best_point = point.copy()
    best_indices = np.array(anchor)
    offset = 0.
    mask = np.ones(3, dtype=bool)
    best = float(jnp.sum(terms(point, anchor, None)))
    trace = []

    def evaluate():
        return np.sum(np.asarray(terms(point, indices, None))[mask]) + offset

    for number, indices in enumerate(plan['indices']):
        if number:
            point[plan['reset']] = np.asarray(base)[plan['reset']]
        start = point.copy()
        before = evaluate()
        mask = plan['channels'][number]
        offset += before - evaluate()  # native setMaskNonDiscreteChannels(true)
        fitted = trial(number, jnp.asarray(point), indices, plan['writes'][number], mask, None)
        point = np.asarray(fitted['values']).copy()
        before = evaluate()
        mask = np.ones(3, dtype=bool)
        offset += before - evaluate()  # native setMaskNonDiscreteChannels(false)
        value = evaluate()
        accept = value < best
        if accept:
            best, best_point, best_indices = value, point.copy(), indices.copy()
        trace.append(dict(value=value, offset_delta=offset, accepted=accept, start=start, end=point.copy()))
    return best, best_point, best_indices, offset, trace


@pytest.mark.parametrize('anchor', list(itertools.product(range(2), repeat=3)))
@pytest.mark.parametrize('batch_size', [1, 2, 8])
def test_batched_trials_replay_native_resets_and_offsets(anchor, batch_size):
    base = jnp.array([.9, -.7, 1.3, -.4, 1.1], dtype=jnp.float64)
    anchor = jnp.array(anchor, dtype=jnp.int32)
    plan = fixture(anchor)
    run = jax.jit(make_batched_candidates(trial, batch_size=batch_size))
    result = run(base, plan, None)
    assert result['complete']
    assert int(result['waves']) <= len(plan['indices'])
    replay = jax.jit(lambda base, anchor, plan, result: replay_native_candidates(
        base, anchor, jnp.sum(terms(base, anchor, None)), plan, result, terms, None))(
            base, anchor, plan, result['results'])
    best, point, indices, offset, trace = literal_reference(base, anchor, plan)
    np.testing.assert_allclose(replay['best_value'], best, atol=2e-14, rtol=0)
    np.testing.assert_allclose(replay['best_point'], point, atol=2e-14, rtol=0)
    np.testing.assert_array_equal(replay['best_indices'], indices)
    np.testing.assert_allclose(replay['offset_delta'], offset, atol=2e-14, rtol=0)
    for key in trace[0]:
        np.testing.assert_allclose(replay['trace'][key], [r[key] for r in trace], atol=2e-14, rtol=0)


def test_initially_inactive_shared_parameter_has_dependency():
    plan = fixture([0, 0, 0])
    # Trial 1 writes shared in category 1, trial 2 reads it in category 0.
    assert plan['dependencies'][2, 1]
    assert plan['sources'][2, 2] == 2
    assert not plan['dependencies'][3].any()
    # If shared is active initially, it resets, removing that state dependency.
    reset_plan = fixture([0, 1, 0])
    assert reset_plan['reset'][2]
    assert not reset_plan['dependencies'].any()


def test_masks_matter_and_strict_ties_keep_first():
    base = jnp.array([.9, -.7, 1.3, -.4, 1.1], dtype=jnp.float64)
    anchor = jnp.array([0, 1, 0])
    plan = fixture(anchor)
    fitted = make_batched_candidates(trial)(base, plan, None)
    result = replay_native_candidates(base, anchor, jnp.sum(terms(base, anchor, None)),
                                      plan, fitted['results'], terms, None)
    assert abs(float(result['offset_delta'])) > .1  # full-NLL-only replay would fail
    fixed = dict(values=jnp.broadcast_to(base, (4, 5)), status=jnp.zeros(4, dtype=jnp.int32))
    zeros = lambda point, indices, payload: jnp.zeros(3, dtype=jnp.float64)
    tied = replay_native_candidates(base, anchor, jnp.float64(0), plan, fixed, zeros, None)
    np.testing.assert_array_equal(tied['best_indices'], anchor)
    assert not tied['trace']['accepted'].any()


def test_native_category_permutation_is_not_alphabetical_order():
    np.testing.assert_array_equal(orthogonal_offsets([2, 3], [1, 0]),
                                  [[0, 0], [0, 1], [0, 2], [1, 0]])
    for sizes, order in [([0], [0]), ([2, 2], [0, 0]), ([2], [.5])]:
        with pytest.raises(ValueError):
            orthogonal_offsets(sizes, order)
