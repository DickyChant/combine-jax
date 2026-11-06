"""Fit progress records: read-only, throttled, one callback per batch update."""
import io
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor import progress
from combine_jax.tensor.optimize import make_bfgs
from combine_jax.progress_display import ProgressDisplay, Fit, bar
from combine_jax.runtime.progress_follow import ProgressFollower


def rosenbrock(x, _):
    return 100.*jnp.sum((x[1:]-x[:-1]**2)**2)+jnp.sum((1.-x[:-1])**2)


def arguments(start, n=6):
    return jnp.full(n, start), {}, jnp.full(n, -5.), jnp.full(n, 5.)


def collect(function, *args):
    records = []
    with progress.sink(records.append):
        result = jax.device_get(jax.block_until_ready(function(*args)))
    return result, records


def test_progress_is_read_only_and_throttled():
    plain = jax.device_get(jax.jit(make_bfgs(rosenbrock, stopping_criterion='edm'))(*arguments(-1.2)))
    fn = jax.jit(make_bfgs(rosenbrock, stopping_criterion='edm', progress_every=7))
    result, records = collect(fn, *arguments(-1.2))
    for name in ('x', 'value', 'iteration', 'status', 'corrected_estimated_edm'):
        np.testing.assert_array_equal(result[name], plain[name])
    periodic, final = records[:-1], records[-1]
    assert [r['iteration'] for r in periodic] == list(range(0, result['iteration'], 7))
    assert periodic[0]['start'] and not any(r['start'] for r in periodic[1:])
    assert final['final'] and final['status'] == result['status'] == 0
    assert final['iteration'] == result['iteration'] and final['value'] == result['value']
    assert final['edm'] == pytest.approx(result['corrected_estimated_edm'], rel=1e-12)


def test_disabled_progress_adds_no_callback():
    trace = jax.make_jaxpr(make_bfgs(rosenbrock))(*arguments(-1.2))
    assert 'callback' not in str(trace)
    with pytest.raises(ValueError, match='progress_every'):
        make_bfgs(rosenbrock, progress_every=-1)


def test_batched_fit_sends_one_record_per_update():
    fn = make_bfgs(rosenbrock, stopping_criterion='edm', progress_every=5)
    starts = jnp.array([-1.2, 0., 2.])
    batched = jax.jit(jax.vmap(lambda s: fn(*arguments(s))))
    result, records = collect(batched, starts)
    plain = make_bfgs(rosenbrock, stopping_criterion='edm')
    expected = jax.device_get(jax.jit(jax.vmap(lambda s: plain(*arguments(s))))(starts))
    np.testing.assert_array_equal(result['x'], expected['x'])
    np.testing.assert_array_equal(result['iteration'], expected['iteration'])
    assert all(len(r['iteration']) == 3 for r in records)
    final = records[-1]
    assert final['final'] == [True]*3
    assert final['iteration'] == result['iteration'].tolist()
    assert final['status'] == result['status'].tolist()
    # A lane that has finished keeps reporting its final state, inactive.
    for r in records[:-1]:
        for lane in range(3):
            if not r['active'][lane]:
                assert r['iteration'][lane] == result['iteration'][lane]
    # One update per due iteration of the longest lane, not one per lane.
    assert len(records)-1 == len(range(0, int(result['iteration'].max()), 5))


def test_task_names_are_attached_on_host():
    fn = jax.jit(make_bfgs(rosenbrock, progress_every=50))
    with progress.task('kl=1'):
        _, records = collect(fn, *arguments(-1.2))
    assert {r['task'] for r in records} == {'kl=1'}


def record(**fields):
    base = dict(iteration=0, value=10., edm=1e2, max_gradient=1., directional_edm=float('nan'),
                repairs=0, validating=False, active=True, start=False, final=False, status=-1,
                budget=10**8, tolerance=1e-4, edm_criterion=True)
    return dict(base, **fields)


def test_convergence_fraction_is_log_scale_and_monotone():
    fit = Fit(record(edm=1e2), 0., 20)
    fit.update(record(iteration=10, edm=1e-1), 1.)
    assert fit.fraction() == pytest.approx(.5)
    fit.update(record(iteration=20, edm=1e1), 2.)   # worse EDM: bar keeps the best
    assert fit.fraction() == pytest.approx(.5)
    fit.update(record(iteration=30, edm=1e-5, validating=True), 3.)
    assert fit.fraction() == .99 and 'validating' in fit.flags()
    fit.update(record(iteration=31, edm=1e-5, final=True, status=0, active=False), 4.)
    assert fit.fraction() == 1. and fit.flags() == '✓'


def test_stalled_flag_and_capped_bar():
    fit = Fit(record(edm=1., budget=1000), 0., 3)
    for i in range(1, 5):
        fit.update(record(iteration=10*i, edm=1., budget=1000), i)
    assert fit.stalled and 'stalled' in fit.flags()
    line = fit.line('t', 5.)
    assert '40/1000' in line and '≤' in line
    assert '/' not in Fit(record(), 0., 3).line('t', 1.).split('converge')[0]
    assert bar(0.) == '░'*12 and bar(1.) == '█'*12


def test_plain_display_prints_one_line_per_finished_fit():
    out = io.StringIO()
    display = ProgressDisplay(out, mode='plain', total=2, plain_interval=1e9)
    for name in ('a', 'b'):
        display.update(record(task=name, start=True))
        display.update(record(task=name, iteration=5, edm=1e-5, final=True, status=0, active=False))
    display.close()
    lines = out.getvalue().splitlines()
    assert lines[-2].startswith('[1/2] a') and lines[-1].startswith('[2/2] b')


def test_follower_waits_for_complete_lines(tmp_path):
    path = tmp_path/'progress.jsonl'
    out = io.StringIO()
    follower = ProgressFollower(path, stream=out, mode='plain', plain_interval=0.)
    follower.poll()
    text = json.dumps(record(task='g', start=True))+'\n'
    path.write_text(text[:20])
    follower.poll()
    assert out.getvalue() == ''
    path.write_text(text)
    follower.poll()
    assert out.getvalue().startswith('g  it 0')
