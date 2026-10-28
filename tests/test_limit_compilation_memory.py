"""Explicit fits remain numerically identical after releasing tracing caches."""
import numpy as np

from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.limit_profile import LimitProfiler


def test_selected_compilations_and_cache_release_preserve_fits(counting_path):
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, 'tests/cards/counting.txt')
    reference = LimitProfiler(model, 'r', global_up=.5, global_start_fraction=.1)
    baseline = reference.global_minimum()
    global_point = reference.point()
    profiled = [reference(mu) for mu in (0., .4, 1.)]
    # The global checkpoint must not become the last conditional warm start.
    saved = reference.records[0]
    assert saved['parameter_names'] == reference.names
    np.testing.assert_array_equal(saved['parameter_values'], global_point)
    assert all('parameter_values' not in row for row in reference.records[1:])
    events = []
    bounded = LimitProfiler(model, 'r', global_up=.5, global_start_fraction=.1,
        compile_kinds=('global', 'conditional'), release_compile_cache=True,
        compile_progress=lambda name, stage, *_: events.append((name, stage)))
    assert set(bounded.executables) == {'global', 'conditional'}
    np.testing.assert_array_equal(bounded.global_minimum(), baseline)
    np.testing.assert_array_equal(bounded.point(), global_point)
    np.testing.assert_array_equal([bounded(mu) for mu in (0., .4, 1.)], profiled)
    assert all(r['status'] == 0 and r['corrected_estimated_edm'] < 1e-4 for r in bounded.records)
    assert events == [(name, stage) for name in ('global', 'conditional')
                      for stage in ('lowering', 'lowered', 'compiling', 'compiled')]
    # The background-only stage needs just this executable. Compare from the
    # same initial point, including a second warm-started conditional fit.
    reference.warm = reference.initial
    expected = [reference(mu) for mu in (0., .4)]
    background = LimitProfiler(model, 'r', compile_kinds=('conditional',),
                               release_compile_cache=True)
    background.warm = background.initial
    assert set(background.executables) == {'conditional'}
    np.testing.assert_array_equal([background(mu) for mu in (0., .4)], expected)
    # Releasing process-wide traces must leave the earlier explicit executable
    # usable too, including its standalone NLL.
    np.testing.assert_array_equal(reference.unprofiled(.4), expected[-1])


def test_deferred_compilation_does_not_retain_the_imported_model(counting_path):
    import gc
    import weakref
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, 'tests/cards/counting.txt')
    direct = LimitProfiler(model, 'r', global_up=.5)
    baseline = direct.global_minimum()
    expected = [direct(mu) for mu in (0., .4, 1.)]
    delayed = LimitProfiler(model, 'r', global_up=.5, defer_compile=True,
                            release_compile_cache=True)
    owner = weakref.ref(model)
    del model
    gc.collect()
    assert owner() is None and delayed.executables == {}
    delayed.compile()
    np.testing.assert_array_equal(delayed.global_minimum(), baseline)
    np.testing.assert_array_equal([delayed(mu) for mu in (0., .4, 1.)], expected)
