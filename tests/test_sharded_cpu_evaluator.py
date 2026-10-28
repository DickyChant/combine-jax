"""Spawned workers preserve every term, shared gradient and a single offset."""

import multiprocessing as mp
import os
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from sharded_cpu_evaluator import ShardedCpuEvaluator, NO_FUSION_OPTIONS, core_digest, digest


@pytest.fixture
def exported(counting_path, tmp_path):
    from combine_jax.likelihood import from_workspace
    from combine_jax.root_io import open_workspace
    from hhvv_export_cache import save_export
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    # A nonzero offset catches accidental addition once per worker.
    likelihood = likelihood.at_zero(likelihood.initial)
    assert likelihood.offset != 0.
    source = dict(synthetic_test="counting_with_nominal_offset")
    path = tmp_path/"export.pkl"
    save_export(likelihood, path, source)
    spec = dict(export=str(path), export_sha256=digest(path), source=source, core_sha256=core_digest())
    return likelihood, spec


@pytest.mark.parametrize("no_fusion", [False, True])
def test_spawned_full_gradient_order_shared_parameters_and_offset(exported, no_fusion):
    likelihood, spec = exported
    affinity = os.sched_getaffinity(0)
    events = []
    with ShardedCpuEvaluator(spec, workers=2, rss_limit_mib=4000., wall_seconds=120.,
                             on_progress=events.append, cpu_no_fusion=no_fusion) as pool:
        with pytest.raises(RuntimeError, match="Prepare"):
            pool.value_and_grad(likelihood.initial)
        pool.prepare()
        assert events and len(pool.preparation) == len(pool.term_metadata)
        assert pool.prepared and all(r["reused"] for r in pool.prepare())
        local = likelihood.local_evaluator(constraint_batch_size=32)
        local.prepare("value_and_grad", likelihood.initial, release_cache=True)
        for point in (likelihood.initial, likelihood.initial+np.array([.3, -.7, .5])):
            value, gradient = pool.value_and_grad(point)
            expected, expected_gradient = local.value_and_grad(point)
            if no_fusion:
                np.testing.assert_allclose(value, expected, atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(gradient, expected_gradient, atol=1e-12, rtol=1e-12)
            else:
                np.testing.assert_equal(value, expected)
                np.testing.assert_array_equal(gradient, expected_gradient)
        with pytest.raises(ValueError, match="shape"):
            pool.value_and_grad([0.])
        with pytest.raises(NotImplementedError, match="no Hessian"):
            pool.hessian(likelihood.initial)
        assert all(w["float64"] and w["devices"] == ["cpu:0"] for w in pool.worker_metadata)
        assert all(not w["root_imported"] for w in pool.worker_metadata)
        assert all(w["gradient_compiler_options"] == (NO_FUSION_OPTIONS if no_fusion else {})
                   for w in pool.worker_metadata)
        assert pool.likelihood.discrete_choices == likelihood.discrete_choices
        assert pool.likelihood.mask_configuration == likelihood.mask_configuration
        assert len(set(pool.cores)) == 3
    assert all(not p.is_alive() and p.exitcode == 0 for p in pool.processes)
    assert os.sched_getaffinity(0) == affinity
    with pytest.raises(RuntimeError, match="Prepare"):
        pool.value_and_grad(likelihood.initial)


def test_aggregate_memory_guard_cleans_up_workers(exported):
    _, spec = exported
    before = {p.pid for p in mp.active_children()}
    affinity = os.sched_getaffinity(0)
    with pytest.raises(MemoryError, match="Aggregate"):
        ShardedCpuEvaluator(spec, workers=2, rss_limit_mib=1., wall_seconds=120.)
    assert {p.pid for p in mp.active_children()} == before
    assert os.sched_getaffinity(0) == affinity


@pytest.mark.parametrize("options", [dict(workers=0), dict(workers=8), dict(rss_limit_mib=12000),
                                    dict(wall_seconds=0), dict(rss_limit_mib=float("nan"))])
def test_invalid_resource_policy_precedes_process_launch(options):
    before = {p.pid for p in mp.active_children()}
    with pytest.raises(ValueError):
        ShardedCpuEvaluator({}, **options)
    assert {p.pid for p in mp.active_children()} == before
