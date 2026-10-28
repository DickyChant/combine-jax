"""Profile grids must be explicit, bounded, shared and bound to accepted fits."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import threading

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import profile_plan as plans
import run_profile_points as runner


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    names = ["nuisance", "poi", "fixed"]
    parameters = [SimpleNamespace(name=n, minimum=lo, maximum=hi)
                  for n, lo, hi in zip(names, [-5., -20., 7.], [5., 20., 7.])]
    likelihood = SimpleNamespace(parameter_names=tuple(names), parameters=parameters,
                                 discrete_choices={"synthetic": 1}, mask_configuration={})
    native = dict(parameter_names=names, discrete_choices=likelihood.discrete_choices,
                  channel_masks={}, fixed_bounds=["fixed"], arrays="synthetic", arrays_sha256="synthetic",
                  model="synthetic", source={"sha256": "synthetic"}, host="synthetic-host",
                  canonical_fitted_nll=2., combine_version="11.0.0", root_version="6.40.00")
    comparison = dict(benchmark="jax", benchmark_sha256="synthetic", reference="native",
                      reference_sha256="synthetic", checker_sha256="synthetic")
    arrays = dict(parameter_names=np.asarray(names), values=np.asarray([0., 10., 7.]),
                  errors=np.asarray([1., 2., .1]))
    monkeypatch.setattr(plans, "validated_comparison", lambda p: (comparison, {}, native))
    monkeypatch.setattr(plans, "load_arrays", lambda r: arrays)
    path = tmp_path/"comparison.json"
    path.write_text('{"synthetic_fixture":true}')
    return likelihood, native, arrays, path


def test_explicit_native_error_grid_and_common_start(setup):
    likelihood, native, arrays, comparison = setup
    plan = plans.make_plan(comparison, likelihood, "nuisance", "poi", [-2., -1., 0., 1., 2.])
    assert [p["value"] for p in plan["points"]] == [-2., -1., 0., 1., 2.]
    assert plan["initial"] == [0., 10., 7.] and plan["baseline_errors"] == [1., 2., .1]
    assert plan["parameter_index"] == 0 and plan["poi_index"] == 1
    assert "not claims" in plan["scope"]


@pytest.mark.parametrize("offsets", [[-1., 1.], [-1., 0., 0., 1.], [0., 1., 2.],
                                    [-1., 0., np.nan], [-6., 0., 1.], [[-1., 0., 1.]]])
def test_invalid_grid_never_clips_or_relabels(setup, offsets):
    likelihood, _, _, comparison = setup
    with pytest.raises(ValueError):
        plans.make_plan(comparison, likelihood, "nuisance", "poi", offsets)


@pytest.mark.parametrize("parameter,poi", [("missing", "poi"), ("nuisance", "missing"),
                                         ("nuisance", "nuisance"), ("fixed", "poi"), ("nuisance", "fixed")])
def test_require_distinct_free_coordinates(setup, parameter, poi):
    likelihood, _, _, comparison = setup
    with pytest.raises(ValueError):
        plans.make_plan(comparison, likelihood, parameter, poi, [-1., 0., 1.])


@pytest.mark.parametrize("fault", ["order", "category", "mask", "start_bounds"])
def test_baseline_configuration_is_bound(setup, fault):
    likelihood, native, arrays, comparison = setup
    if fault == "order":
        native["parameter_names"] = native["parameter_names"][::-1]
    elif fault == "category":
        native["discrete_choices"] = {"synthetic": 2}
    elif fault == "mask":
        native["channel_masks"] = {"changed": True}
    else:
        arrays["values"][1] = 30.
    with pytest.raises(AssertionError):
        plans.make_plan(comparison, likelihood, "nuisance", "poi", [-1., 0., 1.])


@pytest.mark.parametrize("fault", [None, "point", "start", "host", "export", "manifest"])
def test_plan_revalidation_rejects_tampering(setup, fault):
    likelihood, native, _, comparison = setup
    plan = plans.make_plan(comparison, likelihood, "nuisance", "poi", [-1., 0., 1.])
    plans.MANIFEST.parent.mkdir(parents=True)
    export = plans.MANIFEST.parent/"synthetic.pkl"
    export.write_bytes(b"synthetic bytes, never unpickled")
    plans.MANIFEST.write_text(json.dumps(dict(models=dict(synthetic=dict(
        source=native["source"], cache=export.name)))))
    plan.update(manifest=str(plans.MANIFEST), manifest_sha256=plans.digest(plans.MANIFEST),
                export=str(export), export_sha256=plans.digest(export))
    if fault == "point":
        plan["points"][0]["value"] = -.9
    elif fault == "start":
        plan["initial"][1] = 11.
    elif fault == "host":
        plan["host"] = "different-host"
    elif fault == "export":
        export.write_bytes(b"changed")
    elif fault == "manifest":
        plans.MANIFEST.write_text("{}")
    path = Path("plan.json")
    path.write_text(json.dumps(plan))
    if fault is None:
        assert plans.load_plan(path, likelihood, native["source"], host="synthetic-host") == plan
    else:
        with pytest.raises(AssertionError):
            plans.load_plan(path, likelihood, native["source"], host="synthetic-host")


@pytest.mark.parametrize("policy,phase", [("spooled", "smoke"), ("spooled", "hhvv"),
                                         ("scoped", "smoke"), ("scoped", "hhvv")])
def test_profile_overlap_guard_precedes_backend_initialization(tmp_path, monkeypatch, policy, phase):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path(f"reports/gpu_{policy}_{phase}_driver.json").write_text('{"status":"running"}')
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/tmp/sqian/combine-jax")))
    monkeypatch.setattr(runner.platform, "node", lambda: "lxplus905.cern.ch")
    monkeypatch.setattr(sys, "argv", ["run_profile_points.py", "--plan", "missing",
                                    "--engine", "gpu", "--output", "reports/new.json"])
    with pytest.raises(RuntimeError, match="active benchmark"):
        runner.main()
    assert not Path("reports/new.json").exists()


def test_existing_profile_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    path = Path("reports/existing.json")
    path.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["run_profile_points.py", "--plan", "missing",
                                    "--engine", "cpu", "--output", str(path)])
    with pytest.raises(FileExistsError):
        runner.main()
    assert path.read_text() == "preserve"


@pytest.mark.parametrize("interrupted", [False, True])
def test_point_storage_preserves_prior_artifact_on_interrupted_write(tmp_path, monkeypatch, interrupted):
    artifact = tmp_path/"profile.npz"
    artifact.write_bytes(b"previous complete artifact")
    report = {"points": []}
    record = {"valid_minimum": True}
    plan = dict(parameter_names=["a"], initial=[0.], points=[{"value": 1.}])
    snapshots = []
    if interrupted:
        def interrupted_write(path, **arrays):
            Path(path).write_bytes(b"partial new data")
            raise RuntimeError("simulated interruption")
        monkeypatch.setattr(runner.np, "savez_compressed", interrupted_write)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            runner.store_profile_point(artifact, report, record, [[1.]], [2.], plan,
                                       threading.RLock(), lambda: snapshots.append(dict(report)))
        assert artifact.read_bytes() == b"previous complete artifact"
        assert not report["points"] and not snapshots
    else:
        runner.store_profile_point(artifact, report, record, [[1.]], [2.], plan,
                                   threading.RLock(), lambda: snapshots.append(dict(report)))
        assert report["points"] == [record] and len(snapshots) == 1
        assert snapshots[0]["arrays_sha256"] == runner.digest(artifact)
        with np.load(artifact, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["values"], [[1.]])
