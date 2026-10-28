"""Reject unsafe restarts and overlapping benchmark use before numerical work."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fit_exported_likelihood as runner


def arguments(monkeypatch, tag="test"):
    monkeypatch.setattr(sys, "argv", ["fit_exported_likelihood.py", "--model", "counting",
                                     "--backend", "cpu", "--report-tag", tag])


def test_existing_output_is_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    output = Path("reports/gpu_counting_cpu_fit_test.json")
    output.write_text("preserved")
    arguments(monkeypatch)
    with pytest.raises(FileExistsError):
        runner.main()
    assert output.read_text() == "preserved"


@pytest.mark.parametrize("policy,phase", (("spooled","smoke"), ("spooled","hhvv"),
                                         ("scoped","smoke"), ("scoped","hhvv")))
def test_remote_batch_guard_runs_before_backend_initialization(tmp_path, monkeypatch, policy, phase):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path(f"reports/gpu_{policy}_{phase}_driver.json").write_text('{"status":"running"}')
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/tmp/sqian/combine-jax")))
    monkeypatch.setattr(runner.platform, "node", lambda: "lxplus905.cern.ch")
    arguments(monkeypatch)
    with pytest.raises(RuntimeError, match="batch must record completion"):
        runner.main()
    assert not Path("reports/gpu_counting_cpu_fit_test.json").exists()


def test_invalid_output_tag_rejected(monkeypatch):
    arguments(monkeypatch, "../elsewhere")
    with pytest.raises(SystemExit) as exc:
        runner.main()
    assert exc.value.code == 2


@pytest.fixture
def restart(tmp_path):
    parent = tmp_path/"parent.json"
    parent.write_text('{"synthetic":true}')
    source = dict(synthetic=True)
    manifest = dict(combine_version="11.0.0", root_version="6.40.00")
    likelihood = SimpleNamespace(initial=np.array([.2, .5]), parameter_names=("a", "fixed"),
        discrete_choices={"index": 1}, mask_configuration={},
        parameters=[SimpleNamespace(minimum=-1., maximum=1.),
                    SimpleNamespace(minimum=.5, maximum=.5)])
    checked = dict(status="candidate_checked", source=source, parent=str(parent),
        parent_sha256=runner.digest(parent), **manifest,
        discrete_choices=likelihood.discrete_choices, channel_masks={},
        parameter_names=list(likelihood.parameter_names), point=[.1, .5],
        saved_zero_point_absolute_error=0., canonical_offset_absolute_error=0.,
        canonical_restoration_absolute_error=0.,
        probes=[dict(roofit_evaluation_errors=0) for _ in range(5)],
        canonical_candidate_nll=1.25)
    return likelihood, source, manifest, checked, tmp_path/"check.json"


def test_checked_restart_retains_point_and_canonical_reference(restart):
    likelihood, source, manifest, checked, path = restart
    path.write_text(json.dumps(checked))
    point, value = runner.checked_initial(likelihood, source, manifest, path)
    np.testing.assert_array_equal(point, [.1, .5])
    assert value == 1.25


@pytest.mark.parametrize("fault", ["parent_hash", "source", "bounds", "fixed", "order", "roofit_error"])
def test_bad_restart_rejected(restart, fault):
    likelihood, source, manifest, checked, path = restart
    if fault == "parent_hash":
        checked["parent_sha256"] = "changed"
    elif fault == "source":
        checked["source"] = {"different": True}
    elif fault == "bounds":
        checked["point"][0] = 2.
    elif fault == "fixed":
        checked["point"][1] = .6
    elif fault == "order":
        checked["parameter_names"].reverse()
    elif fault == "roofit_error":
        checked["probes"][0]["roofit_evaluation_errors"] = 1
    path.write_text(json.dumps(checked))
    with pytest.raises(AssertionError):
        runner.checked_initial(likelihood, source, manifest, path)
