"""Incomplete producers and failed validations cannot launch a minimum fit."""

import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_sharded_fit_after_validation as runner


@pytest.fixture
def job(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("initial.json").write_text(json.dumps(dict(
        status="candidate_checked", parent="parent.json", parent_sha256="testhash")))
    monkeypatch.setattr(runner, "digest", lambda path: "testhash")
    monkeypatch.setattr(runner.signal, "signal", lambda *args: None)
    monkeypatch.setattr(sys, "argv", ["continuation", "--gradient-report", "gradient.json",
        "--initial-check", "initial.json", "--output", "reports/driver.json",
        "--fit-output", "reports/fit.json", "--wait-seconds", ".001"])
    gradient = dict(status="computed", model="hhvv", float64=True,
        worker_cleanup=[dict(exit_code=0, signals=[])],
        gradient_compiler_options=dict(xla_disable_hlo_passes="fusion", xla_cpu_parallel_codegen_split_count=1),
        completed_operations=["value_and_grad"], reference_validation=[dict(nll_absolute_error=0.)]*26,
        arrays="arrays.npz", arrays_sha256="testhash", runner_sha256="testhash", helper_sha256="testhash")
    launched = []

    def start(command, **kwargs):
        launched.append(command)
        index = len(launched)
        output = ("gradient_resolved_gradient.json" if index == 1 else
                  "gradient_combine_random_gradients_refined.json" if index == 2 else "reports/fit.json")

        def wait():
            Path(output).write_text(json.dumps(dict(status="minimum_computed" if index == 3 else "passed")))
            return 0

        return SimpleNamespace(pid=123, wait=wait)

    monkeypatch.setattr(runner.subprocess, "Popen", start)
    return gradient, launched


@pytest.mark.parametrize("status", ["resource_limit", "failed", "reference_failed"])
def test_failed_producer_never_starts_children(job, status):
    gradient, launched = job
    gradient["status"] = status
    Path("gradient.json").write_text(json.dumps(gradient))
    assert runner.main() == 1
    assert launched == []
    report = json.loads(Path("reports/driver.json").read_bytes())
    assert report["status"] == "producer_failed" and report["stages"] == []


@pytest.mark.parametrize("status", ["running", "computed"])
def test_missing_worker_cleanup_waits_without_launching(job, monkeypatch, status):
    gradient, launched = job
    gradient["status"] = status
    gradient.pop("worker_cleanup")
    Path("gradient.json").write_text(json.dumps(gradient))
    clock = iter(range(100))
    monkeypatch.setattr(runner.time, "monotonic", lambda: next(clock))
    assert runner.main() == 1
    assert launched == []
    assert json.loads(Path("reports/driver.json").read_bytes())["status"] == "observation_limit"


def test_failed_reference_cannot_launch_fit(job, monkeypatch):
    gradient, launched = job
    Path("gradient.json").write_text(json.dumps(gradient))

    def failed(command, **kwargs):
        launched.append(command)
        Path("gradient_resolved_gradient.json").write_text('{"status":"failed"}')
        return SimpleNamespace(pid=123, wait=lambda: 0)

    monkeypatch.setattr(runner.subprocess, "Popen", failed)
    assert runner.main() == 1
    assert len(launched) == 1 and "reconcile_gpu_gradient_references.py" in launched[0][1]
    assert json.loads(Path("reports/driver.json").read_bytes())["status"] == "stage_failed"


def test_only_both_passing_checks_allow_fit(job):
    gradient, launched = job
    Path("gradient.json").write_text(json.dumps(gradient))
    assert runner.main() == 0
    assert len(launched) == 3
    assert "reconcile_gpu_gradient_references.py" in launched[0][1]
    assert "compare_gpu_random_gradients.py" in launched[1][1]
    assert "fit_sharded_cpu.py" in launched[2][1] and "--cpu-no-fusion" in launched[2]
    report = json.loads(Path("reports/driver.json").read_bytes())
    assert report["status"] == "minimum_computed"
    assert all(stage["status"] == "complete" for stage in report["stages"])
