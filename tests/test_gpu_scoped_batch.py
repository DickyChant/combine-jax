"""Prevent mixed evidence and queued jobs after a scoped batch interruption."""

import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import run_gpu_scoped_batch as batch


def authorize_mock_directory(monkeypatch):
    monkeypatch.setattr(batch.Path, "cwd", classmethod(lambda cls: Path("/tmp/sqian/combine-jax")))
    monkeypatch.setattr(batch.socket, "getfqdn", lambda: "lxplus905.cern.ch")


def test_preflight_rejects_other_host(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(RuntimeError, match="authorized directory"):
        batch.check_preflight("smoke", {})


def test_preflight_rejects_active_prior_driver(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    authorize_mock_directory(monkeypatch)
    Path("reports").mkdir()
    Path("reports/gpu_spooled_hhvv_driver.json").write_text('{"status":"running"}')
    with pytest.raises(RuntimeError, match="still marked running"):
        batch.check_preflight("smoke", {})


def test_preflight_preserves_existing_child_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    authorize_mock_directory(monkeypatch)
    Path("reports").mkdir()
    Path("reports/gpu_counting_gpu_scoped.log").write_text("preserved")
    with pytest.raises(FileExistsError, match="Preserve existing evidence"):
        batch.check_preflight("smoke", {})


def test_preflight_binds_hhvv_to_smoke_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    authorize_mock_directory(monkeypatch)
    Path("reports").mkdir()
    Path("reports/gpu_scoped_smoke_driver.json").write_text(json.dumps(
        dict(status="passed", sources={"changed": "source"})))
    Path("reports/gpu_comparison_scoped.json").write_text('{"models":{}}')
    with pytest.raises(AssertionError):
        batch.check_preflight("hhvv", {})


@pytest.mark.parametrize("phase", ("smoke", "hhvv"))
def test_interrupt_terminates_child_and_does_not_launch_next(tmp_path, monkeypatch, phase):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    monkeypatch.setattr(batch, "sources", lambda: {})
    monkeypatch.setattr(batch, "check_preflight", lambda *args: ("counting", "automcstats"))
    monkeypatch.setattr(sys, "argv", ["run_gpu_scoped_batch.py", phase])
    started = []

    class Child:
        pid = 12345

        def __init__(self, command, **kwargs):
            started.append(self)
            self.terminated = False

        def poll(self):
            return None

        def terminate(self):
            self.terminated = True

        def wait(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            return -signal.SIGTERM

    monkeypatch.setattr(batch.subprocess, "Popen", Child)
    with pytest.raises(SystemExit) as error:
        batch.main()
    assert error.value.code == 1
    assert len(started) == 1 and started[0].terminated
    result = json.loads(Path(f"reports/gpu_scoped_{phase}_driver.json").read_text())
    assert result["status"] == "interrupted" and result["signal"] == "SIGTERM"
    assert result["jobs"][0]["exit_code"] == -signal.SIGTERM
    assert result["jobs"][0]["command"] == batch.job_command("counting", "gpu")
