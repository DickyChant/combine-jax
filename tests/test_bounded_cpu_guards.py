"""The bounded fit preserves evidence and rejects invalid work before JAX starts."""

from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fit_bounded_cpu as runner


def arguments(monkeypatch, *extra):
    monkeypatch.setattr(sys, "argv", ["fit_bounded_cpu.py", "--model", "counting",
                                     "--output", "reports/test.json", *extra])


def test_existing_report_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("reports/test.json").write_text("preserved")
    arguments(monkeypatch)
    with pytest.raises(FileExistsError):
        runner.main()
    assert Path("reports/test.json").read_text() == "preserved"


@pytest.mark.parametrize("options", [("--rss-limit-mib", "nan"), ("--rss-limit-mib", "12000"),
                                    ("--wall-seconds", "0"), ("--max-calls", "0"),
                                    ("--migrad-iterate", "0"), ("--migrad-iterate", "-1")])
def test_invalid_limits_precede_initialization(monkeypatch, options):
    arguments(monkeypatch, *options)
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_local_experiment_cannot_start_on_lxplus(monkeypatch):
    arguments(monkeypatch)
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/tmp/sqian/combine-jax")))
    with pytest.raises(RuntimeError, match="local only"):
        runner.main()


def test_hhvv_minimum_requires_checked_restart(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_bounded_cpu.py", "--model", "hhvv",
                                     "--output", "reports/test.json"])
    with pytest.raises(SystemExit) as error:
        runner.main()
    assert error.value.code == 2


def test_reference_failure_restores_arguments_and_preserves_log(tmp_path, monkeypatch):
    original = ["original", "arguments"]
    monkeypatch.setattr(sys, "argv", original)

    def failed():
        print("reference failed")
        raise AssertionError("failed reference")

    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(main=failed))
    log = tmp_path/"reference.log"
    with pytest.raises(AssertionError, match="failed reference"):
        runner.run_checker("reference", ["example.json"], log)
    assert sys.argv is original
    assert log.read_text() == "reference failed\n"
