"""A fit requires complete, source-bound gradient evidence and fresh output."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fit_sharded_cpu as runner


@pytest.fixture
def manifest():
    path = Path("build/gpu_exports/manifest_random.json")
    if not path.exists():
        pytest.skip("Generate trusted benchmark exports before this artifact integration check")
    result = json.loads(path.read_bytes())
    for model in ("counting", "automcstats"):
        report = json.loads(Path(f"reports/gpu_local_{model}_sharded_cpu_gradient_options_default.json").read_bytes())
        if not Path(report["arrays"]).exists() or not (Path("build/gpu_exports")/result["models"][model]["cache"]).exists():
            pytest.skip("Generate trusted smoke arrays/exports before this artifact integration check")
    return result


@pytest.mark.parametrize("model", ["counting", "automcstats"])
@pytest.mark.parametrize("mode", ["default", "no_fusion"])
def test_real_smoke_evidence_passes(model, mode, manifest):
    path = Path(f"reports/gpu_local_{model}_sharded_cpu_gradient_options_{mode}.json")
    report, provenance = runner.check_gradient_evidence(path, model, manifest)
    assert report["status"] == "passed"
    assert provenance[str(path)] == runner.digest(path)


@pytest.mark.parametrize("fault", ["core", "helper", "runner", "manifest", "status",
                                 "incomplete", "unclean_exit", "arrays_hash", "options"])
def test_unusable_evidence_rejected(tmp_path, manifest, fault):
    report = json.loads(Path("reports/gpu_local_counting_sharded_cpu_gradient_options_default.json").read_bytes())
    if fault in ("core", "helper", "runner", "manifest", "arrays_hash"):
        field = dict(core="source_sha256", helper="helper_sha256", runner="runner_sha256",
                     manifest="reference_manifest_sha256", arrays_hash="arrays_sha256")[fault]
        report[field] = "changed"
    elif fault == "status":
        report["status"] = "running"
    elif fault == "incomplete":
        report["reference_validation"].pop()
    elif fault == "unclean_exit":
        report["worker_cleanup"][0]["exit_code"] = -9
    elif fault == "options":
        report["gradient_compiler_options"] = {"unsupported": True}
    path = tmp_path/"changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        runner.check_gradient_evidence(path, "counting", manifest)


@pytest.mark.parametrize("field", ["validation_vg_values", "validation_gradients", "validation_points"])
def test_saved_arrays_rechecked_against_reference(tmp_path, manifest, field):
    report = json.loads(Path("reports/gpu_local_counting_sharded_cpu_gradient_options_default.json").read_bytes())
    with np.load(report["arrays"], allow_pickle=False) as loaded:
        arrays = {k: loaded[k] for k in loaded.files}
    arrays[field] = arrays[field]+1.
    artifact = tmp_path/"changed.npz"
    np.savez_compressed(artifact, **arrays)
    report.update(arrays=str(artifact), arrays_sha256=runner.digest(artifact))
    path = tmp_path/"changed.json"
    path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):
        runner.check_gradient_evidence(path, "counting", manifest)


@pytest.mark.parametrize("no_fusion", [False, True])
def test_fit_cannot_change_validated_compiler_options(monkeypatch, manifest, no_fusion):
    mode = "default" if no_fusion else "no_fusion"
    output = Path("reports/test_sharded_option_mismatch.json")
    assert not output.exists()
    argv = ["fit_sharded_cpu.py", "--model", "counting", "--validated-gradient",
            f"reports/gpu_local_counting_sharded_cpu_gradient_options_{mode}.json",
            "--output", str(output)]
    if no_fusion:
        argv.append("--cpu-no-fusion")
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(AssertionError, match="compiler options differ"):
        runner.main()
    assert not output.exists()


def arguments(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["fit_sharded_cpu.py", "--model", "counting",
        "--validated-gradient", "absent.json", "--output", "reports/test.json"])


def test_existing_output_precedes_validation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path("reports/test.json").write_text("preserved")
    arguments(monkeypatch)
    with pytest.raises(FileExistsError):
        runner.main()
    assert Path("reports/test.json").read_text() == "preserved"


@pytest.mark.parametrize("policy,phase", [("spooled", "smoke"), ("spooled", "hhvv"),
                                        ("scoped", "smoke"), ("scoped", "hhvv")])
def test_node_overlap_precedes_validation(tmp_path, monkeypatch, policy, phase):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    Path(f"reports/gpu_{policy}_{phase}_driver.json").write_text('{"status":"running"}')
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: Path("/tmp/sqian/combine-jax")))
    arguments(monkeypatch)
    with pytest.raises(RuntimeError, match="active benchmark"):
        runner.main()
    assert not Path("reports/test.json").exists()
