"""Synthetic evidence tests: reject partial stages, mixed sources and copy races."""

import io
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import capture_scoped_gpu_results as capture


def fixture(stage="gradient_all_points", *, nonfinite=False):
    points = np.zeros((26, 3), dtype=np.float64)
    reference = dict(parameter_names=["a", "b", "c"], cases=[
        dict(theta=p.tolist(), label=f"case_{i}", combine_nll=2.) for i, p in enumerate(points)])
    arrays = dict(parameter_names=np.asarray(reference["parameter_names"]), point=points[0],
                  validation_points=points, validation_labels=np.asarray([c["label"] for c in reference["cases"]]),
                  value=np.asarray(2.), validation_values=np.full(26, 2.),
                  term_values=np.asarray([1., 1.]), vg_value=np.asarray(2.), gradient=np.ones(3),
                  validation_vg_values=np.full(26, 2.), validation_gradients=np.ones((26, 3)),
                  hessian=np.eye(3))
    if nonfinite:
        arrays["hessian"][0, 0] = np.nan
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    operation = capture.OPERATIONS[stage]
    terms = [dict(name=name, parameters=3, compiler_options={}) for name in ("first", "second")]
    timing = dict(term_timings=terms, first_execution_seconds=2., warm_seconds=[1.]*5,
                  median_warm_seconds=1.,
                  spooled_timed_calls=[dict(total_seconds=t, kernel_calls=2) for t in [2., *([1.]*5)]],
                  scoped_load_sessions={t["name"]: [dict(calls=26, exception_type=None)] for t in terms},
                  spooled_all_calls={t["name"]: dict(count=32) for t in terms})
    report = dict(parameters=3, terms=2, hessian_compiler_options={}, timings={operation: timing},
                  preparation_timings={operation: dict(completed_terms=2)},
                  completed_operations=["value", "value_and_grad", "hessian"],
                  reference_validation=[dict(nll_absolute_error=0., value_gradient_nll_absolute_error=0.)
                                        for _ in range(26)],
                  arrays_sha256=capture.transport.digest(buffer.getvalue()), status="running",
                  arrays="build/gpu_results/hhvv_gpu_scoped.npz")
    return report, buffer.getvalue(), reference


@pytest.mark.parametrize("stage", capture.STAGES)
def test_complete_stage_audit(stage):
    report, payload, reference = fixture(stage)
    audit = capture.audit_arrays(report, payload, stage, reference)
    assert audit["nll_pass"] and audit["nll_max_absolute_error"] == 0.
    if stage in ("gradient_all_points", "hessian"):
        assert audit["value_gradient_nll_pass"]
    if stage == "hessian":
        assert audit["hessian_max_asymmetry"] == 0.


@pytest.mark.parametrize("fault", ["checksum", "partial_terms", "duplicate_terms", "calls",
                                 "timing", "incomplete_batch", "missing_scope", "wrong_policy",
                                 "reference_point", "recorded_error", "missing_validation"])
def test_reject_inconsistent_complete_gradient(fault):
    report, payload, reference = fixture()
    timing = report["timings"]["value_and_grad"]
    if fault == "checksum":
        report["arrays_sha256"] = "wrong"
    elif fault == "partial_terms":
        timing["term_timings"].pop()
    elif fault == "duplicate_terms":
        timing["term_timings"][1]["name"] = "first"
    elif fault == "calls":
        timing["spooled_timed_calls"][0]["kernel_calls"] -= 1
    elif fault == "timing":
        timing["spooled_timed_calls"][0]["total_seconds"] += 1
    elif fault == "incomplete_batch":
        timing["scoped_load_sessions"]["first"][0]["calls"] = 25
    elif fault == "missing_scope":
        timing["scoped_load_sessions"].pop("first")
    elif fault == "wrong_policy":
        timing["term_timings"][0]["compiler_options"] = {"xla_disable_hlo_passes": "fusion"}
    elif fault == "reference_point":
        reference["cases"][3]["theta"][1] = 2.
    elif fault == "recorded_error":
        report["reference_validation"][3] = dict(nll_absolute_error=1., value_gradient_nll_absolute_error=0.)
    elif fault == "missing_validation":
        report["reference_validation"] = [dict(nll_absolute_error=0.)]*26
    with pytest.raises(AssertionError):
        capture.audit_arrays(report, payload, "gradient_all_points", reference)


def test_reject_nonfinite_hessian():
    report, payload, reference = fixture("hessian", nonfinite=True)
    with pytest.raises(AssertionError, match="hessian"):
        capture.audit_arrays(report, payload, "hessian", reference)


def test_nll_disagreement_is_preserved_without_derivative_acceptance():
    report, payload, reference = fixture("value")
    reference["cases"][0]["combine_nll"] = 3.
    report["reference_validation"][0]["nll_absolute_error"] = 1.
    result = capture.audit_arrays(report, payload, "value", reference)
    assert result == {"nll_pass": False, "nll_max_absolute_error": 1.}


def test_capture_preserves_raw_arrays_and_refuses_overwrite(tmp_path, monkeypatch):
    report, payload, reference = fixture()
    monkeypatch.setattr(capture, "check_report", lambda r, b: reference)
    # Source provenance is recorded from the real checkout, before entering scratch.
    script = Path(capture.__file__).read_bytes()
    transport = Path(capture.transport.__file__).read_bytes()
    monkeypatch.chdir(tmp_path)
    Path("scripts").mkdir()
    for name, data in (("capture_scoped_gpu_results.py", script), ("capture_spooled_gpu_gradient.py", transport)):
        Path("scripts", name).write_bytes(data)
    monkeypatch.setattr(capture, "__file__", str(tmp_path/"scripts/capture_scoped_gpu_results.py"))
    monkeypatch.setattr(capture.transport, "__file__", str(tmp_path/"scripts/capture_spooled_gpu_gradient.py"))
    Path("reports").mkdir()
    Path("build/gpu_results").mkdir(parents=True)
    raw = json.dumps(report).encode()
    result = capture.capture(raw, payload, "gpu", "gradient_all_points")
    saved = json.loads(Path(result["report"]).read_bytes())
    assert saved["status"] == "checkpoint" and saved["checkpoint_parent_status"] == "running"
    assert Path(saved["checkpoint_raw"]).read_bytes() == raw
    assert Path(saved["arrays"]).read_bytes() == payload
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        capture.capture(raw, payload, "gpu", "gradient_all_points")
    assert before == {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}


def test_report_array_race_never_writes_checkpoint(monkeypatch):
    report, payload, reference = fixture()
    report["arrays_sha256"] = "new arrays not yet paired with report"
    monkeypatch.setattr(capture, "check_report", lambda r, b: reference)
    responses = iter([json.dumps(report).encode(), payload])
    monkeypatch.setattr(capture.transport, "fetch", lambda p: next(responses))
    monkeypatch.setattr(capture, "capture", lambda *a: pytest.fail("Must not capture mismatched bytes"))
    state = dict(checkpoints=[])
    assert not capture.poll_once("gpu", ["gradient_all_points"], state)
    assert not state["checkpoints"] and "race" in state["last_retry"]


@pytest.mark.parametrize("key,value", [("source_sha256", "wrong"), ("backend", "cpu"),
    ("float64", False), ("scoped_executable_sha256", "wrong"),
    ("hessian_compiler_options", {"xla_disable_hlo_passes": "fusion"}),
    ("reference_manifest_sha256", "wrong")])
def test_reject_mixed_provenance(key, value, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("src/combine_jax").mkdir(parents=True)
    Path("src/combine_jax/synthetic.py").write_text("# synthetic source\n")
    Path("scripts").mkdir()
    for name in ("gpu_benchmark.py", "spooled_executable.py", "scoped_executable.py"):
        Path("scripts", name).write_text("# synthetic helper\n")
    capture.MANIFEST.parent.mkdir(parents=True)
    manifest = dict(combine_version="11.0.0", root_version="6.40.00",
                    models=dict(hhvv=dict(source={"synthetic": True},
                                         reference=dict(parameter_names=["synthetic"]))))
    capture.MANIFEST.write_text(json.dumps(manifest))
    report = dict(model="hhvv", backend="gpu", host="lxplus905.cern.ch", workdir="/tmp/sqian/combine-jax",
        devices=["cuda:0"], float64=True, jax_version="0.10.1", jaxlib_version="0.10.1",
        lcg_view="/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh",
        parameters=2656, terms=524, report_tag="scoped", scoped_loads=True,
        spool_value_gradient=True, spool_hessian_executables=True, cpu_hessian_no_fusion=True,
        trim_compilation_cache=True, release_stages=True, hessian_chunk_size=4, constraint_batch_size=32,
        hessian_compiler_options={}, arrays="build/gpu_results/hhvv_gpu_scoped.npz",
        source_sha256=capture.transport.digest(b"src/combine_jax/synthetic.py\0# synthetic source\n"),
        export_stack=dict(combine_version="11.0.0", root_version="6.40.00"), source={"synthetic": True},
        **{k: capture.transport.digest(Path(p).read_bytes()) for k, p in {
            "benchmark_sha256": "scripts/gpu_benchmark.py",
            "spooled_executable_sha256": "scripts/spooled_executable.py",
            "scoped_executable_sha256": "scripts/scoped_executable.py",
            "reference_manifest_sha256": str(capture.MANIFEST)}.items()})
    assert capture.check_report(report, "gpu")["parameter_names"]
    report[key] = value
    with pytest.raises(AssertionError):
        capture.check_report(report, "gpu")
