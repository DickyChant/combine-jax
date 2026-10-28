"""Stage acceptance and provenance checks using a two-parameter polynomial."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_gpu_gradient_checkpoints import compare, random_validation


@pytest.mark.parametrize("change", ("none", "gradient", "nll", "point", "hash", "stage", "manifest",
                                    "nonfinite", "reference", "missing_hhvv_reference",
                                    "spool_hessian_executables", "spool_value_gradient", "hessian_chunk_size", "spooled_executable_sha256"))
def test_stage_acceptance_without_hessian(tmp_path, change):
    points = np.random.default_rng(512).normal(size=(26, 2))
    x,y = points.T
    values = x*x+3*x*y+2*y*y
    gradients = np.column_stack((2*x+3*y, 3*x+4*y))
    timing = dict(lowering_seconds=1., compile_seconds=2., median_warm_seconds=.2,
                  warm_seconds=[.1,.2,.3], release_seconds=.1)
    paths = []
    for backend in ("cpu", "gpu"):
        data = dict(parameter_names=np.array(["x", "y"]), point=points[0].copy(), value=values[0],
                    vg_value=values[0], gradient=gradients[0].copy(), validation_points=points.copy(),
                    validation_labels=np.array([str(i) for i in range(26)]),
                    validation_values=values.copy(), validation_vg_values=values.copy(),
                    validation_gradients=gradients.copy())
        report = dict(model="counting", source={"synthetic_polynomial": True}, source_sha256="source",
                      benchmark_sha256="runner", host="test", lcg_view="test", jax_version="test",
                      jaxlib_version="test", float64=True, cpu_affinity=[0], cpu_model="test",
                      gpu_metadata="test", constraint_batch_size=1, release_stages=True,
                      trim_compilation_cache=True, xla_flags="", effective_xla_options={},
                      reference_manifest_sha256="manifest", parameters=2, channels=1, terms=1,
                      backend=backend, status="resource_limit", completed_operations=["value", "value_and_grad"],
                      nll=values[0], value_gradient_nll=values[0],
                      timings={op:timing for op in ("value", "value_and_grad")},
                      reference_validation=[dict(nll_absolute_error=0., value_gradient_nll_absolute_error=0.,
                                                 gradient_max_relative_error=0.) for _ in range(26)])
        if change == "missing_hhvv_reference":
            report["model"] = "hhvv"
        if backend == "gpu":
            if change == "gradient":
                data["validation_gradients"][8,1] *= 1.01
            elif change == "nll":
                data["validation_values"][4] += 2e-5
            elif change == "point":
                data["validation_points"][4,1] += .1
            elif change == "stage":
                report["completed_operations"] = ["value"]
            elif change == "manifest":
                report["reference_manifest_sha256"] = "different"
            elif change == "nonfinite":
                data["validation_gradients"][8,1] = np.nan
            elif change == "reference":
                report["reference_validation"][7]["gradient_max_relative_error"] = .01
            elif change in ("spool_hessian_executables", "spool_value_gradient", "hessian_chunk_size", "spooled_executable_sha256"):
                report[change] = {"spool_hessian_executables":True, "spool_value_gradient":True, "hessian_chunk_size":3,
                                  "spooled_executable_sha256":"different helper"}[change]
        artifact = tmp_path/(backend+".npz")
        np.savez(artifact, **data)
        report.update(arrays=str(artifact), arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        if backend == "gpu" and change == "hash":
            report["arrays_sha256"] = "incorrect"
        path = tmp_path/(backend+".json")
        path.write_text(json.dumps(report))
        paths.append(path)
    if change in ("point", "hash", "stage", "manifest", "nonfinite", "spool_hessian_executables",
                  "spool_value_gradient", "hessian_chunk_size", "spooled_executable_sha256"):
        with pytest.raises(AssertionError):
            compare(*paths)
    elif change == "missing_hhvv_reference":
        with pytest.raises(FileNotFoundError):
            compare(*paths)
    else:
        result = compare(*paths)
        assert result["status"] == ("passed" if change == "none" else "failed")
        assert set(result["timings"]) == {"value", "value_and_grad"}
        assert all(p["parent_status"] == "resource_limit" for p in result["provenance"].values())


def test_random_reference_identity_and_uncertainty(tmp_path):
    path = tmp_path/"benchmark.json"
    reference = tmp_path/"reference.json"
    reference.write_text("{}")
    checksum = hashlib.sha256(reference.read_bytes()).hexdigest()
    benchmark = dict(source="test", arrays_sha256="arrays")
    check = dict(source="test", benchmark_sha256="benchmark", arrays_sha256="arrays",
                 reference=str(reference), reference_sha256=checksum,
                 stable_reference=str(reference), stable_reference_sha256=checksum,
                 points=[{} for _ in range(25)], status="reference_uncertain", status_counts={"reference_uncertain":1})
    output = tmp_path/"benchmark_combine_random_gradients_stable.json"
    output.write_text(json.dumps(check))
    assert random_validation(path, benchmark, "benchmark")["status"] == "reference_uncertain"
    with pytest.raises(AssertionError):
        random_validation(path, benchmark, "changed benchmark")
    reference.write_text('{"changed":true}')
    with pytest.raises(AssertionError):
        random_validation(path, benchmark, "benchmark")


@pytest.mark.parametrize("change", ("none", "counts", "missing_case", "method", "nll",
                                    "reference_hash", "uncertain", "failed"))
def test_refined_evidence_is_explicit_and_bound_to_benchmark(tmp_path, change):
    original = Path("reports/gpu_hhvv_gpu_cachetrim_gradient_checkpoint.json")
    benchmark = json.loads(original.read_bytes())
    checksum = hashlib.sha256(original.read_bytes()).hexdigest()
    check = json.loads(original.with_name(original.stem+"_combine_random_gradients_refined.json").read_text())
    point = next(p for p in check["points"] if p["refined_checks"])
    if change == "counts":
        point["status_counts"]["passed"] -= 1
    elif change == "missing_case":
        point["refined_checks"].pop()
    elif change == "method":
        point["refined_checks"][0]["reference_method"] = "different estimator"
    elif change == "nll":
        point["nll_absolute_errors"]["validation_values"] = .001
    elif change == "reference_hash":
        key = next(iter(check["refinements"]["documents_sha256"]))
        check["refinements"]["documents_sha256"][key] = "changed"
    elif change in ("uncertain", "failed"):
        check["status"] = "reference_uncertain" if change == "uncertain" else "failed"
    path = tmp_path/"benchmark.json"
    output = tmp_path/"benchmark_combine_random_gradients_refined.json"
    output.write_text(json.dumps(check))
    # Supplying only a refined report cannot silently change the default policy.
    with pytest.raises(FileNotFoundError):
        random_validation(path, benchmark, checksum)
    if change in ("counts", "missing_case", "method", "nll", "reference_hash"):
        with pytest.raises(AssertionError):
            random_validation(path, benchmark, checksum, refined=True)
    else:
        result = random_validation(path, benchmark, checksum, refined=True)
        assert result["status"] == check["status"]
        assert len(result["refinement_documents_sha256"]) == 12


def test_original_failed_evidence_remains_available():
    path = Path("reports/gpu_hhvv_gpu_cachetrim_gradient_checkpoint.json")
    raw = path.read_bytes()
    result = random_validation(path, json.loads(raw), hashlib.sha256(raw).hexdigest())
    assert result["status"] == "failed" and result["status_counts"]["failed"] == 14
    assert result["refinement_documents_sha256"] is None
