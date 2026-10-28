"""Audit every recovered HHVV GPU Hessian entry against the saved CPU reference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_local_cpu_hessian import (
    OPTIONS, digest, load_arrays, matrix_errors, reassemble_local, source_evidence)
from measure_lcg_recovered_hessian import checked_catalog, compare_hessian_repeat


def verify_saved_repeats(benchmark, reference, first):
    """Recompute each comparison from its saved matrix, including failed bitwise equality."""
    policy = benchmark["repeat_validation"]
    assert policy["policy"] == "every_call_cpu_reference_and_first_matrix"
    assert policy["scaled_absolute_tolerance"] == 1e-6
    assert policy["bitwise_equality_required"] is False
    assert policy["checker_sha256"] == digest("scripts/compare_local_cpu_hessian.py")
    calls = benchmark["calls"]
    assert len({call["arrays"] for call in calls}) == len(calls)
    assert calls[0]["arrays"] == benchmark["arrays"]
    assert calls[0]["arrays_sha256"] == benchmark["arrays_sha256"]
    results = []
    for call in calls:
        candidate = load_arrays(dict(call, parameters=benchmark["parameters"]))
        actual = compare_hessian_repeat(reference, candidate, first["hessian"])
        assert actual == call["validation"], "Stored per-call comparison differs from saved arrays"
        assert actual["passed"], "A saved call fails the unchanged CPU/reference tolerance"
        assert call["bitwise_equal_to_first"] is actual["bitwise_equal_to_first"]
        assert np.isfinite(call["artifact_and_validation_seconds"]) and call["artifact_and_validation_seconds"] >= 0
        results.append(dict(index=call["index"], arrays_sha256=call["arrays_sha256"], **actual))
    return results


def compare(local_path, benchmark_path):
    local = json.loads(local_path.read_text())
    benchmark = json.loads(benchmark_path.read_text())
    assert local["status"] in ("computed", "passed")
    assert benchmark["status"] == "computed" and benchmark["completed_operations"] == ["hessian"]
    prep_path = Path(benchmark["preparation"])
    assert digest(prep_path) == benchmark["preparation_sha256"]
    preparation = json.loads(prep_path.read_text())
    original_path = Path(benchmark["original_producer"])
    assert digest(original_path) == benchmark["original_producer_sha256"] == preparation["producer_sha256"]
    original = json.loads(original_path.read_text())
    assert str(original_path) == preparation["producer"]
    catalog = checked_catalog(preparation, original)
    assert digest(preparation["plan"]) == preparation["plan_sha256"]
    plan = json.loads(Path(preparation["plan"]).read_text())
    assert plan["local_reference"] == str(local_path) and plan["local_reference_sha256"] == digest(local_path)
    assert plan["local_reference_arrays_sha256"] == local["arrays_sha256"]
    assert plan["producer_sha256"] == preparation["producer_sha256"]
    assert plan["frozen_sources"] == preparation["frozen_sources"]
    for path, checksum in preparation["frozen_sources"].items():
        assert digest(path) == checksum
    assert benchmark["runner_sha256"] == digest("scripts/measure_lcg_recovered_hessian.py")
    assert benchmark["recovery_helper_sha256"] == digest("scripts/trusted_lcg_recovery.py")
    assert preparation["runner_sha256"] == digest("scripts/prepare_lcg_hessian_continuation.py")
    for key, value in preparation["runtime"].items():
        assert benchmark[key] == value
    assert benchmark["backend"] == "gpu" and local["backend"] == "cpu"
    assert benchmark["host"] == "lxplus905.cern.ch" and benchmark["workdir"] == "/tmp/sqian/combine-jax"
    assert benchmark["jax_version"] == benchmark["jaxlib_version"] == benchmark["cuda_plugin_version"] == "0.10.1"
    assert benchmark["lcg_view"] == "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"
    assert benchmark["devices"] == ["cuda:0"] and benchmark["float64"] and not benchmark["root_imported"]
    assert local["hessian_compiler_options"] == OPTIONS and benchmark["hessian_compiler_options"] == {}
    for key in ("model", "source", "source_sha256", "export_stack", "parameters",
                "constraint_batch_size", "hessian_chunk_size", "export_sha256"):
        assert local[key] == benchmark[key], key
    source = hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()
    assert source == local["source_sha256"]
    producer_sources = [source_evidence(p, h) for p, h in local["runner_sha256"].items()]
    manifest_path = Path(benchmark["reference_manifest"])
    assert digest(manifest_path) == benchmark["reference_manifest_sha256"] == local["manifest_sha256"]
    model = json.loads(manifest_path.read_text())["models"]["hhvv"]
    assert digest(Path("build/gpu_exports")/model["cache"]) == benchmark["export_sha256"]
    assert model["source"] == benchmark["source"]
    assert digest(local["validation_gate"]) == local["validation_gate_sha256"]
    if local.get("nominal_fd_refinement"):
        refinement = local["nominal_fd_refinement"]
        assert digest(refinement["report"]) == refinement["report_sha256"]
    left, right = load_arrays(local), load_arrays(benchmark)
    np.testing.assert_array_equal(right["point"], model["reference"]["cases"][0]["theta"])
    np.testing.assert_array_equal(right["parameter_names"], model["reference"]["parameter_names"])
    assembly = reassemble_local(local, left)
    assert len(catalog) == benchmark["terms"] == local["expected_terms"] == 524
    assert [(r["name"], r["parameters"]) for r in catalog] == [
        (r["name"], r["parameters"]) for r in local["terms"]]
    assert [r["index"] for r in benchmark["metadata_recovery"]] == list(range(524))
    for record, recovered in zip(catalog, benchmark["metadata_recovery"], strict=True):
        assert record["name"] == recovered["name"]
        if record["parameters"]:
            assert record["executable_sha256"] == recovered["executable_sha256"]
    nonempty = sum(r["parameters"] > 0 for r in catalog)
    kernel_calls = sum((r["parameters"]+3)//4 for r in catalog)
    assert len(benchmark["calls"]) >= 2
    assert [c["index"] for c in benchmark["calls"]] == list(range(len(benchmark["calls"])))
    repeat_checks = None
    if "repeat_validation" in benchmark:
        policy = benchmark["repeat_validation"]
        assert policy["reference"] == str(local_path)
        assert policy["reference_sha256"] == digest(local_path)
        assert policy["reference_arrays_sha256"] == local["arrays_sha256"]
        repeat_checks = verify_saved_repeats(benchmark, left, right)
    for call in benchmark["calls"]:
        assert call["term_loads"] == nonempty and call["kernel_calls"] == kernel_calls
        assert call["seconds"] > 0 and call["gpu_memory_after"]["error"] is None
        assert all(np.isfinite(v) and v >= 0 for v in call["component_totals"].values())
        if call["index"] and repeat_checks is None:
            assert call["bitwise_equal_to_first"]
    assert benchmark["first_execution_seconds"] == benchmark["calls"][0]["seconds"]
    assert benchmark["warm_seconds"] == [c["seconds"] for c in benchmark["calls"][1:]]
    assert benchmark["median_warm_seconds"] == float(np.median(benchmark["warm_seconds"]))
    for part, records in (("original_90", catalog[:90]), ("continued_434", catalog[90:])):
        for key, value in benchmark["recorded_compilation"][part].items():
            assert value == sum(r.get(key, 0.) for r in records)
    errors = matrix_errors(left, right)
    return dict(status="passed" if errors["passed"] else "failed", model="hhvv",
        benchmark=str(benchmark_path), benchmark_sha256=digest(benchmark_path),
        arrays_sha256=benchmark["arrays_sha256"], preparation=str(prep_path),
        preparation_sha256=digest(prep_path), local_reference=str(local_path),
        local_reference_sha256=digest(local_path), local_arrays_sha256=local["arrays_sha256"],
        local_producer_sources=producer_sources, local_reassembly=assembly,
        checker_sha256=digest(__file__), errors=errors, scaled_absolute_tolerance=1e-6,
        call_coverage=dict(full_calls=len(benchmark["calls"]), term_loads_per_call=nonempty,
                           column_batches_per_call=kernel_calls),
        first_execution_seconds=benchmark["first_execution_seconds"],
        warm_seconds=benchmark["warm_seconds"], median_warm_seconds=benchmark["median_warm_seconds"],
        recorded_compilation=benchmark["recorded_compilation"],
        metadata_recovery_seconds=benchmark["metadata_recovery_seconds"],
        gpu_memory=benchmark["gpu_memory"], peak_rss_mib=benchmark["peak_rss_mib"],
        independently_recomputed_call_comparisons=repeat_checks,
        runtime=preparation["runtime"],
        scope="Every nominal Hessian entry, including fixed rows/columns, versus the "
        "independently reassembled local CPU reference. No cross-host speed ratio, "
        "fit covariance or independent Combine curvature acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-reference", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = compare(args.local_reference, args.benchmark)
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(report["status"], report["errors"], flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
