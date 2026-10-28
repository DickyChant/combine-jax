"""Compare complete benchmark Hessians with the saved local CPU block reference.

Read-only numerical validation, including independent reassembly of every local
block. This neither imports ROOT/JAX nor compares timings across the two hosts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np


OPTIONS = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}
TOLERANCE = 1e-6


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_evidence(path, checksum, revision=None):
    """Verify current source or an explicitly selected, exact historical blob."""
    path = Path(path)
    relative = str(path.resolve().relative_to(Path.cwd().resolve()))
    current = digest(path)
    result = dict(path=relative, reported_sha256=checksum, current_sha256=current)
    if current == checksum:
        return dict(result, verified_from="working_tree")
    if revision is None:
        raise ValueError(f"Changed producer source needs its explicit historical revision: {relative}")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Use the full historical Git commit ID")
    payload = subprocess.check_output(["git", "show", f"{revision}:{relative}"])
    assert hashlib.sha256(payload).hexdigest() == checksum, "Historical source hash differs"
    return dict(result, verified_from="git_blob", revision=revision)


def load_arrays(report):
    assert digest(report["arrays"]) == report["arrays_sha256"], "Numerical artifact changed"
    with np.load(report["arrays"], allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in ("point", "parameter_names", "parameter_scales", "hessian")}
    size = report["parameters"]
    assert arrays["parameter_names"].shape == (size,)
    assert len(set(arrays["parameter_names"])) == size
    for key, shape in (("point", (size,)), ("parameter_scales", (size,)), ("hessian", (size, size))):
        assert arrays[key].shape == shape and arrays[key].dtype == np.float64, key
        assert np.isfinite(arrays[key]).all(), key
    assert np.all(arrays["parameter_scales"] > 0)
    return arrays


def reassemble_local(report, arrays):
    """Reproduce the saved full matrix from all hash-verified blocks in order."""
    records = report["terms"]
    assert len(records) == report["expected_terms"]
    assert len({record["name"] for record in records}) == len(records)
    restored = np.zeros_like(arrays["hessian"])
    size = len(arrays["point"])
    for index, record in enumerate(records):
        assert record["index"] == index
        assert digest(record["arrays"]) == record["arrays_sha256"], "Local block changed"
        with np.load(record["arrays"], allow_pickle=False) as block:
            indices = block["indices"]
            width = record["parameters"]
            assert indices.shape == (width,) and np.issubdtype(indices.dtype, np.integer)
            assert len(set(indices)) == width and np.all((indices >= 0) & (indices < size))
            np.testing.assert_array_equal(block["point"], arrays["point"][indices])
            matrix = block["hessian"]
            assert matrix.shape == (width, width) and matrix.dtype == np.float64
            assert np.isfinite(matrix).all()
            restored[np.ix_(indices, indices)] += matrix
    np.testing.assert_array_equal(restored, arrays["hessian"])
    return dict(terms=len(records), block_hashes_verified=True, reassembly_bitwise_equal=True)


def matrix_errors(reference, candidate):
    for key in ("parameter_names", "point", "parameter_scales"):
        np.testing.assert_array_equal(candidate[key], reference[key], err_msg=key)
    left, right = candidate["hessian"], reference["hessian"]
    assert left.shape == right.shape and left.ndim == 2
    assert left.dtype == right.dtype == np.float64
    assert np.isfinite(left).all() and np.isfinite(right).all()
    scales = reference["parameter_scales"]
    assert scales.shape == (len(right),) and np.all(scales > 0)
    difference = abs(left-right)
    scaled = difference*np.outer(scales, scales)
    worst = np.unravel_index(np.argmax(scaled), scaled.shape) if scaled.size else None
    return dict(
        compared_entries=int(right.size), differing_entries=int(np.count_nonzero(left != right)),
        maximum_absolute_error=float(difference.max(initial=0.)),
        maximum_relative_error=float((difference/np.maximum(abs(right), 1e-8)).max(initial=0.)),
        maximum_scaled_absolute_error=float(scaled.max(initial=0.)),
        worst_scaled_entry=None if worst is None else dict(
            indices=[int(i) for i in worst],
            parameters=[str(reference["parameter_names"][i]) for i in worst],
            reference=float(right[worst]), candidate=float(left[worst])),
        reference_maximum_asymmetry=float(abs(right-right.T).max(initial=0.)),
        candidate_maximum_asymmetry=float(abs(left-left.T).max(initial=0.)),
        passed=bool(scaled.max(initial=0.) < TOLERANCE))


def compare(local_path, benchmark_path, *, local_builder_revision=None):
    local = json.loads(local_path.read_bytes())
    benchmark = json.loads(benchmark_path.read_bytes())
    assert local["status"] in ("computed", "passed") and local["backend"] == "cpu"
    assert benchmark["status"] in ("computed", "passed", "checkpoint")
    if benchmark["status"] == "checkpoint":
        assert benchmark["checkpoint_stage"] == "hessian"
        assert digest(benchmark["checkpoint_raw"]) == benchmark["checkpoint_parent_report_sha256"]
    assert all("hessian" in report["completed_operations"] for report in (local, benchmark))
    for key in ("model", "source", "source_sha256", "export_stack", "parameters",
                "constraint_batch_size", "hessian_chunk_size"):
        assert local[key] == benchmark[key], key
    current_core = hashlib.sha256(b"".join(str(path).encode()+b"\0"+path.read_bytes()
        for path in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()
    assert local["source_sha256"] == current_core
    assert local["float64"] and benchmark["float64"]
    assert local["hessian_compiler_options"] == OPTIONS
    assert benchmark["hessian_compiler_options"] == (OPTIONS if benchmark["backend"] == "cpu" else {})
    assert benchmark["backend"] in ("cpu", "gpu") and benchmark["report_tag"] == "scoped"
    assert benchmark["host"] == "lxplus905.cern.ch"
    assert benchmark["workdir"] == "/tmp/sqian/combine-jax"
    assert benchmark["devices"] == (["cuda:0"] if benchmark["backend"] == "gpu" else ["cpu:0"])
    assert benchmark["jax_version"] == benchmark["jaxlib_version"] == "0.10.1"
    assert benchmark["lcg_view"] == "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"
    producer_sources = [
        source_evidence(path, checksum,
                        local_builder_revision if Path(path).name == "build_cpu_hessian_reference.py" else None)
        for path, checksum in local["runner_sha256"].items()]
    for key, path in (("benchmark_sha256", "scripts/gpu_benchmark.py"),
                      ("scoped_executable_sha256", "scripts/scoped_executable.py"),
                      ("spooled_executable_sha256", "scripts/spooled_executable.py")):
        assert benchmark[key] == digest(path), key
    assert digest(local["validation_gate"]) == local["validation_gate_sha256"]
    if local.get("nominal_fd_refinement"):
        refinement = local["nominal_fd_refinement"]
        assert digest(refinement["report"]) == refinement["report_sha256"]
    manifest = json.loads(Path(local["manifest"]).read_bytes())
    assert digest(local["manifest"]) == local["manifest_sha256"] == benchmark["reference_manifest_sha256"]
    assert digest(benchmark["reference_manifest"]) == local["manifest_sha256"]
    model = manifest["models"][local["model"]]
    assert local["source"] == model["source"]
    assert digest(Path("build/gpu_exports")/model["cache"]) == local["export_sha256"]
    left, right = load_arrays(local), load_arrays(benchmark)
    np.testing.assert_array_equal(left["point"], local["point"])
    np.testing.assert_array_equal(left["point"], model["reference"]["cases"][0]["theta"])
    if "parameter_names" in model["reference"]:
        np.testing.assert_array_equal(left["parameter_names"], model["reference"]["parameter_names"])
    assembly = reassemble_local(local, left)
    assert local["expected_terms"] == benchmark["terms"]
    records = benchmark["timings"]["hessian"]["term_timings"]
    assert [(r["name"], r["parameters"]) for r in local["terms"]] == [
        (r["name"], r["parameters"]) for r in records]
    assert benchmark["preparation_timings"]["hessian"]["completed_terms"] == benchmark["terms"]
    errors = matrix_errors(left, right)
    metadata = {}
    for name, report in (("local", local), ("benchmark", benchmark)):
        metadata[name] = {key: report.get(key) for key in (
            "host", "backend", "jax_version", "jaxlib_version", "float64", "cpu_affinity",
            "lcg_view", "gpu_metadata", "hessian_compiler_options")}
    return dict(status="passed" if errors["passed"] else "failed", model=local["model"],
        local_report=str(local_path), local_report_sha256=digest(local_path),
        benchmark_report=str(benchmark_path), benchmark_report_sha256=digest(benchmark_path),
        local_arrays_sha256=local["arrays_sha256"], benchmark_arrays_sha256=benchmark["arrays_sha256"],
        source=local["source"], source_sha256=local["source_sha256"],
        checker_sha256=digest(__file__), metadata=metadata, local_producer_sources=producer_sources,
        local_reassembly=assembly, errors=errors,
        scaled_absolute_tolerance=TOLERANCE,
        scaled_error_definition="abs(candidate_ij-reference_ij)*parameter_scale_i*parameter_scale_j",
        scope="Every nominal Hessian entry, including fixed-bound rows and columns, compared across local JAX 0.12 and the pinned LCG runtime. Numerical agreement only: no cross-host timing ratio, fit covariance or independent Combine curvature acceptance.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-reference", type=Path, required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-builder-revision",
                        help="Explicit full commit ID only when the saved local builder predates the current file")
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use a fresh reports/*.json output")
    if args.output.exists():
        raise FileExistsError("Preserve prior comparison evidence")
    result = compare(args.local_reference, args.benchmark, local_builder_revision=args.local_builder_revision)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(args.output, result["status"], result["errors"])
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
