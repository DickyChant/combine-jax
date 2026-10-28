"""Measure complete GPU Hessian calls from a verified preparation catalog."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import threading
import time
import traceback


def checked_catalog(preparation, producer):
    """Require exactly the recorded prefix followed by every remaining term."""
    assert preparation["status"] == "prepared"
    old = preparation["recovered_terms"]
    new = preparation["new_terms"]
    assert len(old) == 90 and len(new) == 434
    catalog = old+new
    assert [r["index"] for r in catalog] == list(range(524))
    assert preparation["completed_terms"] == preparation["expected_terms"] == producer["terms"] == 524
    assert all(c["status"] == "passed" and c["max_scaled_error"] < 1e-6
               for c in preparation["recovered_reference_checks"])
    assert [c["index"] for c in preparation["recovered_reference_checks"]] == [0, 2]
    assert producer["compile_progress"]["operation"] == "hessian"
    assert old == [dict(index=i, **r)
                   for i, r in enumerate(producer["compile_progress"]["term_timings"])]
    for index, record in enumerate(catalog):
        assert record["compiler_options"] == {} and record["kernel"] == "hessian_columns"
        if index >= 90 and record["parameters"]:
            assert Path(record["executable"]) == Path(preparation["executable_directory"])/f"{index}.bin"
    return catalog


def assemble_hessian(evaluator, stores, point):
    """Use the original scoped benchmark's column and accumulation order."""
    import numpy as np

    matrix = np.zeros((len(point), len(point)), dtype=np.float64)
    for term, stored in zip(evaluator.terms, stores, strict=True):
        size = len(term.indices)
        if not size:
            assert stored is None
            continue
        with stored as kernel:
            chunk = min(4, size)
            for first in range(0, size, chunk):
                width = min(chunk, size-first)
                directions = np.zeros((chunk, size), dtype=np.float64)
                directions[np.arange(width), first+np.arange(width)] = 1.
                columns = kernel(point[term.indices], directions)[:width].T
                matrix[np.ix_(term.indices, term.indices[first:first+width])] += columns
    return matrix


def compare_hessian_repeat(reference, candidate, first_matrix):
    """Keep the existing scaled tolerance and report bitwise variation separately."""
    import numpy as np
    from compare_local_cpu_hessian import matrix_errors

    cpu = matrix_errors(reference, candidate)
    first = matrix_errors(dict(candidate, hessian=first_matrix), candidate)
    return dict(cpu_reference=cpu, first_matrix=first,
        bitwise_equal_to_first=bool(np.array_equal(candidate["hessian"], first_matrix)),
        passed=cpu["passed"] and first["passed"])


def activity():
    return {
        "device": subprocess.check_output(["nvidia-smi",
            "--query-gpu=utilization.gpu,clocks.current.sm,temperature.gpu",
            "--format=csv,noheader"], text=True).strip(),
        "compute_processes": subprocess.check_output(["nvidia-smi",
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader"], text=True).strip(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--preparation-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--reference-checked-repeats", action="store_true",
        help="Save every matrix and check every call against the frozen CPU reference "
             "at scaled absolute tolerance 1e-6; bitwise equality is diagnostic")
    parser.add_argument("--rss-limit-mib", type=float, default=11000)
    parser.add_argument("--wall-seconds", type=float, default=21600)
    args = parser.parse_args()
    if not 1 <= args.repeat <= 20 or not 0 < args.rss_limit_mib <= 11000 or not 0 < args.wall_seconds <= 21600:
        parser.error("Require repeat 1–20, RSS in (0,11000] MiB, wall in (0,21600] s")
    artifact = Path("build/gpu_results")/(args.output.stem+".npz")
    call_artifacts = [artifact]+[
        artifact.with_name(f"{artifact.stem}_call_{i+1}.npz") for i in range(1, args.repeat+1)]
    if args.output.exists() or any(path.exists() for path in call_artifacts):
        raise FileExistsError("Preserve prior full-Hessian measurements")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    os.environ.update(JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true",
        JAX_ENABLE_COMPILATION_CACHE="false", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import jax
    print("GPU_FIRST", jax.devices(), "float64", jax.config.jax_enable_x64, flush=True)
    assert all(d.platform == "gpu" for d in jax.devices()) and jax.config.jax_enable_x64
    import numpy as np
    from trusted_lcg_recovery import digest, checked_path, checked_runtime, checked_record, RecoveredExecutable
    from hhvv_export_cache import load_export
    from gpu_benchmark import MemoryMonitor, process_mapping_count
    assert digest(checked_path(args.preparation)) == args.preparation_sha256
    preparation = json.loads(args.preparation.read_text())
    assert digest(checked_path(preparation["producer"])) == preparation["producer_sha256"]
    producer = json.loads(Path(preparation["producer"]).read_text())
    runtime = checked_runtime(producer)
    assert runtime == preparation["runtime"]
    assert digest(checked_path(preparation["plan"])) == preparation["plan_sha256"]
    plan = json.loads(Path(preparation["plan"]).read_text())
    assert plan["frozen_sources"] == preparation["frozen_sources"]
    for path, checksum in preparation["frozen_sources"].items():
        assert digest(checked_path(path)) == checksum
    catalog = checked_catalog(preparation, producer)
    manifest_path = checked_path(preparation["reference_manifest"])
    assert digest(manifest_path) == preparation["reference_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"]["hhvv"]
    assert model["source"] == preparation["source"] == producer["source"]
    export = checked_path(Path("build/gpu_exports")/model["cache"])
    assert digest(export) == preparation["export_sha256"]
    started = time.perf_counter()
    monitor = MemoryMonitor()
    report = dict(status="running", stage="setup", model="hhvv", backend="gpu",
        **runtime, pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        source=model["source"], export_sha256=preparation["export_sha256"],
        export_stack={k: manifest[k] for k in ("root_version", "combine_version")},
        reference_manifest=str(manifest_path), reference_manifest_sha256=digest(manifest_path),
        preparation=str(args.preparation), preparation_sha256=args.preparation_sha256,
        original_producer=preparation["producer"], original_producer_sha256=preparation["producer_sha256"],
        runner_sha256=digest(__file__), recovery_helper_sha256=digest("scripts/trusted_lcg_recovery.py"),
        gpu_activity_at_start=activity(),
        constraint_batch_size=32, hessian_chunk_size=4, hessian_compiler_options={},
        terms=524, completed_operations=[], metadata_recovery=[], calls=[],
        rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        mapping_guard=int(Path("/proc/sys/vm/max_map_count").read_text())-4096,
        execution="Every full Hessian entry; one binary verification/load per nonempty term "
        "per complete call; all four-column batches synchronized to the host; "
        "executable/cache cleanup and global host accumulation included.",
        preparation_scope="Compilation happened in the original and continuation processes. "
        "Only complete recorded terms contribute to summed compile/lowering times; "
        "unrecorded work lost in the interrupted run is excluded. Metadata reconstruction "
        "in this process is timed separately and excluded from full-call latency.",
        scope="Full nominal GPU Hessian timing and numerical artifact. Requires separate "
        "complete CPU/Combine-reference validation; no fit covariance acceptance.")
    if args.reference_checked_repeats:
        report["repeat_validation"] = dict(
            policy="every_call_cpu_reference_and_first_matrix",
            scaled_absolute_tolerance=1e-6,
            bitwise_equality_required=False,
            checker_sha256=digest("scripts/compare_local_cpu_hessian.py"),
            artifact_and_validation_scope="Per-call artifact writing and CPU-reference "
            "comparison happen after the synchronized full-call timer stops.")
    report["recorded_compilation"] = {
        part: {key: sum(r.get(key, 0.) for r in rows) for key in (
            "lowering_seconds", "compile_seconds", "serialization_seconds",
            "write_seconds", "cache_trim_seconds")}
        for part, rows in (("original_90", catalog[:90]), ("continued_434", catalog[90:]))}
    lock = threading.RLock()
    stop = threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                gpu_memory=monitor.snapshot(), mapping_count=process_mapping_count(),
                root_imported="ROOT" in sys.modules)
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = time.perf_counter()
        while not stop.wait(.25):
            elapsed = time.perf_counter()-started
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if rss > args.rss_limit_mib or elapsed > args.wall_seconds or process_mapping_count() > report["mapping_guard"]:
                report.update(status="resource_limit", exception="Own RSS, wall or mapping guard exceeded")
                save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], elapsed, rss, flush=True)
                last = time.perf_counter()

    def interrupted(signum, frame):
        report.update(status="interrupted", signal=signal.Signals(signum).name)
        save()
        raise SystemExit(128+signum)

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        point = likelihood.initial
        np.testing.assert_array_equal(point, model["reference"]["cases"][0]["theta"])
        np.testing.assert_array_equal(likelihood.parameter_names, model["reference"]["parameter_names"])
        assert len(evaluator.terms) == len(catalog)
        scales = np.asarray([p.error if p.error > 0 else max(abs(p.value), 1.) for p in likelihood.parameters])
        report["parameters"] = len(point)
        reference_arrays = None
        if args.reference_checked_repeats:
            from compare_local_cpu_hessian import load_arrays
            local_path = checked_path(plan["local_reference"])
            assert digest(local_path) == plan["local_reference_sha256"]
            local = json.loads(local_path.read_text())
            assert local["status"] == "computed" and local["backend"] == "cpu"
            assert local["arrays_sha256"] == plan["local_reference_arrays_sha256"]
            for key in ("model", "source", "source_sha256", "parameters", "export_sha256"):
                assert local[key] == report[key], key
            reference_arrays = load_arrays(local)
            report["repeat_validation"].update(
                reference=str(local_path), reference_sha256=digest(local_path),
                reference_arrays_sha256=local["arrays_sha256"])
        stores = []
        metadata_start = time.perf_counter()
        for index, (record, term) in enumerate(zip(catalog, evaluator.terms, strict=True)):
            report["stage"] = f"recover metadata {index+1}/524 {term.name}"
            assert record["name"] == term.name and record["parameters"] == len(term.indices)
            if index < 90:
                checked_record(record, term, index, producer)
            if len(term.indices):
                path = checked_path(record["executable"])
                assert digest(path) == record["executable_sha256"] and path.stat().st_size == record["serialized_bytes"]
                stored = RecoveredExecutable(record, term, point)
                stores.append(stored)
                report["metadata_recovery"].append(dict(index=index, name=term.name,
                    lowering_seconds=stored.lowering_seconds, executable_sha256=record["executable_sha256"]))
            else:
                assert record["skipped_empty_hessian"]
                stores.append(None)
                report["metadata_recovery"].append(dict(index=index, name=term.name, skipped_empty=True))
            if (index+1) % 10 == 0:
                save()
        report["metadata_recovery_seconds"] = time.perf_counter()-metadata_start
        first_matrix = None
        for call_index in range(args.repeat+1):
            report["stage"] = f"full Hessian call {call_index+1}/{args.repeat+1}"
            monitor.stage = report["stage"]
            save()
            start = time.perf_counter()
            matrix = assemble_hessian(evaluator, stores, point)
            seconds = time.perf_counter()-start
            loads = [s.loads[-1] for s in stores if s is not None]
            assert len(loads) == sum(bool(len(t.indices)) for t in evaluator.terms)
            record = dict(index=call_index, seconds=seconds, term_loads=len(loads),
                kernel_calls=sum(c["calls"] for c in loads),
                component_totals={k: sum(c[k] for c in loads) for k in (
                    "read_and_verify_seconds", "deserialize_seconds", "execute_and_copy_seconds", "cleanup_seconds")},
                gpu_memory_after=monitor.snapshot())
            if args.reference_checked_repeats:
                # Persist each complete result and its latency before any numerical
                # assertion; failed comparisons must retain their actual matrices.
                validation_start = time.perf_counter()
                candidate = dict(hessian=matrix, point=point,
                    parameter_names=np.asarray(likelihood.parameter_names), parameter_scales=scales)
                call_artifact = call_artifacts[call_index]
                np.savez_compressed(call_artifact, **candidate)
                record.update(arrays=str(call_artifact), arrays_sha256=digest(call_artifact))
                report["calls"].append(record)
                if first_matrix is None:
                    first_matrix = matrix.copy()
                    report.update(arrays=str(artifact), arrays_sha256=digest(artifact),
                        completed_operations=["hessian"],
                        hessian_max_asymmetry=(float(np.max(abs(matrix-matrix.T)))
                                               if np.isfinite(matrix).all() else None))
                save()
                comparison = compare_hessian_repeat(reference_arrays, candidate, first_matrix)
                record.update(validation=comparison,
                    bitwise_equal_to_first=comparison["bitwise_equal_to_first"],
                    artifact_and_validation_seconds=time.perf_counter()-validation_start)
                save()
                assert comparison["passed"], "Complete Hessian differs from CPU reference or first matrix"
            elif first_matrix is None:
                assert np.isfinite(matrix).all()
                first_matrix = matrix.copy()
                np.savez_compressed(artifact, hessian=matrix, point=point,
                    parameter_names=np.asarray(likelihood.parameter_names), parameter_scales=scales)
                report.update(arrays=str(artifact), arrays_sha256=digest(artifact),
                    completed_operations=["hessian"],
                    hessian_max_asymmetry=float(np.max(abs(matrix-matrix.T))))
            else:
                assert np.isfinite(matrix).all()
                np.testing.assert_array_equal(matrix, first_matrix)
                record["bitwise_equal_to_first"] = True
            if not args.reference_checked_repeats:
                report["calls"].append(record)
            save()
        assert checked_runtime(producer) == runtime
        assert digest(args.preparation) == args.preparation_sha256
        for path, checksum in preparation["frozen_sources"].items():
            assert digest(path) == checksum
        if args.reference_checked_repeats:
            assert digest(local_path) == report["repeat_validation"]["reference_sha256"]
            assert digest(local["arrays"]) == report["repeat_validation"]["reference_arrays_sha256"]
            assert digest("scripts/compare_local_cpu_hessian.py") == report["repeat_validation"]["checker_sha256"]
        warm = [c["seconds"] for c in report["calls"][1:]]
        report.update(status="computed", stage="finished", first_execution_seconds=report["calls"][0]["seconds"],
            warm_seconds=warm, median_warm_seconds=float(np.median(warm)), min_warm_seconds=min(warm),
            gpu_activity_at_end=activity(),
            finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print("computed", report["median_warm_seconds"], flush=True)
    except BaseException:
        if report["status"] == "running":
            report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()
        monitor.close()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
