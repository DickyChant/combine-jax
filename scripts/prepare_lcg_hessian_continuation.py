"""Checkpoint the remaining HHVV GPU Hessian compilation after trusted recovery.

Original reports and binaries remain untouched. Every newly completed term is
durably recorded separately; this stage does not claim a complete Hessian call.
"""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import signal
import sys
import threading
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rss-limit-mib", type=float, default=11000)
    parser.add_argument("--wall-seconds", type=float, default=21600)
    args = parser.parse_args()
    if not 0 < args.rss_limit_mib <= 11000 or not 0 < args.wall_seconds <= 21600:
        parser.error("Require positive RSS <=11000 MiB and wall <=21600 seconds")
    directory = Path("build/gpu_executables")/args.output.stem/"hessian"
    if args.output.exists() or directory.exists():
        raise FileExistsError("Preserve existing continuation evidence")
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    os.environ.update(JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true",
        JAX_ENABLE_COMPILATION_CACHE="false", JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="false", OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import jax
    print("GPU_FIRST", jax.devices(), "float64", jax.config.jax_enable_x64, flush=True)
    assert all(d.platform == "gpu" for d in jax.devices()) and jax.config.jax_enable_x64
    import numpy as np
    from trusted_lcg_recovery import (digest, checked_path, checked_runtime, records,
        checked_record, RecoveredExecutable, all_columns)
    from hhvv_export_cache import load_export
    from spooled_executable import SpooledExecutable
    from combine_jax.local import release_compilation_cache
    from gpu_benchmark import MemoryMonitor, process_mapping_count
    assert digest(checked_path(args.plan)) == args.plan_sha256
    plan = json.loads(args.plan.read_text())
    parent_path = checked_path(plan["producer"])
    assert digest(parent_path) == plan["producer_sha256"]
    producer = json.loads(parent_path.read_text())
    runtime = checked_runtime(producer)
    assert producer["model"] == "hhvv"
    for path, checksum in plan["frozen_sources"].items():
        assert digest(checked_path(path)) == checksum, path
    for entry in plan["smoke_gates"]:
        assert digest(checked_path(entry["report"])) == entry["sha256"]
        smoke = json.loads(Path(entry["report"]).read_text())
        assert smoke["status"] == "passed" and smoke["prior_nominal_bitwise_equal"]
        assert smoke["recovery_helper_sha256"] == digest("scripts/trusted_lcg_recovery.py")
        assert smoke["runtime"] == runtime
        assert digest(checked_path(smoke["arrays"])) == smoke["arrays_sha256"]
    manifest_path = checked_path(producer["reference_manifest"])
    assert digest(manifest_path) == producer["reference_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"]["hhvv"]
    assert model["source"] == producer["source"]
    export = checked_path(Path("build/gpu_exports")/model["cache"])
    assert digest(export) == plan["export_sha256"]
    started = time.perf_counter()
    monitor = MemoryMonitor()
    report = dict(status="running", stage="setup", model="hhvv", backend="gpu",
        pid=os.getpid(), process_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        started_utc=datetime.now(timezone.utc).isoformat(), runtime=runtime,
        plan=str(args.plan), plan_sha256=args.plan_sha256,
        producer=str(parent_path), producer_sha256=plan["producer_sha256"],
        source=model["source"], source_sha256=runtime["source_sha256"],
        export_sha256=plan["export_sha256"], reference_manifest=str(manifest_path),
        reference_manifest_sha256=digest(manifest_path),
        runner_sha256=digest(__file__), frozen_sources=plan["frozen_sources"],
        rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        mapping_guard=int(Path("/proc/sys/vm/max_map_count").read_text())-4096,
        executable_directory=str(directory), recovered_reference_checks=[],
        recovered_terms=[], new_terms=[],
        scope="Preparation continuation only. Reuses recorded trusted original terms; "
        "records new lower/compile/serialization times per term. No full Hessian "
        "execution latency, full-model curvature or matched CPU acceptance.")
    stop = threading.Event()
    lock = threading.RLock()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                gpu_memory=monitor.snapshot(), mapping_count=process_mapping_count(),
                root_imported="ROOT" in sys.modules)
            temp = args.output.with_suffix(".json.tmp")
            temp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temp.replace(args.output)

    def watch():
        last = time.perf_counter()
        while not stop.wait(.25):
            elapsed = time.perf_counter()-started
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            mappings = process_mapping_count()
            if rss > args.rss_limit_mib or elapsed > args.wall_seconds or mappings > report["mapping_guard"]:
                report.update(status="resource_limit",
                    exception="Own RSS, wall or mapping guard exceeded")
                save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], elapsed, rss, mappings, flush=True)
                last = time.perf_counter()

    def interrupted(signum, frame):
        report.update(status="interrupted", signal=signal.Signals(signum).name)
        save()
        raise SystemExit(128+signum)

    previous_handlers = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        point = likelihood.initial
        prior = records(producer)
        assert len(prior) == plan["recovered_terms"] == 90
        assert len(evaluator.terms) == producer["terms"] == 524
        for index, record in enumerate(prior):
            checked_record(record, evaluator.terms[index], index, producer)
            report["recovered_terms"].append(dict(index=index, **record))
        report.update(parameters=len(point), expected_terms=len(evaluator.terms))
        save()
        for reference in plan["local_block_checks"]:
            index = reference["index"]
            term = evaluator.terms[index]
            block_path = checked_path(reference["arrays"])
            assert digest(block_path) == reference["arrays_sha256"]
            with np.load(block_path, allow_pickle=False) as block:
                np.testing.assert_array_equal(block["indices"], term.indices)
                np.testing.assert_array_equal(block["point"], point[term.indices])
                expected = block["hessian"].copy()
            report["stage"] = f"recover and validate original Hessian term {index}"
            save()
            recovered = RecoveredExecutable(prior[index], term, point)
            with recovered as kernel:
                actual = all_columns(kernel, point[term.indices])
            scales = np.asarray(reference["parameter_scales"])
            error = float(np.max(abs(actual-expected)*np.outer(scales, scales)))
            assert error < 1e-6
            report["recovered_reference_checks"].append(dict(index=index,
                name=term.name, local_reference=reference, max_scaled_error=error,
                lowering_seconds=recovered.lowering_seconds, loads=recovered.loads,
                status="passed"))
            save()
        directory.mkdir(parents=True)
        for index in range(len(prior), len(evaluator.terms)):
            term = evaluator.terms[index]
            report["stage"] = f"compile Hessian {index+1}/{len(evaluator.terms)} {term.name}"
            monitor.stage = report["stage"]
            save()
            record = dict(index=index, name=term.name, parameters=len(term.indices),
                kernel="hessian_columns", compiler_options={})
            if not len(term.indices):
                record.update(skipped_empty_hessian=True, lowering_seconds=0., compile_seconds=0.)
            else:
                begin = time.perf_counter()
                lowered = term.hessian_columns.lower(
                    point[term.indices], np.zeros((min(4, len(term.indices)), len(term.indices))))
                record["lowering_seconds"] = time.perf_counter()-begin
                begin = time.perf_counter()
                compiled = lowered.compile()
                record["compile_seconds"] = time.perf_counter()-begin
                stored, storage = SpooledExecutable.write(compiled, directory/f"{index}.bin", backend="gpu")
                record.update(executable=str(stored.path), **storage)
                del compiled, lowered, stored
                term.reset_kernel("hessian_columns")
                begin = time.perf_counter()
                release_compilation_cache()
                record["cache_trim_seconds"] = time.perf_counter()-begin
            with lock:
                report["new_terms"].append(record)
                save()
        assert [r["index"] for r in report["recovered_terms"]+report["new_terms"]] == list(range(524))
        assert checked_runtime(producer) == runtime
        for path, checksum in plan["frozen_sources"].items():
            assert digest(path) == checksum, path
        assert digest(args.plan) == args.plan_sha256
        assert digest(parent_path) == plan["producer_sha256"] and digest(export) == plan["export_sha256"]
        report.update(status="prepared", stage="finished",
            new_compile_seconds=sum(r["compile_seconds"] for r in report["new_terms"]),
            new_lowering_seconds=sum(r["lowering_seconds"] for r in report["new_terms"]),
            completed_terms=524, finished_utc=datetime.now(timezone.utc).isoformat())
        save()
        print("prepared", report["elapsed_seconds"], flush=True)
    except BaseException:
        if report["status"] == "running":
            report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()
        monitor.close()
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
