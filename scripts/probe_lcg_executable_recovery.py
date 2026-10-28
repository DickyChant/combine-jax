"""Test fresh-process recovery of trusted saved GPU Hessians, without ROOT."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--export-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
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
    from gpu_benchmark import MemoryMonitor
    assert digest(checked_path(args.producer)) == args.producer_sha256
    producer = json.loads(args.producer.read_text())
    runtime = checked_runtime(producer)
    assert producer["model"] in ("counting", "automcstats"), "Smoke recovery first"
    manifest_path = checked_path(producer["reference_manifest"])
    assert digest(manifest_path) == producer["reference_manifest_sha256"]
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"][producer["model"]]
    assert model["source"] == producer["source"]
    export = checked_path(Path("build/gpu_exports")/model["cache"])
    assert digest(export) == args.export_sha256
    artifact = Path("build/gpu_results")/(args.output.stem+".npz")
    if artifact.exists():
        raise FileExistsError(artifact)
    started = time.perf_counter()
    monitor = MemoryMonitor()
    report = dict(status="running", model=producer["model"], backend="gpu",
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid(),
        producer=str(args.producer), producer_sha256=args.producer_sha256,
        export_sha256=args.export_sha256, reference_manifest=str(manifest_path),
        reference_manifest_sha256=digest(manifest_path), source=model["source"],
        runtime=runtime, runner_sha256=digest(__file__),
        recovery_helper_sha256=digest("scripts/trusted_lcg_recovery.py"), terms=[],
        compile_calls=0, scope="Fresh-process recovery on the same exact LCG GPU runtime. "
        "Every smoke Hessian entry at all three saved Hessian reference points. "
        "No cross-version portability or full HHVV performance claim.")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            gpu_memory=monitor.snapshot(), root_imported="ROOT" in sys.modules)
        temp = args.output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temp.replace(args.output)

    try:
        save()
        likelihood = load_export(export, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        cases = [c for c in model["reference"]["cases"] if "hessian" in c]
        assert len(cases) == 3
        points = np.asarray([c["theta"] for c in cases], dtype=np.float64)
        matrices = np.zeros((len(cases), len(points[0]), len(points[0])))
        saved_records = records(producer)
        assert len(saved_records) == len(evaluator.terms)
        for index, (term, record) in enumerate(zip(evaluator.terms, saved_records, strict=True)):
            checked_record(record, term, index, producer)
            if not len(term.indices):
                report["terms"].append(dict(index=index, name=term.name, skipped_empty=True))
                continue
            recovered = RecoveredExecutable(record, term, likelihood.initial)
            start = time.perf_counter()
            with recovered as kernel:
                for n, point in enumerate(points):
                    matrices[n][np.ix_(term.indices, term.indices)] += all_columns(kernel, point[term.indices])
            report["terms"].append(dict(index=index, name=term.name,
                executable_sha256=record["executable_sha256"],
                lowering_seconds=recovered.lowering_seconds,
                recovered_three_point_batch_seconds=time.perf_counter()-start,
                loads=recovered.loads))
            save()
        scales = np.asarray(model["reference"]["parameter_scales"])
        expected = np.asarray([c["hessian"] for c in cases])
        errors = abs(matrices-expected)*np.outer(scales, scales)
        assert np.isfinite(matrices).all() and matrices.dtype == np.float64
        assert float(errors.max()) < 1e-6
        assert digest(producer["arrays"]) == producer["arrays_sha256"]
        with np.load(producer["arrays"], allow_pickle=False) as prior:
            np.testing.assert_array_equal(prior["point"], points[0])
            np.testing.assert_array_equal(prior["parameter_names"], likelihood.parameter_names)
            np.testing.assert_array_equal(matrices[0], prior["hessian"])
        np.savez_compressed(artifact, points=points, hessians=matrices,
            parameter_names=np.asarray(likelihood.parameter_names), parameter_scales=scales)
        assert checked_runtime(producer) == runtime
        assert digest(args.producer) == args.producer_sha256 and digest(export) == args.export_sha256
        report.update(status="passed", arrays=str(artifact), arrays_sha256=digest(artifact),
            reference_max_scaled_error=float(errors.max()), prior_nominal_bitwise_equal=True,
            compared_entries=int(matrices.size), parameters=len(points[0]))
        save()
        print(report["status"], report["reference_max_scaled_error"], flush=True)
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
