"""Measure scoped reloads on the largest HHVV term without changing active jobs."""

import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback

import jax
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export
from scoped_executable import LoadedExecutable
from spooled_executable import SpooledExecutable
from validate_cpu_no_fusion_hessian import OPTIONS, all_columns, digest, scaled_error


def main():
    output = Path("reports/gpu_local_hhvv_scoped_reload.json")
    directory = Path("build/gpu_results/local_hhvv_scoped_reload")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior scoped reload evidence")
    directory.mkdir(parents=True)
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[2]])
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    parent_path = Path("reports/hhvv_cpu_no_fusion_hessian.json")
    parent = json.loads(parent_path.read_text())
    source_hash = hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()
    assert parent["source_sha256"] == source_hash and parent["jax_version"] == jax.__version__
    assert parent["hessian_compiler_options"] == OPTIONS
    manifest_path = Path(parent["manifest"])
    assert digest(manifest_path) == parent["manifest_sha256"]
    model = json.loads(manifest_path.read_text())["models"]["hhvv"]
    export = Path("build/gpu_exports")/model["cache"]
    assert digest(export) == parent["export_sha256"]
    started = time.perf_counter()
    report = dict(status="running", stage="setup", source=model["source"],
        source_sha256=source_hash, backend="cpu", jax_version=jax.__version__,
        jaxlib_version=__import__("jaxlib").__version__, float64=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), hessian_compiler_options=OPTIONS,
        parent=str(parent_path), parent_sha256=digest(parent_path),
        runner_sha256={str(p): digest(p) for p in [Path(__file__),
            Path("scripts/scoped_executable.py"), Path("scripts/spooled_executable.py"),
            Path("scripts/validate_cpu_no_fusion_hessian.py")]},
        scope="Largest local term only, with one load per complete local Hessian; numerical/serialization check overlapping full local reference. No full-model, GPU or isolated performance acceptance.",
        rss_limit_mib=3500, wall_limit_seconds=600)
    stop = threading.Event()
    lock = threading.RLock()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def watch():
        last = started
        while not stop.wait(.1):
            elapsed = time.perf_counter()-started
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if rss > report["rss_limit_mib"] or elapsed > report["wall_limit_seconds"]:
                report.update(status="resource_limit", exception="Own RSS or wall-time guard exceeded")
                save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                print(report["stage"], elapsed, rss, flush=True)
                last = time.perf_counter()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        index = max(range(len(evaluator.terms)), key=lambda i: len(evaluator.terms[i].indices))
        term = evaluator.terms[index]
        prior = next(c for c in parent["cases"] if c["index"] == index)
        assert prior["name"] == term.name and digest(prior["arrays"]) == prior["arrays_sha256"]
        with np.load(prior["arrays"], allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["parameter_names"], term.parameter_names)
            points = arrays["points"].copy()
            previous_hessians = arrays["original_hessians"].copy()
            scales = arrays["parameter_scales"].copy()
        report.update(stage="compile largest no-fusion Hessian", term_index=index,
                      term_name=term.name, parameters=len(term.indices), points=len(points))
        save()
        begin = time.perf_counter()
        lowered = term.hessian_columns.lower(points[0], np.zeros((4, len(term.indices))))
        report["lowering_seconds"] = time.perf_counter()-begin
        begin = time.perf_counter()
        compiled = lowered.compile(compiler_options=OPTIONS)
        report["compile_seconds"] = time.perf_counter()-begin
        expected = np.asarray([all_columns(compiled, p) for p in points])
        errors = [scaled_error(a, b, scales) for a, b in zip(expected, previous_hessians, strict=True)]
        assert max(errors) < 1e-10
        stored, storage = SpooledExecutable.write(compiled, directory/"largest.bin", backend="cpu")
        report.update(storage, previous_hessian_scaled_errors=errors,
                      previous_hessians_bitwise_equal=bool(np.array_equal(expected, previous_hessians)))
        del compiled, lowered
        term.reset_kernel("hessian_columns")
        release_compilation_cache()
        report["stage"] = "scoped reload, one per complete local Hessian"
        save()
        actual, times = [], []
        for point in points:
            begin = time.perf_counter()
            with LoadedExecutable(stored) as loaded:
                actual.append(all_columns(loaded, point))
            times.append(time.perf_counter()-begin)
        actual = np.asarray(actual)
        np.testing.assert_array_equal(actual, expected)
        assert len(stored.scoped_loads) == len(points) == 2
        assert len(stored.calls) == 68 and [s["calls"] for s in stored.scoped_loads] == [34, 34]
        artifact = directory/"hessians.npz"
        np.savez_compressed(artifact, points=points, parameter_names=np.asarray(term.parameter_names),
            original_hessians=expected, scoped_hessians=actual, parameter_scales=scales)
        report.update(status="passed", stage="finished", arrays=str(artifact),
            arrays_sha256=digest(artifact), reload_max_absolute_error=0., full_local_call_seconds=times,
            kernel_calls=len(stored.calls), scoped_loads=stored.scoped_loads, call_timings=stored.calls,
            reload_component_totals={k: sum(c[k] for c in stored.calls) for k in stored.calls[0]})
        save()
        print(report["status"], times, report["reload_component_totals"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()


if __name__ == "__main__":
    main()
