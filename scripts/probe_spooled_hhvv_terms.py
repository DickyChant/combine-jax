"""Check reloads of exact Hessian columns on two independently selected HHVV terms."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import threading
import time
import traceback

import jax
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export
from spooled_executable import SpooledExecutable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/gpu_local_spooled_hhvv_terms.json"))
    parser.add_argument("--directory", type=Path, default=Path("build/gpu_results/local_spooled_hhvv_terms"))
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64 and "ROOT" not in sys.modules
    started = time.perf_counter()
    output = arguments.output
    directory = arguments.directory
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior HHVV executable probe evidence")
    directory.mkdir(parents=True)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    previous_path = Path("reports/gpu_hhvv_gpu_cachetrim.json")
    previous = json.loads(previous_path.read_text())
    assert previous["status"] == "resource_limit" and previous["compile_progress"]["operation"] == "hessian"
    report = dict(status="running", stage="load_export",
        scope="Two local CPU HHVV term Hessians; serialization equivalence only; no Combine Hessian, full-model, GPU or isolated performance acceptance",
        selection="First largest local parameter count, then the term compiling at the preserved GPU resource limit",
        jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__, float64=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), backend="cpu", root_imported=False,
        hessian_chunk_size=4, constraint_batch_size=32,
        manifest=str(manifest_path), manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        previous_gpu_report=str(previous_path), previous_gpu_report_sha256=hashlib.sha256(previous_path.read_bytes()).hexdigest(),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in (Path(__file__), Path("scripts/spooled_executable.py"))},
        source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
            for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
        memory_guard="Stop if own peak plus running reference-fit peak exceeds 11000 MiB; sampled every 0.1 s",
        terms=[])
    lock = threading.RLock()
    stopped = threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def watch_memory():
        while not stopped.wait(.1):
            own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            fit_path = Path("reports/hhvv_combine_fit.json")
            try:
                fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
            except json.JSONDecodeError:
                continue
            other = fit.get("peak_rss_mib", 0.) if fit.get("status") == "running" else 0.
            if own+other > 11000:
                with lock:
                    report.update(status="resource_limit", fit_peak_rss_mib=other)
                    save()
                os._exit(86)

    save()
    watcher = threading.Thread(target=watch_memory, daemon=True)
    watcher.start()
    try:
        config = manifest["models"]["hhvv"]
        likelihood = load_export(Path("build/gpu_exports")/config["cache"], config["source"])
        assert config["source"] == previous["source"]
        report["source"] = config["source"]
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        largest = max(range(len(evaluator.terms)), key=lambda i:len(evaluator.terms[i].indices))
        failure = previous["compile_progress"]["completed_terms"]
        assert previous["current_stage"].endswith(evaluator.terms[failure].name)
        indices = list(dict.fromkeys((largest, failure)))
        points = np.asarray([c["theta"] for c in config["reference"]["cases"][:2]])

        def all_columns(kernel, point):
            size = len(point)
            eye = np.eye(size)
            result = np.empty((size, size))
            chunk = min(4, size)
            for first in range(0, size, chunk):
                count = min(chunk, size-first)
                vectors = np.zeros((chunk, size))
                vectors[:count] = eye[first:first+count]
                result[:, first:first+count] = np.asarray(kernel(point, vectors))[:count].T
            assert np.isfinite(result).all()
            return result

        for index in indices:
            term = evaluator.terms[index]
            local_points = points[:, term.indices]
            case = dict(index=index, name=term.name, parameters=len(term.indices), points=len(points))
            with lock:
                report["terms"].append(case)
                report["stage"] = "compile "+term.name
                save()
            begin = time.perf_counter()
            lowered = term.hessian_columns.lower(local_points[0], np.zeros((min(4,len(term.indices)),len(term.indices))))
            case["lowering_seconds"] = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile()
            case["compile_seconds"] = time.perf_counter()-begin
            expected = np.asarray([all_columns(compiled, point) for point in local_points])
            stored, record = SpooledExecutable.write(compiled, directory/f"{index}.bin", backend="cpu")
            case.update(record)
            term.hessian_columns = stored
            del compiled, lowered
            release_compilation_cache()
            with lock:
                report["stage"] = "reload "+term.name
                save()
            actual = np.asarray([all_columns(stored, point) for point in local_points])
            np.testing.assert_array_equal(actual, expected)
            artifact = directory/f"{index}.npz"
            np.savez_compressed(artifact, parameter_names=np.asarray(likelihood.parameter_names)[term.indices],
                points=local_points, original_hessians=expected, spooled_hessians=actual)
            case.update(status="passed", max_absolute_error=float(np.max(abs(actual-expected),initial=0.)),
                arrays=str(artifact), arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                kernel_calls=len(stored.calls), call_timings=stored.calls)
            assert len(stored.calls) == len(points)*int(np.ceil(len(term.indices)/4))
            print(term.name, case["status"], case["parameters"], "parameters", flush=True)
            evaluator.clear_cache(release_cache=True)
            save()
        report.update(status="passed", stage="complete")
        save()
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stopped.set()
        watcher.join()


if __name__ == "__main__":
    main()
