"""Measure CPU executable retention after a trusted serialization round trip.

All compared kernels are compiled in this process. No cross-process cache
compatibility or full-model memory feasibility is assumed.
"""

import argparse
from contextlib import ExitStack
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

import numpy as np

from sharded_cpu_evaluator import NO_FUSION_OPTIONS, core_digest, digest, rss


def mappings():
    return len(Path("/proc/self/maps").read_text().splitlines())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use a fresh reports/*.json output")
    directory = Path("build/gpu_results")/args.output.stem
    if args.output.exists() or directory.exists():
        raise FileExistsError("Preserve earlier retention measurements")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This probe is local only; preserve the remote benchmark")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[0]])
    import jax
    import jaxlib
    from combine_jax.local import release_compilation_cache
    from hhvv_export_cache import load_export
    from scoped_executable import LoadedExecutable
    from spooled_executable import SpooledExecutable

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert "ROOT" not in sys.modules
    jax.config.update("jax_enable_compilation_cache", False)
    reference_path = Path(f"reports/gpu_local_{args.model}_gradient_compiler_no_fusion.json")
    reference = json.loads(reference_path.read_bytes())
    assert reference["status"] == "passed" and reference["compiler_options"] == NO_FUSION_OPTIONS
    assert reference["source_sha256"] == core_digest()
    assert reference["jax_version"] == jax.__version__ and reference["jaxlib_version"] == jaxlib.__version__
    assert reference["arrays_sha256"] == digest(reference["arrays"])
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    assert reference["source"] == model["source"] and reference["export_sha256"] == digest(export)
    with np.load(reference["arrays"], allow_pickle=False) as arrays:
        previous = {k: arrays[k] for k in arrays.files}
    likelihood = load_export(export, model["source"])
    evaluator = likelihood.local_evaluator(constraint_batch_size=32)
    points = np.asarray([c["theta"] for c in model["reference"]["cases"]])
    np.testing.assert_array_equal(previous["points"], points)
    np.testing.assert_array_equal(previous["parameter_names"], likelihood.parameter_names)
    selected = reference["selected_terms"]
    directory.mkdir(parents=True)
    started = time.perf_counter()
    report = dict(status="running", stage="setup", model=args.model, host=platform.node(),
        backend="cpu", float64=True, jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        source=model["source"], source_sha256=core_digest(), export_sha256=digest(export),
        manifest_sha256=digest(manifest_path), runner_sha256=digest(__file__),
        helper_sha256={p: digest(p) for p in ("scripts/sharded_cpu_evaluator.py",
            "scripts/spooled_executable.py", "scripts/scoped_executable.py")},
        reference=str(reference_path), reference_sha256=digest(reference_path),
        compiler_options=NO_FUSION_OPTIONS, selected_terms=selected,
        cpu_affinity=sorted(os.sched_getaffinity(0)), root_imported=False,
        baseline_rss_mib=rss(os.getpid()), baseline_mappings=mappings(),
        rss_limit_mib=4500., wall_limit_seconds=1800., cases=[],
        scope="Selected CPU gradients retained after process-local serialization/reload; all 26 points must agree bitwise before/after reload. No full HHVV, independent Combine, fitting or GPU performance acceptance.",
        elapsed_scope="After device initialization, model load and reference checks; includes compilation, serialization, calls, validation and final executable cleanup.")
    lock, stop = threading.RLock(), threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          current_rss_mib=rss(os.getpid()), mappings=mappings(),
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = started
        while not stop.wait(.1):
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if memory > report["rss_limit_mib"] or time.perf_counter()-started > report["wall_limit_seconds"]:
                with lock:
                    if report["status"] != "running":
                        return
                    report.update(status="resource_limit", exception="Configured probe guard reached")
                    save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], report["elapsed_seconds"], report["current_rss_mib"], flush=True)
                last = time.perf_counter()

    def evaluate(kernel, indices):
        values, gradients = [], []
        for point in points:
            value, gradient = kernel(point[indices])
            values.append(float(value))
            gradients.append(np.array(gradient, copy=True))
        return np.asarray(values), np.asarray(gradients)

    data = dict(points=points, parameter_names=np.asarray(likelihood.parameter_names))
    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with ExitStack() as retained:
            kernels = []
            for index in selected:
                term = evaluator.terms[index]
                record = dict(index=index, name=term.name, parameters=len(term.indices),
                              rss_before_compile_mib=rss(os.getpid()), mappings_before_compile=mappings())
                with lock:
                    report["stage"] = f"compile term {index}: {term.name}"
                    save()
                begin = time.perf_counter()
                lowered = term.value_and_grad.lower(points[0, term.indices])
                record["lowering_seconds"] = time.perf_counter()-begin
                begin = time.perf_counter()
                original = lowered.compile(compiler_options=NO_FUSION_OPTIONS)
                record["compile_seconds"] = time.perf_counter()-begin
                del lowered
                release_compilation_cache()
                record.update(original_retained_rss_mib=rss(os.getpid()), original_retained_mappings=mappings())
                expected_values, expected_gradients = evaluate(original, term.indices)
                np.testing.assert_array_equal(expected_values, previous[f"values_{index}"])
                np.testing.assert_array_equal(expected_gradients, previous[f"gradients_{index}"])
                stored, serialization = SpooledExecutable.write(original, directory/f"{index}.bin", backend="cpu")
                record.update(serialization)
                del original
                release_compilation_cache()
                record.update(unloaded_rss_mib=rss(os.getpid()), unloaded_mappings=mappings())
                loaded = retained.enter_context(LoadedExecutable(stored))
                release_compilation_cache()
                record.update(reloaded_retained_rss_mib=rss(os.getpid()),
                              reloaded_retained_mappings=mappings(),
                              read_and_verify_seconds=loaded.read_seconds, load_seconds=loaded.load_seconds)
                actual_values, actual_gradients = evaluate(loaded, term.indices)
                np.testing.assert_array_equal(actual_values, expected_values)
                np.testing.assert_array_equal(actual_gradients, expected_gradients)
                samples = []
                for _ in range(6):
                    begin = time.perf_counter()
                    loaded(points[0, term.indices])
                    samples.append(time.perf_counter()-begin)
                record.update(reload_bitwise_equal=True, previous_probe_bitwise_equal=True,
                              first_timed_call_seconds=samples[0], warm_seconds=samples[1:],
                              warm_median_seconds=float(np.median(samples[1:])))
                data.update({f"indices_{index}": term.indices, f"values_{index}": actual_values,
                             f"gradients_{index}": actual_gradients})
                kernels.append((index, term.indices, loaded))
                with lock:
                    report["cases"].append(record)
                    save()
            # Earlier reloaded kernels must still work after all later compilations.
            for index, indices, loaded in kernels:
                values, gradients = evaluate(loaded, indices)
                np.testing.assert_array_equal(values, data[f"values_{index}"])
                np.testing.assert_array_equal(gradients, data[f"gradients_{index}"])
            with lock:
                report.update(all_retained_rss_mib=rss(os.getpid()), all_retained_mappings=mappings(),
                              all_kernels_rechecked_after_preparation=True)
                save()
        artifact = directory/"arrays.npz"
        np.savez_compressed(artifact, **data)
        with lock:
            report.update(status="passed", stage="finished", arrays=str(artifact),
                          arrays_sha256=digest(artifact),
                          load_scope_calls={str(index): len(kernel.records) for index, _, kernel in kernels},
                          after_release_rss_mib=rss(os.getpid()), after_release_mappings=mappings())
            save()
    except BaseException:
        with lock:
            report.update(status="failed", exception=traceback.format_exc())
            save()
        raise
    finally:
        stop.set()
        watcher.join()
    print(args.output, report["status"], report["all_retained_rss_mib"], report["after_release_rss_mib"], flush=True)


if __name__ == "__main__":
    main()
