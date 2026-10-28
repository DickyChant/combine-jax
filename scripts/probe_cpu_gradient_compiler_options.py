"""Measure selected retained CPU gradients with explicit per-kernel options.

Compare all saved points against default-compiled gradients. This is a compiler
diagnostic, not full-model Combine acceptance or a GPU benchmark.
"""

import argparse
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

from sharded_cpu_evaluator import core_digest, digest, rss


OPTIONS = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--terms", type=int, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-fusion", action="store_true")
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use fresh reports/*.json output")
    if args.no_fusion != bool(args.reference):
        parser.error("A no-fusion probe requires its completed default-options reference")
    artifact = Path("build/gpu_results")/(args.output.stem+".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError("Preserve prior compiler measurements")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This diagnostic is local only; preserve the lxplus benchmark")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[0]])
    import jax
    import jaxlib
    from jaxlib import xla_client
    from combine_jax.local import release_compilation_cache
    from hhvv_export_cache import load_export

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert "ROOT" not in sys.modules
    jax.config.update("jax_enable_compilation_cache", False)
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    debug_options = build_options.debug_options
    assert debug_options.xla_disable_hlo_passes == ""
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    likelihood = load_export(export, model["source"])
    evaluator = likelihood.local_evaluator(constraint_batch_size=32)
    selected = args.terms if args.terms is not None else list(range(len(evaluator.terms)))
    assert selected and len(set(selected)) == len(selected)
    assert all(0 <= i < len(evaluator.terms) for i in selected)
    points = np.asarray([c["theta"] for c in model["reference"]["cases"]], dtype=np.float64)
    assert len(points) == 26
    options = OPTIONS if args.no_fusion else {}
    started = time.perf_counter()
    report = dict(status="running", stage="setup", model=args.model, host=platform.node(),
        source=model["source"], source_sha256=core_digest(), export_sha256=digest(export),
        manifest_sha256=digest(manifest_path), runner_sha256=digest(__file__),
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        jax_version=jax.__version__, jaxlib_version=jaxlib.__version__, float64=True,
        backend="cpu", cpu_affinity=sorted(os.sched_getaffinity(0)), root_imported=False,
        xla_flags=os.environ.get("XLA_FLAGS"), default_disabled_hlo_passes="",
        compiler_options=options, selected_terms=selected, cases=[],
        baseline_rss_mib=rss(os.getpid()), baseline_mappings=len(Path("/proc/self/maps").read_text().splitlines()),
        rss_limit_mib=4500., wall_limit_seconds=1800.,
        elapsed_scope="After model load and device initialization; includes all selected preparation, calls, validation and cleanup.",
        scope="Selected local retained CPU gradient kernels at all 26 fixed manifest points; comparison with default compiler options only. No full HHVV, independent Combine, fit, lxplus or GPU acceptance.",
        reference=str(args.reference) if args.reference else None,
        reference_sha256=digest(args.reference) if args.reference else None)
    reference_report = None
    expected = {}
    if args.reference:
        reference_report = json.loads(args.reference.read_bytes())
        assert reference_report["status"] == "computed" and reference_report["compiler_options"] == {}
        for key in ("model", "source", "source_sha256", "export_sha256", "manifest_sha256",
                    "runner_sha256", "jax_version", "jaxlib_version", "float64", "selected_terms"):
            assert report[key] == reference_report[key], key
        assert digest(reference_report["arrays"]) == reference_report["arrays_sha256"]
        with np.load(reference_report["arrays"], allow_pickle=False) as arrays:
            expected = {key: arrays[key] for key in arrays.files}
        np.testing.assert_array_equal(expected["points"], points)
        np.testing.assert_array_equal(expected["parameter_names"], likelihood.parameter_names)
    lock, stop = threading.RLock(), threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                          current_rss_mib=rss(os.getpid()),
                          mappings=len(Path("/proc/self/maps").read_text().splitlines()))
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
                    report.update(status="resource_limit", exception="Configured memory/wall guard reached")
                    save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], report["elapsed_seconds"], report["current_rss_mib"], flush=True)
                last = time.perf_counter()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    data = dict(points=points, parameter_names=np.asarray(likelihood.parameter_names))
    try:
        for index in selected:
            term = evaluator.terms[index]
            point = points[0, term.indices]
            record = dict(index=index, name=term.name, parameters=len(point))
            with lock:
                report["stage"] = f"compile term {index}: {term.name}"
                save()
            begin = time.perf_counter()
            lowered = term.value_and_grad.lower(point)
            record["lowering_seconds"] = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile(compiler_options=options)
            record["compile_seconds"] = time.perf_counter()-begin
            term.value_and_grad = compiled
            del lowered, compiled
            release_compilation_cache()
            record.update(retained_rss_mib=rss(os.getpid()),
                          retained_mappings=len(Path("/proc/self/maps").read_text().splitlines()))
            samples = []
            for _ in range(6):
                begin = time.perf_counter()
                value, gradient = term.value_and_grad(point)
                float(value), np.asarray(gradient)
                samples.append(time.perf_counter()-begin)
            record.update(first_call_seconds=samples[0], warm_seconds=samples[1:],
                          warm_median_seconds=float(np.median(samples[1:])))
            values, gradients = [], []
            for full_point in points:
                value, gradient = term.value_and_grad(full_point[term.indices])
                values.append(float(value))
                gradients.append(np.asarray(gradient))
            values, gradients = np.asarray(values), np.asarray(gradients)
            assert np.isfinite(values).all() and np.isfinite(gradients).all()
            data.update({f"indices_{index}": term.indices, f"values_{index}": values,
                         f"gradients_{index}": gradients})
            if expected:
                np.testing.assert_array_equal(expected[f"indices_{index}"], term.indices)
                value_error = float(np.max(abs(values-expected[f"values_{index}"])))
                gradient_error = float(np.max(abs(gradients-expected[f"gradients_{index}"])/
                                               np.maximum(1., abs(expected[f"gradients_{index}"])), initial=0.))
                record.update(value_max_absolute_error=value_error,
                              gradient_max_scaled_error=gradient_error,
                              comparison_passed=value_error < 1e-8 and gradient_error < 1e-10)
            with lock:
                report["cases"].append(record)
                save()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **data)
        temporary.replace(artifact)
        with lock:
            passed = all(c["comparison_passed"] for c in report["cases"]) if expected else True
            report.update(status=("passed" if expected else "computed") if passed else "failed",
                          stage="finished", arrays=str(artifact), arrays_sha256=digest(artifact))
            save()
    except BaseException:
        with lock:
            report.update(status="failed", exception=traceback.format_exc())
            save()
        raise
    finally:
        stop.set()
        watcher.join()
    print(args.output, report["status"], report["peak_rss_mib"], flush=True)
    raise SystemExit(0 if report["status"] in ("passed", "computed") else 1)


if __name__ == "__main__":
    main()
