"""Validate captured-array arguments before any full resident-gradient retry."""

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

from sharded_cpu_evaluator import NO_FUSION_OPTIONS, core_digest, digest, rss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use fresh reports/*.json output")
    artifact = Path("build/gpu_results")/(args.output.stem+".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous constant-argument evidence")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This probe is local only")
    assert "jax" not in sys.modules, "Set the constant policy before importing JAX"
    os.environ["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = "true"
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[0]])
    import jax
    import jaxlib
    from jax._src.interpreters import mlir
    from jax.experimental import serialize_executable
    from combine_jax.local import release_compilation_cache
    from hhvv_export_cache import load_export

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert jax.config.jax_use_simplified_jaxpr_constants
    assert mlir.LoweringParameters().hoist_constants_as_args
    assert "ROOT" not in sys.modules
    jax.config.update("jax_enable_compilation_cache", False)
    reference_path = Path(f"reports/gpu_local_{args.model}_gradient_compiler_no_fusion.json")
    reference = json.loads(reference_path.read_bytes())
    assert reference["status"] == "passed" and reference["compiler_options"] == NO_FUSION_OPTIONS
    assert reference["source_sha256"] == core_digest()
    assert reference["jax_version"] == jax.__version__ and reference["jaxlib_version"] == jaxlib.__version__
    assert reference["arrays_sha256"] == digest(reference["arrays"])
    with np.load(reference["arrays"], allow_pickle=False) as arrays:
        previous = {k: arrays[k] for k in arrays.files}
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    assert model["source"] == reference["source"] and digest(export) == reference["export_sha256"]
    likelihood = load_export(export, model["source"])
    evaluator = likelihood.local_evaluator(constraint_batch_size=32)
    points = np.asarray([c["theta"] for c in model["reference"]["cases"]])
    np.testing.assert_array_equal(previous["points"], points)
    np.testing.assert_array_equal(previous["parameter_names"], likelihood.parameter_names)
    started = time.perf_counter()
    report = dict(status="running", stage="setup", model=args.model, host=platform.node(),
        backend="cpu", float64=True, jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        source=model["source"], source_sha256=core_digest(), export_sha256=digest(export),
        manifest_sha256=digest(manifest_path), runner_sha256=digest(__file__),
        reference=str(reference_path), reference_sha256=digest(reference_path),
        compiler_options=NO_FUSION_OPTIONS, selected_terms=reference["selected_terms"],
        simplified_jaxpr_constants=True, lowering_hoists_constants_as_arguments=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), root_imported=False,
        baseline_rss_mib=rss(os.getpid()), baseline_mappings=len(Path("/proc/self/maps").read_text().splitlines()),
        rss_limit_mib=4500., wall_limit_seconds=1800., cases=[],
        scope="Selected retained CPU gradient kernels with captured-array arguments, compared at all 26 points against the preceding no-fusion kernels. No full HHVV, independent Combine, fit or GPU acceptance.",
        policy=dict(value_absolute_tolerance=1e-8, gradient_scaled_tolerance=1e-10,
                    gradient_denominator="max(1, abs(reference gradient))"),
        elapsed_scope="After device/model initialization and reference checks; includes compilation, all calls, checks and final kernel cleanup.")
    lock, stop = threading.RLock(), threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                current_rss_mib=rss(os.getpid()), peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                mappings=len(Path("/proc/self/maps").read_text().splitlines()))
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = started
        while not stop.wait(.1):
            if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > report["rss_limit_mib"]
                    or time.perf_counter()-started > report["wall_limit_seconds"]):
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

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    data = dict(points=points, parameter_names=np.asarray(likelihood.parameter_names))
    try:
        for index in report["selected_terms"]:
            term = evaluator.terms[index]
            record = dict(index=index, name=term.name, parameters=len(term.indices))
            with lock:
                report["stage"] = f"compile term {index}: {term.name}"
                save()
            begin = time.perf_counter()
            lowered = term.value_and_grad.lower(points[0, term.indices])
            record["lowering_seconds"] = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile(compiler_options=NO_FUSION_OPTIONS)
            record["compile_seconds"] = time.perf_counter()-begin
            constants = compiled._params.const_args
            record.update(constant_arguments=len(constants),
                          constant_argument_bytes=sum(int(c.nbytes) for c in constants))
            if constants:
                try:
                    serialize_executable.serialize(compiled)
                except NotImplementedError as error:
                    record["serialization"] = dict(supported=False, exception=str(error))
                else:
                    record["serialization"] = dict(supported=True)
            else:
                record["serialization"] = dict(attempted=False, reason="No captured-array arguments")
            term.value_and_grad = compiled
            del compiled, lowered, constants
            release_compilation_cache()
            record.update(retained_rss_mib=rss(os.getpid()),
                          retained_mappings=len(Path("/proc/self/maps").read_text().splitlines()))
            samples = []
            for _ in range(6):
                begin = time.perf_counter()
                value, gradient = term.value_and_grad(points[0, term.indices])
                float(value), np.asarray(gradient)
                samples.append(time.perf_counter()-begin)
            values, gradients = [], []
            for point in points:
                value, gradient = term.value_and_grad(point[term.indices])
                values.append(float(value))
                gradients.append(np.array(gradient, copy=True))
            values, gradients = np.asarray(values), np.asarray(gradients)
            assert np.isfinite(values).all() and np.isfinite(gradients).all()
            value_error = float(np.max(abs(values-previous[f"values_{index}"])))
            gradient_error = float(np.max(abs(gradients-previous[f"gradients_{index}"])/
                                          np.maximum(1., abs(previous[f"gradients_{index}"])), initial=0.))
            record.update(first_call_seconds=samples[0], warm_seconds=samples[1:],
                warm_median_seconds=float(np.median(samples[1:])),
                value_max_absolute_error=value_error, gradient_max_scaled_error=gradient_error,
                comparison_passed=value_error < 1e-8 and gradient_error < 1e-10)
            data.update({f"indices_{index}": term.indices, f"values_{index}": values,
                         f"gradients_{index}": gradients})
            with lock:
                report["cases"].append(record)
                save()
        for index in report["selected_terms"]:
            term = evaluator.terms[index]
            value, gradient = term.value_and_grad(points[0, term.indices])
            np.testing.assert_equal(float(value), data[f"values_{index}"][0])
            np.testing.assert_array_equal(gradient, data[f"gradients_{index}"][0])
        with lock:
            report.update(all_retained_rss_mib=rss(os.getpid()),
                          all_retained_mappings=len(Path("/proc/self/maps").read_text().splitlines()),
                          all_nominal_outputs_rechecked_after_preparation=True)
        for index in report["selected_terms"]:
            evaluator.terms[index].reset_kernel("value_and_grad")
        release_compilation_cache()
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **data)
        temporary.replace(artifact)
        with lock:
            report.update(status="passed" if all(c["comparison_passed"] for c in report["cases"]) else "failed",
                stage="finished", arrays=str(artifact), arrays_sha256=digest(artifact),
                after_release_rss_mib=rss(os.getpid()))
            save()
    except BaseException:
        with lock:
            report.update(status="failed", exception=traceback.format_exc())
            save()
        raise
    finally:
        stop.set()
        watcher.join()
    print(args.output, report["status"], report["all_retained_rss_mib"], flush=True)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
