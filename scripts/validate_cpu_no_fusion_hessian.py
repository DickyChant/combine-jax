"""Validate a per-Hessian CPU compiler option without changing likelihood code."""

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
from jaxlib import xla_client
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export
from spooled_executable import SpooledExecutable


OPTIONS = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}
FD_STEPS = (1e-3, 3e-4, 1e-4)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaled_error(left, right, scales):
    left = np.asarray(left)*np.outer(scales, scales)
    right = np.asarray(right)*np.outer(scales, scales)
    return float(np.max(abs(left-right)/np.maximum(1., abs(right)), initial=0.))


def all_columns(kernel, point, chunk=4):
    size = len(point)
    chunk = min(chunk, size)
    result = np.empty((size, size))
    for first in range(0, size, chunk):
        width = min(chunk, size-first)
        vectors = np.zeros((chunk, size))
        vectors[np.arange(width), first+np.arange(width)] = 1.
        result[:, first:first+width] = np.asarray(kernel(point, vectors))[:width].T
    assert np.isfinite(result).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--rss-limit-mib", type=float, default=3500.)
    args = parser.parse_args()
    if not 0 < args.rss_limit_mib <= 11000:
        parser.error("RSS guard must be in (0, 11000] MiB")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert "ROOT" not in sys.modules
    options = xla_client.CompileOptions()
    build_options = options.executable_build_options
    debug_options = build_options.debug_options
    assert debug_options.xla_disable_hlo_passes == "", "Gradient reference must retain default fusion"
    output = Path(f"reports/{args.model}_cpu_no_fusion_hessian.json")
    directory = Path(f"build/gpu_results/{args.model}_cpu_no_fusion_hessian")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior compiler validation evidence")
    directory.mkdir(parents=True)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"][args.model]
    export_path = Path("build/gpu_exports")/model["cache"]
    started = time.perf_counter()
    report = dict(
        status="running", stage="setup", model=args.model, source=model["source"],
        jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__,
        float64=True, backend="cpu", cpu_affinity=sorted(os.sched_getaffinity(0)),
        xla_flags=os.environ.get("XLA_FLAGS"), hessian_compiler_options=OPTIONS,
        default_disabled_hlo_passes=debug_options.xla_disable_hlo_passes,
        persistent_cache_enabled=False, rss_limit_mib=args.rss_limit_mib,
        manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
        export_sha256=digest(export_path),
        runner_sha256={str(p): digest(p) for p in (Path(__file__), Path("scripts/spooled_executable.py"))},
        source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
            for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
        scope="Local CPU compiler-option and serialization validation; no full HHVV Hessian, independent Combine curvature, GPU or isolated performance acceptance",
        policy=dict(smoke_scaled_tolerance=1e-10, fd_scaled_relative_tolerance=1e-4,
                    fd_steps=FD_STEPS, fd_formula="fourth-order central difference of default-compiled gradients",
                    fd_acceptance="Both last two predeclared steps and their mutual difference below tolerance",
                    fixed_parameters="Included in matrices and reload checks; excluded from finite-difference rows and columns"),
        cases=[])
    stopped = threading.Event()
    lock = threading.RLock()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def monitor():
        last_print = started
        while not stopped.wait(.1):
            if time.perf_counter()-last_print > 20:
                save()
                print(report["stage"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
                last_print = time.perf_counter()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > args.rss_limit_mib:
                with lock:
                    report.update(status="resource_limit", exception="Configured own-RSS guard exceeded")
                    save()
                os._exit(86)

    save()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export_path, model["source"])
        points = np.asarray([c["theta"] for c in model["reference"]["cases"]])
        scales = np.array([p.error if p.error > 0 else max(abs(p.value), 1.)
                           for p in likelihood.parameters])
        if args.model != "hhvv":
            # Compile both variants from the same lowered graph. No option is
            # changed for the default reference.
            report["stage"] = "compile smoke Hessians"
            lowered = likelihood.hessian.lower(points[0])
            normal = lowered.compile()
            modified = lowered.compile(compiler_options=OPTIONS)
            expected = np.asarray([normal(p) for p in points])
            actual = np.asarray([modified(p) for p in points])
            assert np.isfinite(expected).all() and np.isfinite(actual).all()
            errors = [scaled_error(a, b, scales) for a, b in zip(actual, expected, strict=True)]
            saved = []
            for i, case in enumerate(model["reference"]["cases"]):
                if "hessian" in case:
                    saved.append(dict(point=i, scaled_error=scaled_error(actual[i], case["hessian"], scales)))
            assert len(saved) == 3
            passed = max(errors) < 1e-10 and max(c["scaled_error"] for c in saved) < 1e-10
            artifact = directory/"hessians.npz"
            np.savez_compressed(artifact, points=points, default_hessians=expected, no_fusion_hessians=actual)
            report["cases"].append(dict(points=len(points), default_scaled_errors=errors,
                saved_cpu_reference_errors=saved, arrays=str(artifact), arrays_sha256=digest(artifact),
                status="passed" if passed else "failed"))
        else:
            evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
            previous_path = Path("reports/gpu_hhvv_gpu_cachetrim.json")
            previous = json.loads(previous_path.read_text())
            assert previous["status"] == "resource_limit" and previous["compile_progress"]["operation"] == "hessian"
            failure = previous["compile_progress"]["completed_terms"]
            assert previous["current_stage"].endswith(evaluator.terms[failure].name)
            largest = max(range(len(evaluator.terms)), key=lambda i: len(evaluator.terms[i].indices))
            selected = list(dict.fromkeys((largest, failure)))
            report.update(selection="Largest local parameter count and previous GPU failure term; nominal and first certified random point",
                          previous_gpu_report=str(previous_path), previous_gpu_report_sha256=digest(previous_path))
            for index in selected:
                term = evaluator.terms[index]
                local_points = points[:2, term.indices]
                parameters = [likelihood.parameters[i] for i in term.indices]
                local_scales = scales[term.indices]
                free = np.array([p.minimum != p.maximum for p in parameters])
                free_indices = np.flatnonzero(free)
                case = dict(index=index, name=term.name, parameters=len(parameters),
                            free_parameters=int(free.sum()), status="running")
                report["cases"].append(case)
                report["stage"] = "compile no-fusion Hessian "+term.name
                save()
                begin = time.perf_counter()
                lowered = term.hessian_columns.lower(local_points[0],
                    np.zeros((min(4, len(parameters)), len(parameters))))
                case["lowering_seconds"] = time.perf_counter()-begin
                begin = time.perf_counter()
                compiled = lowered.compile(compiler_options=OPTIONS)
                case["compile_seconds"] = time.perf_counter()-begin
                expected = np.asarray([all_columns(compiled, p) for p in local_points])
                stored, record = SpooledExecutable.write(compiled, directory/f"{index}.bin", backend="cpu")
                case.update(record)
                del compiled, lowered
                release_compilation_cache()
                report["stage"] = "reload no-fusion Hessian "+term.name
                actual = np.asarray([all_columns(stored, p) for p in local_points])
                np.testing.assert_array_equal(actual, expected)
                case.update(reload_max_absolute_error=0., kernel_calls=len(stored.calls),
                            call_timings=stored.calls,
                            max_asymmetry=float(abs(expected-expected.transpose(0, 2, 1)).max()))
                report["stage"] = "compile default gradient "+term.name
                save()
                begin = time.perf_counter()
                gradient = term.value_and_grad.lower(local_points[0]).compile()
                case["default_gradient_preparation_seconds"] = time.perf_counter()-begin
                release_compilation_cache()
                references = []
                validations = []
                for point_index, point in enumerate(local_points):
                    matrices = []
                    for step in FD_STEPS:
                        report["stage"] = f"default-gradient finite differences {term.name} point {point_index} step {step}"
                        matrix = np.empty((len(free_indices), len(free_indices)))
                        for column, coordinate in enumerate(free_indices):
                            h = step*local_scales[coordinate]
                            par = parameters[coordinate]
                            assert par.minimum <= point[coordinate]-2*h and point[coordinate]+2*h <= par.maximum
                            samples = []
                            for multiplier in (-2., -1., 1., 2.):
                                displaced = point.copy()
                                displaced[coordinate] += multiplier*h
                                value, sample = gradient(displaced)
                                assert np.isfinite(float(value))
                                sample = np.asarray(sample)[free]
                                assert np.isfinite(sample).all()
                                samples.append(sample)
                            matrix[:, column] = (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*h)
                        assert np.isfinite(matrix).all()
                        matrices.append(matrix)
                    references.append(matrices)
                    hessian = expected[point_index][np.ix_(free, free)]
                    errors = [scaled_error(hessian, m, local_scales[free]) for m in matrices]
                    convergence = scaled_error(matrices[-2], matrices[-1], local_scales[free])
                    passed = max(errors[-2:]+[convergence]) < 1e-4
                    validations.append(dict(point=point_index, fd_errors=errors,
                                            last_two_step_difference=convergence, passed=passed))
                artifact = directory/f"{index}.npz"
                np.savez_compressed(artifact, parameter_names=np.asarray(term.parameter_names), points=local_points,
                    original_hessians=expected, spooled_hessians=actual, free_indices=free_indices,
                    fd_steps=FD_STEPS, default_gradient_fd_hessians=np.asarray(references), parameter_scales=local_scales)
                case.update(status="passed" if all(v["passed"] for v in validations) else "failed",
                            finite_difference_checks=validations, arrays=str(artifact), arrays_sha256=digest(artifact))
                del gradient, stored
                evaluator.clear_cache(release_cache=True)
                print(term.name, case["status"], validations, flush=True)
                save()
        report.update(status="passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed",
                      stage="finished")
        save()
        assert report["status"] == "passed", "Numerical compiler validation failed; evidence retained"
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stopped.set()
        watcher.join()


if __name__ == "__main__":
    main()
