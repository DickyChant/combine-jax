"""Validate all points, then fit with resident NLLs and bounded gradient loads."""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib
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

from fit_exported_likelihood import checked_initial
from measure_sharded_cpu_gradient import compiler_validation
from sharded_cpu_evaluator import NO_FUSION_OPTIONS, core_digest, digest


def run_checker(module_name, arguments, log):
    original_arguments = sys.argv
    try:
        sys.argv = [module_name+".py", *arguments]
        with log.open("x") as stream, redirect_stdout(stream), redirect_stderr(stream):
            importlib.import_module(module_name).main()
    finally:
        sys.argv = original_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-check", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--strategy", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--max-calls", type=int, default=50000)
    parser.add_argument("--migrad-iterate", type=int, default=5,
                        help="Maximum MIGRAD attempts; retries may escalate the Minuit strategy")
    parser.add_argument("--no-simplex", action="store_true",
                        help="Disable the simplex fallback between MIGRAD attempts")
    parser.add_argument("--rss-limit-mib", type=float, default=11000.)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    args = parser.parse_args()
    if (args.output.parent != Path("reports") or args.output.suffix != ".json" or args.max_calls < 1
            or args.migrad_iterate < 1
            or not 0 < args.rss_limit_mib <= 11000 or not 0 < args.wall_seconds < np.inf):
        parser.error("Use reports/*.json and positive limits, with RSS <=11000 MiB")
    if args.model == "hhvv" and not args.validate_only and not args.initial_check:
        parser.error("The HHVV fit requires its source-bound checked restart")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This experimental fit is local only")
    checkpoint_path = args.output.with_name(args.output.stem+"_gradient_validation.json")
    directory = Path("build/fit_results")/(args.output.stem+"_executables")
    validation_arrays = Path("build/fit_results")/(args.output.stem+"_validation.npz")
    fit_arrays = Path("build/fit_results")/(args.output.stem+".npz")
    if any(p.exists() for p in (args.output, checkpoint_path, directory, validation_arrays, fit_arrays)):
        raise FileExistsError("Preserve previous fit and validation evidence")
    started = time.perf_counter()
    assert "jax" not in sys.modules
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                      JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[0]])
    import jax
    import jaxlib
    import iminuit
    from jaxlib import xla_client
    from hhvv_export_cache import load_export
    from resident_value_spooled_gradient import ResidentValueSpooledGradient, fit_separate_calls

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert not jax.config.jax_use_simplified_jaxpr_constants and "ROOT" not in sys.modules
    options = xla_client.CompileOptions()
    build_options = options.executable_build_options
    debug_options = build_options.debug_options
    assert debug_options.xla_disable_hlo_passes == ""
    jax.config.update("jax_enable_compilation_cache", False)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    specification = dict(source=model["source"], core_sha256=core_digest(), export_sha256=digest(export))
    compiler_gate = compiler_validation(args.model, specification)
    assert compiler_gate["jax_version"] == jax.__version__ and compiler_gate["jaxlib_version"] == jaxlib.__version__
    frozen = {p: digest(p) for p in ("scripts/fit_bounded_cpu.py",
        "scripts/resident_value_spooled_gradient.py", "scripts/fit_exported_likelihood.py",
        "scripts/hhvv_export_cache.py", "scripts/spooled_executable.py", "scripts/scoped_executable.py",
        "scripts/compare_gpu_results.py", "scripts/gpu_gradient_evidence.py",
        "scripts/reconcile_gpu_gradient_references.py", "scripts/compare_gpu_random_gradients.py",
        "scripts/gpu_random_refinements.py", "scripts/fd_reference.py")}
    mapping_limit = int(Path("/proc/sys/vm/max_map_count").read_text())
    report = dict(status="running", stage="load numerical export", model=args.model, backend="cpu",
        host=platform.node(), workdir=str(Path.cwd()), python=platform.python_version(),
        jax_version=jax.__version__, jaxlib_version=jaxlib.__version__, iminuit_version=iminuit.__version__,
        float64=True, devices=[str(d) for d in jax.devices()], cpu_affinity=sorted(os.sched_getaffinity(0)),
        lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"), source=model["source"],
        source_sha256=specification["core_sha256"], export_sha256=specification["export_sha256"],
        manifest_sha256=digest(manifest_path), runner_sha256=digest(__file__), frozen_source_files=frozen,
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        compiler_validation=compiler_gate, gradient_compiler_options=NO_FUSION_OPTIONS,
        value_compiler_options={}, simplified_jaxpr_constants=False, xla_flags=os.environ.get("XLA_FLAGS"),
        optimizer=dict(errordef=.5, tolerance=1e-4, strategy=args.strategy, max_calls=args.max_calls,
                       migrad_iterate=args.migrad_iterate, use_simplex=not args.no_simplex,
                       objective="canonical NLL(theta) - canonical NLL(checked initial point)",
                       gradient="full canonical NLL gradient; the constant offset has zero derivative"),
        covariance_requested=False, memory_policy="resident_value_spooled_gradient",
        rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        mapping_limit=mapping_limit, mapping_guard=mapping_limit-4096,
        initial_check=str(args.initial_check) if args.initial_check else None,
        initial_check_sha256=digest(args.initial_check) if args.initial_check else None,
        scope="Local conditional minimum only. NLL callbacks use resident kernels; gradient callbacks load every gradient term one at a time. All 26 saved points require unchanged Combine checks before minimization. No AD covariance, full fit acceptance, lxplus timing or GPU speedup claim.",
        elapsed_scope="After CLI/output checks; includes device initialization, model loading, all preparation/validation, minimization, final derivative and cleanup. Python startup and preceding imports excluded.")
    lock, stop = threading.RLock(), threading.Event()
    evaluator = None
    original_stdout = sys.stdout
    best = np.inf

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                mapping_count=len(Path("/proc/self/maps").read_text().splitlines()))
            if evaluator is not None:
                report["evaluation_counters"] = evaluator.snapshot()
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = started
        while not stop.wait(.25):
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            maps = len(Path("/proc/self/maps").read_text().splitlines())
            reason = ("RSS" if memory > args.rss_limit_mib else "mappings" if maps >= report["mapping_guard"]
                      else "wall time" if time.perf_counter()-started > args.wall_seconds else None)
            if reason:
                with lock:
                    if report["status"] != "running":
                        return
                    report.update(status="resource_limit", exception=f"Configured {reason} guard reached")
                    save()
                os._exit(86)
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], report["elapsed_seconds"], report["peak_rss_mib"],
                      report.get("evaluation_counters"), flush=True, file=original_stdout)
                last = time.perf_counter()

    def progress(event):
        with lock:
            report.update(stage=f"{event['operation']} {event['index']+1}/{event['terms']} {event['name']}",
                          preparation_progress=event)
            save()

    def observe(kind, point, value):
        nonlocal best
        with lock:
            report["stage"] = ("minimize / load full gradient" if kind == "gradient_start"
                               else f"minimize / completed {kind} call")
            if kind == "value" and np.isfinite(value) and value < best:
                best = value
                report.update(best_seen_nll=value, best_seen_values=point.tolist())
            if kind == "gradient_start":
                save()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        np.testing.assert_array_equal(likelihood.initial, model["reference"]["cases"][0]["theta"])
        if "parameter_names" in model["reference"]:
            np.testing.assert_array_equal(likelihood.parameter_names, model["reference"]["parameter_names"])
        point = likelihood.initial.copy()
        expected_initial = float(model["reference"]["cases"][0]["combine_nll"])
        if args.initial_check:
            point, expected_initial = checked_initial(likelihood, model["source"], manifest, args.initial_check)
        with lock:
            report.update(parameter_names=list(likelihood.parameter_names), initial=point.tolist(),
                discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
                fixed_bounds=[p.name for p in likelihood.parameters if p.minimum == p.maximum])
            save()
        reference_points = np.asarray([c["theta"] for c in model["reference"]["cases"]])
        evaluator = ResidentValueSpooledGradient(likelihood, directory,
            gradient_compiler_options=NO_FUSION_OPTIONS, on_progress=progress)
        begin = time.perf_counter()
        arrays = evaluator.prepare(reference_points)
        preparation_seconds = time.perf_counter()-begin
        checks = []
        for i, case in enumerate(model["reference"]["cases"]):
            check = dict(nll_absolute_error=abs(float(arrays["validation_values"][i])-case["combine_nll"]),
                         value_gradient_nll_absolute_error=abs(float(arrays["validation_vg_values"][i])-case["combine_nll"]))
            if "combine_fd_gradient" in case:
                reference = np.asarray(case["combine_fd_gradient"])
                check["gradient_max_relative_error"] = float(np.max(
                    abs(arrays["validation_gradients"][i]-reference)/np.maximum(abs(reference), 1e-8), initial=0.))
            checks.append(check)
        passed = all(c["nll_absolute_error"] < 1e-6 and c["value_gradient_nll_absolute_error"] < 1e-6
                     and c.get("gradient_max_relative_error", 0.) < 1e-4 for c in checks)
        validation_arrays.parent.mkdir(parents=True, exist_ok=True)
        temporary = validation_arrays.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **arrays, parameter_names=np.asarray(likelihood.parameter_names),
            point=reference_points[0], value=arrays["validation_values"][0], vg_value=arrays["validation_vg_values"][0],
            gradient=arrays["validation_gradients"][0], validation_points=reference_points,
            validation_labels=np.asarray([c.get("label", f"case_{i}") for i, c in enumerate(model["reference"]["cases"])]))
        temporary.replace(validation_arrays)
        checkpoint = dict(status=("computed" if args.model == "hhvv" else "passed") if passed else "reference_failed",
            model=args.model, backend="cpu", source=model["source"], source_sha256=core_digest(),
            host=platform.node(), jax_version=jax.__version__, jaxlib_version=jaxlib.__version__, float64=True,
            parameters=len(likelihood.parameter_names), parameter_names=list(likelihood.parameter_names),
            terms=len(evaluator.local.terms), completed_operations=["value", "value_and_grad"],
            arrays=str(validation_arrays), arrays_sha256=digest(validation_arrays), reference_validation=checks,
            nll=float(arrays["validation_values"][0]), value_gradient_nll=float(arrays["validation_vg_values"][0]),
            export_sha256=digest(export), reference_manifest=str(manifest_path),
            reference_manifest_sha256=digest(manifest_path), runner_sha256=digest(__file__),
            helper_sha256=digest("scripts/resident_value_spooled_gradient.py"),
            gradient_compiler_options=NO_FUSION_OPTIONS, compiler_validation=compiler_gate,
            preparation_wall_seconds=preparation_seconds,
            value_preparation=evaluator.value_preparation, gradient_preparation=evaluator.gradient_preparation,
            scope="Complete 26-point resident-NLL and loaded-gradient validation, assembled in original term order. Preparation wall includes all reference-batch evaluations and cleanup; this supplies no standalone full-gradient call latency.")
        temporary = checkpoint_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(checkpoint, indent=2, allow_nan=False)+"\n")
        temporary.replace(checkpoint_path)
        with lock:
            report.update(stage="check gradient reference evidence", gradient_checkpoint=str(checkpoint_path),
                          gradient_checkpoint_sha256=digest(checkpoint_path))
            save()
        assert passed, "Saved reference values/gradients failed"
        if args.model == "hhvv":
            run_checker("reconcile_gpu_gradient_references", [str(checkpoint_path), "--directions-reference",
                "reports/gpu_combine_directions_refined.json"],
                checkpoint_path.with_name(checkpoint_path.stem+"_resolved_gradient.log"))
            run_checker("compare_gpu_random_gradients", [str(checkpoint_path),
                "--reference", "reports/gpu_combine_random_gradient_extended.json",
                "--stable-reference", "reports/gpu_random_stable_fd_corrected.json",
                "--refined-full-nll-reference", "reports/gpu_random_gradient_failure_fd.json",
                "--refined-intensity-reference", "reports/gpu_random_intensity_fd_dense_local.json"],
                checkpoint_path.with_name(checkpoint_path.stem+"_combine_random_gradients_refined.log"))
            from compare_gpu_results import verified_refinement
            from gpu_gradient_evidence import random_validation
            nominal = verified_refinement(checkpoint_path, checkpoint)
            random = random_validation(checkpoint_path, checkpoint, digest(checkpoint_path), refined=True)
            assert nominal["status"] == nominal["directions_status"] == random["status"] == "passed"
            with lock:
                report["gradient_reference_evidence"] = dict(nominal=nominal, random=random)
                save()
        assert core_digest() == report["source_sha256"]
        assert all(digest(path) == checksum for path, checksum in frozen.items()), "Source changed before fit"
        assert "ROOT" not in sys.modules
        with lock:
            report["root_imported"] = False
        if args.validate_only:
            with lock:
                report.update(status="validated", stage="finished")
                save()
        else:
            with lock:
                report["stage"] = "minimize conditional likelihood"
                save()
            result = fit_separate_calls(likelihood, evaluator, initial=point, max_calls=args.max_calls,
                strategy=args.strategy, tolerance=1e-4, expected_initial_nll=expected_initial, on_evaluation=observe,
                migrad_iterate=args.migrad_iterate, use_simplex=not args.no_simplex)
            temporary = fit_arrays.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, parameter_names=np.asarray(likelihood.parameter_names),
                initial=point, values=result["values"], gradient=result["gradient"],
                hessian=np.empty((0, 0)), covariance=np.empty((0, 0)))
            temporary.replace(fit_arrays)
            with lock:
                final = {k: (v if np.isfinite(v) else None) if isinstance(v, (float, np.floating)) else v
                         for k, v in result.items() if k not in ("values", "gradient", "timings")}
                final["values"] = [float(v) if np.isfinite(v) else None for v in result["values"]]
                report.update(status="minimum_computed" if result["valid_minimum"] else "fit_incomplete",
                    stage="finished", arrays=str(fit_arrays), arrays_sha256=digest(fit_arrays),
                    final=final, initial_combine_nll_absolute_error=result["initial_reference_absolute_error"],
                    timings=result["timings"], root_imported="ROOT" in sys.modules)
                assert not report["root_imported"]
                save()
    except BaseException:
        with lock:
            report.update(status="reference_failed" if report["stage"].startswith("check gradient") else "failed",
                          exception=traceback.format_exc())
            save()
        raise
    finally:
        if evaluator is not None:
            evaluator.close()
        stop.set()
        watcher.join()
        save()
    print(args.output, report["status"], report["evaluation_counters"], flush=True)
    raise SystemExit(0 if report["status"] in ("validated", "minimum_computed") else 1)


if __name__ == "__main__":
    main()
