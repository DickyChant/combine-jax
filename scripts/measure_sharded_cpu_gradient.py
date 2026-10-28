"""Measure and validate a full CPU value/gradient across retained worker kernels."""

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import numpy as np

from sharded_cpu_evaluator import ShardedCpuEvaluator, NO_FUSION_OPTIONS, core_digest, digest


def compiler_validation(model, specification):
    """Require the saved option diagnostic before a whole-model measurement."""
    path = Path(f"reports/gpu_local_{model}_gradient_compiler_no_fusion.json")
    report = json.loads(path.read_bytes())
    assert report["status"] == "passed" and report["model"] == model
    assert report["compiler_options"] == NO_FUSION_OPTIONS
    for key in ("source", "export_sha256"):
        assert report[key] == specification[key]
    assert report["source_sha256"] == specification["core_sha256"]
    assert report["runner_sha256"] == digest("scripts/probe_cpu_gradient_compiler_options.py")
    assert report["manifest_sha256"] == digest("build/gpu_exports/manifest_random.json")
    assert report["reference_sha256"] == digest(report["reference"])
    assert report["arrays_sha256"] == digest(report["arrays"])
    reference = json.loads(Path(report["reference"]).read_bytes())
    assert reference["status"] == "computed" and reference["compiler_options"] == {}
    assert reference["arrays_sha256"] == digest(reference["arrays"])
    assert len(report["cases"]) == len(report["selected_terms"]) > 0
    assert all(c["comparison_passed"] and c["value_max_absolute_error"] < 1e-8
               and c["gradient_max_scaled_error"] < 1e-10 for c in report["cases"])
    return dict(report=str(path), report_sha256=digest(path), arrays_sha256=report["arrays_sha256"],
                jax_version=report["jax_version"], jaxlib_version=report["jaxlib_version"],
                scope="Selected-term compiler diagnostic only; full gradients still require the unchanged Combine references.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--rss-limit-mib", type=float, default=10500.)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    parser.add_argument("--cpu-no-fusion", action="store_true")
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json":
        parser.error("Use a fresh reports/*.json output")
    artifact = Path("build/gpu_results")/(args.output.stem+".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError("Preserve earlier sharded measurements")
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    specification = dict(export=str(export), export_sha256=digest(export),
                         source=model["source"], core_sha256=core_digest())
    validation = compiler_validation(args.model, specification) if args.cpu_no_fusion else None
    started = time.perf_counter()
    report = dict(status="running", stage="initialize spawned workers", model=args.model,
        backend="cpu", host=platform.node(), source=model["source"], source_sha256=core_digest(),
        runner_sha256=digest(__file__), helper_sha256=digest("scripts/sharded_cpu_evaluator.py"),
        export_sha256=digest(export), reference_manifest=str(manifest_path),
        reference_manifest_sha256=digest(manifest_path),
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        xla_flags=os.environ.get("XLA_FLAGS"), workers_requested=args.workers,
        gradient_compiler_options=dict(NO_FUSION_OPTIONS) if args.cpu_no_fusion else {},
        compiler_validation=validation,
        aggregate_rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        scope="Experimental local full CPU value/gradient with separately retained executables; every term is evaluated and accumulated in original global order. No Hessian, covariance, lxplus timing or GPU speedup acceptance.",
        value_field_scope="All NLL fields come from value-and-gradient calls. There is no separately timed NLL-only operation.",
        memory_scope="Sampled sum of current RSS for this parent and all descendants, including workers and the Python process tracker; shared resident pages may be counted repeatedly.",
        elapsed_scope="Begins after CLI and manifest/export hash checks; includes worker startup, compilation, execution, validation and cleanup. Excludes Python startup and preceding module imports.",
        completed_operations=[], reference_validation=[])
    pool = None
    last_print = started

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        if pool is not None:
            report["resource_snapshot"] = pool.snapshot()
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    def progress(snapshot):
        nonlocal last_print
        report["resource_snapshot"] = snapshot
        report["stage"] = "compile full gradient"
        save()
        if time.perf_counter()-last_print > 20:
            done = {worker: record["index"] for worker, record in snapshot["progress"].items()}
            print(report["stage"], done, report["elapsed_seconds"],
                  snapshot["aggregate_rss_peak_mib"], flush=True)
            last_print = time.perf_counter()

    save()
    try:
        pool = ShardedCpuEvaluator(specification, workers=args.workers,
                                  rss_limit_mib=args.rss_limit_mib, wall_seconds=args.wall_seconds,
                                  on_progress=progress, cpu_no_fusion=args.cpu_no_fusion)
        metadata = pool.worker_metadata[0]
        assert metadata["helper_sha256"] == report["helper_sha256"]
        assert metadata["gradient_compiler_options"] == report["gradient_compiler_options"]
        if validation:
            for key in ("jax_version", "jaxlib_version"):
                assert metadata[key] == validation[key]
        point = pool.initial
        reference = model["reference"]
        np.testing.assert_array_equal(point, reference["cases"][0]["theta"])
        if "parameter_names" in reference:
            np.testing.assert_array_equal(pool.parameter_names, reference["parameter_names"])
        report.update(parameters=len(point), terms=len(pool.term_metadata), float64=True,
            jax_version=metadata["jax_version"], jaxlib_version=metadata["jaxlib_version"],
            devices=metadata["devices"], cpu_affinity=pool.cores, parameter_names=list(pool.parameter_names),
            term_partition=pool.term_metadata,
            worker_metadata=[{k: v for k, v in worker.items() if k not in
                              ("parameters", "parameter_names", "initial", "terms")}
                             for worker in pool.worker_metadata])
        begin = time.perf_counter()
        preparation = pool.prepare()
        preparation_wall = time.perf_counter()-begin
        report.update(stage="execute standalone value/gradient calls", preparation=preparation)
        save()
        samples = []
        nominal = None
        for i in range(6):
            begin = time.perf_counter()
            value, gradient = pool.value_and_grad(point)
            samples.append(time.perf_counter()-begin)
            assert np.isfinite(value) and np.isfinite(gradient).all()
            if nominal is None:
                nominal = value, gradient
            else:
                np.testing.assert_equal(value, nominal[0])
                np.testing.assert_array_equal(gradient, nominal[1])
        report.update(stage="validate all reference points",
            timings=dict(value_and_grad=dict(total_preparation_seconds=preparation_wall,
                lowering_sum_seconds=sum(r["lowering_seconds"] for r in preparation),
                compilation_sum_seconds=sum(r["compile_seconds"] for r in preparation),
                cache_release_sum_seconds=sum(r["release_seconds"] for r in preparation),
                compilation_scope="Sums of per-kernel wall durations; workers compile concurrently, so these sums are not elapsed preparation time. Preparation wall time includes parent progress reporting.",
                first_execution_seconds=samples[0], warm_seconds=samples[1:],
                median_warm_seconds=float(np.median(samples[1:])))))
        save()
        values, gradients = [], []
        for index, case in enumerate(reference["cases"]):
            value, gradient = pool.value_and_grad(case["theta"])
            assert np.isfinite(value) and np.isfinite(gradient).all()
            values.append(value)
            gradients.append(gradient)
            check = dict(nll_absolute_error=abs(value-case["combine_nll"]),
                         value_gradient_nll_absolute_error=abs(value-case["combine_nll"]))
            if "combine_fd_gradient" in case:
                expected = np.asarray(case["combine_fd_gradient"])
                check["gradient_max_relative_error"] = float(
                    (abs(gradient-expected)/np.maximum(abs(expected), 1e-8)).max(initial=0.))
            report["reference_validation"].append(check)
            report["stage"] = f"validate reference {index+1}/{len(reference['cases'])}"
            save()
        np.testing.assert_equal(values[0], nominal[0])
        np.testing.assert_array_equal(gradients[0], nominal[1])
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, parameter_names=np.asarray(pool.parameter_names), point=point,
            value=np.asarray(nominal[0]), vg_value=np.asarray(nominal[0]), gradient=nominal[1],
            validation_points=np.asarray([c["theta"] for c in reference["cases"]]),
            validation_labels=np.asarray([c.get("label", f"case_{i}") for i, c in enumerate(reference["cases"])]),
            validation_values=np.asarray(values), validation_vg_values=np.asarray(values),
            validation_gradients=np.asarray(gradients))
        temporary.replace(artifact)
        good = all(c["nll_absolute_error"] < 1e-6 and c.get("gradient_max_relative_error", 0.) < 1e-4
                   for c in report["reference_validation"])
        all_gradient_references = all("combine_fd_gradient" in c for c in reference["cases"])
        report.update(status=("passed" if all_gradient_references else "computed") if good else "reference_failed",
                      stage="finished", completed_operations=["value_and_grad"],
                      arrays=str(artifact), arrays_sha256=digest(artifact),
                      nll=nominal[0], value_gradient_nll=nominal[0],
                      nominal_value_absolute_error=abs(nominal[0]-reference["cases"][0]["combine_nll"]),
                      all_manifest_gradient_references_available=all_gradient_references,
                      remaining_validation=None if all_gradient_references else
                      "Apply the unchanged frozen nominal/directional and 25-point Combine gradient references")
        save()
    except BaseException as error:
        report.update(status="resource_limit" if isinstance(error, (MemoryError, TimeoutError)) else "failed",
                      exception=traceback.format_exc())
        save()
        raise
    finally:
        if pool is not None:
            pool.close(force=report["status"] not in ("passed", "computed"))
            report["worker_cleanup"] = pool.cleanup
        save()
    print(args.output, report["status"], report.get("resource_snapshot"), flush=True)
    raise SystemExit(0 if report["status"] in ("passed", "computed") else 1)


if __name__ == "__main__":
    main()
