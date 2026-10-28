"""Fit an already validated numerical export with retained spawned CPU gradients.

This computes a conditional minimum only. Full fit-point AD curvature and a
separate Combine comparison remain necessary for fit/covariance acceptance.
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

from fit_exported_likelihood import ObservedEvaluator, checked_initial
from compare_gpu_results import verified_refinement
from gpu_gradient_evidence import random_validation
from sharded_cpu_evaluator import ShardedCpuEvaluator, NO_FUSION_OPTIONS, core_digest, digest


def check_gradient_evidence(path, model, manifest):
    """Bind a completed all-point validation to the exact evaluator and export."""
    report = json.loads(path.read_bytes())
    expected = manifest["models"][model]
    assert report["model"] == model and report["source"] == expected["source"]
    assert report["backend"] == "cpu" and report["float64"]
    assert report["gradient_compiler_options"] in ({}, NO_FUSION_OPTIONS)
    assert report["source_sha256"] == core_digest()
    assert report["helper_sha256"] == digest("scripts/sharded_cpu_evaluator.py")
    assert report["runner_sha256"] == digest("scripts/measure_sharded_cpu_gradient.py")
    assert report["export_sha256"] == digest(Path("build/gpu_exports")/expected["cache"])
    assert report["reference_manifest_sha256"] == digest("build/gpu_exports/manifest_random.json")
    assert report["completed_operations"] == ["value_and_grad"]
    assert report["status"] == ("computed" if model == "hhvv" else "passed")
    assert report["arrays_sha256"] == digest(report["arrays"])
    assert len(report["reference_validation"]) == len(expected["reference"]["cases"]) == 26
    assert all(c["nll_absolute_error"] < 1e-6 for c in report["reference_validation"])
    assert all(w["exit_code"] == 0 and not w["signals"] for w in report["worker_cleanup"])
    with np.load(report["arrays"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["validation_points"],
            [c["theta"] for c in expected["reference"]["cases"]])
        assert arrays["validation_gradients"].shape == (26, report["parameters"])
        assert arrays["validation_gradients"].dtype == np.float64
        assert np.isfinite(arrays["validation_gradients"]).all()
        values = arrays["validation_vg_values"]
        assert np.isfinite(values).all()
        assert np.max(abs(values-[c["combine_nll"] for c in expected["reference"]["cases"]])) < 1e-6
        if model != "hhvv":
            references = np.asarray([c["combine_fd_gradient"] for c in expected["reference"]["cases"]])
            assert np.max(abs(arrays["validation_gradients"]-references)/
                          np.maximum(abs(references), 1e-8)) < 1e-4
    provenance = {str(path): digest(path), report["arrays"]: report["arrays_sha256"]}
    if model == "hhvv":
        nominal = verified_refinement(path, report)
        random = random_validation(path, report, digest(path), refined=True)
        assert nominal["status"] == nominal["directions_status"] == random["status"] == "passed"
        provenance.update(nominal_validation=nominal, random_validation=random)
    return report, provenance


class ProgressEvaluator(ObservedEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_save = time.monotonic()

    def value_and_grad(self, point):
        result = super().value_and_grad(point)
        if time.monotonic()-self.last_save > 20:
            self.save()
            print(self.report["stage"], self.calls, self.report.get("best_seen_nll"),
                  self.report["elapsed_seconds"], flush=True)
            self.last_save = time.monotonic()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--validated-gradient", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--initial-check", type=Path)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--strategy", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--max-calls", type=int, default=50000)
    parser.add_argument("--rss-limit-mib", type=float, default=10500.)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    parser.add_argument("--cpu-no-fusion", action="store_true")
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json" or args.max_calls < 1:
        parser.error("Use a fresh reports/*.json output and a positive call limit")
    artifact = Path("build/fit_results")/(args.output.stem+".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError("Preserve existing fit evidence")
    # The helper independently enforces this before spawning numerical workers.
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        for policy in ("spooled", "scoped"):
            for phase in ("smoke", "hhvv"):
                path = Path(f"reports/gpu_{policy}_{phase}_driver.json")
                if path.exists() and json.loads(path.read_bytes())["status"] == "running":
                    raise RuntimeError("Finish the active benchmark before a fit")
    started = time.perf_counter()
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    evidence, provenance = check_gradient_evidence(args.validated_gradient, args.model, manifest)
    options = dict(NO_FUSION_OPTIONS) if args.cpu_no_fusion else {}
    assert evidence["gradient_compiler_options"] == options, "Fit and validated compiler options differ"
    model = manifest["models"][args.model]
    specification = dict(export=str(Path("build/gpu_exports")/model["cache"]),
        export_sha256=evidence["export_sha256"], source=model["source"], core_sha256=core_digest())
    report = dict(status="running", stage="initialize gradient workers", model=args.model,
        backend="cpu", host=platform.node(), workdir=str(Path.cwd()), python=platform.python_version(),
        float64=True, source=model["source"], source_sha256=core_digest(),
        export_sha256=evidence["export_sha256"], manifest_sha256=digest(manifest_path),
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        runner_sha256=digest(__file__), helper_sha256=digest("scripts/sharded_cpu_evaluator.py"),
        observer_sha256=digest("scripts/fit_exported_likelihood.py"),
        gradient_validation=provenance, workers_requested=args.workers,
        gradient_compiler_options=options,
        optimizer=dict(errordef=.5, tolerance=1e-4, strategy=args.strategy, max_calls=args.max_calls,
                       migrad_iterate=5, use_simplex=True),
        covariance_requested=False, memory_policy="spawned_retained", constraint_batch_size=32,
        aggregate_rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        resource_guard_scope="The evaluator checks aggregate parent/descendant RSS, worker mappings and wall time while waiting for worker replies and at each completed call. Optimizer-only intervals between calls are not sampled.",
        elapsed_scope="Begins after CLI/output/overlap checks, includes validation, worker startup, preparation, minimization and cleanup; excludes Python startup and preceding imports.",
        initial_check=str(args.initial_check) if args.initial_check else None,
        initial_check_sha256=digest(args.initial_check) if args.initial_check else None,
        scope="Local conditional continuous minimum only, at fixed exported categories and masks. No covariance, full fit acceptance, profile acceptance, lxplus timing or GPU speedup claim.")
    pool = None
    lock = threading.RLock()
    last_progress = started

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          parent_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if pool is not None:
                report["resource_snapshot"] = pool.snapshot()
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def progress(snapshot):
        nonlocal last_progress
        report.update(stage="compile full gradient", resource_snapshot=snapshot)
        save()
        if time.perf_counter()-last_progress > 20:
            print(report["stage"], {k: v["index"] for k, v in snapshot["progress"].items()},
                  report["elapsed_seconds"], snapshot["aggregate_rss_peak_mib"], flush=True)
            last_progress = time.perf_counter()

    save()
    try:
        pool = ShardedCpuEvaluator(specification, workers=args.workers, on_progress=progress,
            rss_limit_mib=args.rss_limit_mib, wall_seconds=args.wall_seconds,
            cpu_no_fusion=args.cpu_no_fusion)
        import iminuit
        from combine_jax.fitting import fit

        likelihood = pool.likelihood
        point = pool.initial.copy()
        expected = float(model["reference"]["cases"][0]["combine_nll"])
        if args.initial_check:
            point, expected = checked_initial(likelihood, model["source"], manifest, args.initial_check)
        np.testing.assert_array_equal(pool.parameter_names, evidence["parameter_names"])
        worker = pool.worker_metadata[0]
        assert worker["helper_sha256"] == report["helper_sha256"]
        assert worker["gradient_compiler_options"] == options
        assert worker["jax_version"] == evidence["jax_version"]
        assert worker["jaxlib_version"] == evidence["jaxlib_version"]
        report.update(jax_version=worker["jax_version"], jaxlib_version=worker["jaxlib_version"],
            iminuit_version=iminuit.__version__, devices=worker["devices"], cpu_affinity=pool.cores,
            parameter_names=list(pool.parameter_names), initial=point.tolist(),
            discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
            fixed_bounds=[p.name for p in likelihood.parameters if p.minimum == p.maximum])
        observed = ProgressEvaluator(pool, report, lock, save, expected)
        result = fit(likelihood, initial=point, evaluator=observed, max_calls=args.max_calls,
                     tolerance=1e-4, strategy=args.strategy, covariance=False, memory_policy="release")
        assert "ROOT" not in sys.modules
        artifact.parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, parameter_names=np.asarray(result.parameter_names),
            initial=point, values=result.values, gradient=result.gradient, hessian=result.hessian,
            covariance=np.empty((0, 0)))
        temporary.replace(artifact)
        report.update(status="minimum_computed" if result.valid_minimum else "fit_incomplete",
            stage="finished", root_imported=False, arrays=str(artifact), arrays_sha256=digest(artifact),
            final=dict(values=[float(x) if np.isfinite(x) else None for x in result.values],
                nll=result.nll if np.isfinite(result.nll) else None,
                valid_minimum=result.valid_minimum, edm=result.edm if np.isfinite(result.edm) else None,
                covariance_status=result.covariance_status, fixed=result.fixed, at_bounds=result.at_bounds,
                function_calls=result.function_calls, gradient_calls=result.gradient_calls),
            timings=result.timings)
        save()
    except BaseException as error:
        report.update(status="resource_limit" if isinstance(error, (MemoryError, TimeoutError)) else "failed",
                      exception=traceback.format_exc())
        save()
        raise
    finally:
        if pool is not None:
            pool.close(force=report["status"] not in ("minimum_computed", "fit_incomplete"))
            report["worker_cleanup"] = pool.cleanup
        save()
    print(args.output, report["status"], report["final"], flush=True)
    raise SystemExit(0 if report["status"] == "minimum_computed" else 1)


if __name__ == "__main__":
    main()
