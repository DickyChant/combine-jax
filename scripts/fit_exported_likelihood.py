"""Guarded CPU/GPU fit of a trusted numerical export, without importing ROOT.

Numerical agreement and fit timing comparisons require separate reference checks.
The package's fit driver supplies Minuit gradients and full AD covariance.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import subprocess
import sys
import threading
import time
import traceback

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_initial(likelihood, source, manifest, path):
    """Use only a separately checked, source-bound Combine candidate."""
    checked = json.loads(path.read_text())
    assert checked["status"] == "candidate_checked" and checked["source"] == source
    assert digest(checked["parent"]) == checked["parent_sha256"]
    assert checked["combine_version"] == manifest["combine_version"]
    assert checked["root_version"] == manifest["root_version"]
    assert checked["discrete_choices"] == json.loads(json.dumps(likelihood.discrete_choices))
    assert checked["channel_masks"] == json.loads(json.dumps(likelihood.mask_configuration))
    np.testing.assert_array_equal(checked["parameter_names"], likelihood.parameter_names)
    point = np.asarray(checked["point"], dtype=np.float64)
    assert point.shape == likelihood.initial.shape and np.isfinite(point).all()
    for value, parameter, nominal in zip(point, likelihood.parameters, likelihood.initial, strict=True):
        assert parameter.minimum <= value <= parameter.maximum
        if parameter.minimum == parameter.maximum:
            assert value == nominal
    for key in ("saved_zero_point_absolute_error", "canonical_offset_absolute_error",
                "canonical_restoration_absolute_error"):
        assert checked[key] < 1e-6
    assert all(p["roofit_evaluation_errors"] == 0 for p in checked["probes"])
    return point, float(checked["canonical_candidate_nll"])


class ObservedEvaluator:
    """Observe the existing evaluator without changing its numerical methods."""

    def __init__(self, evaluator, report, lock, save, expected_initial_nll):
        self.base, self.report, self.lock, self.save = evaluator, report, lock, save
        self.parameter_names = evaluator.parameter_names
        self.expected_initial_nll = expected_initial_nll
        self.calls = 0
        self.best = np.inf

    def prepare(self, *args, **kwargs):
        with self.lock:
            self.report["stage"] = "prepare retained gradient kernels"
            self.save()
        result = self.base.prepare(*args, **kwargs)
        with self.lock:
            self.report["gradient_preparation"] = result
            self.save()
        return result

    def value_and_grad(self, point):
        with self.lock:
            self.report["stage"] = "evaluate value and gradient / minimize"
        begin = time.perf_counter()
        value, gradient = self.base.value_and_grad(point)
        elapsed = time.perf_counter()-begin
        self.calls += 1
        finite = bool(np.isfinite(point).all() and np.isfinite(value) and np.isfinite(gradient).all())
        with self.lock:
            self.report.update(evaluator_calls=self.calls,
                evaluator_seconds=self.report.get("evaluator_seconds", 0.)+elapsed)
            if not finite:
                self.report["nonfinite_evaluator_calls"] = self.report.get("nonfinite_evaluator_calls", 0)+1
            if finite and value < self.best:
                self.best = float(value)
                self.report.update(best_seen_nll=self.best, best_seen_values=np.asarray(point).tolist())
            if self.calls == 1:
                error = abs(float(value)-self.expected_initial_nll)
                self.report["initial_combine_nll_absolute_error"] = error
                self.save()
                assert finite and error < 1e-6, "Starting point disagrees with its Combine reference"
        return value, gradient

    def clear_cache(self, **kwargs):
        with self.lock:
            self.report["stage"] = "release gradient kernels before Hessian"
            self.save()
        return self.base.clear_cache(**kwargs)

    def hessian(self, point, **kwargs):
        with self.lock:
            self.report.update(stage="stream full AD Hessian",
                               curvature_point=np.asarray(point).tolist())
            self.save()
        return self.base.hessian(point, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--report-tag", required=True)
    parser.add_argument("--initial-check", type=Path)
    parser.add_argument("--strategy", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--max-calls", type=int, default=50000)
    parser.add_argument("--wall-seconds", type=float, default=21600)
    parser.add_argument("--rss-limit-mib", type=float, default=11000)
    parser.add_argument("--cores", type=int, default=8)
    parser.add_argument("--core-offset", type=int, default=0)
    parser.add_argument("--no-covariance", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.report_tag):
        parser.error("Use a lowercase alphanumeric/underscore report tag")
    if (args.max_calls < 1 or args.wall_seconds <= 0 or not 0 < args.rss_limit_mib <= 11000
            or not 1 <= args.cores <= 8 or args.core_offset < 0 or args.core_offset+args.cores > 8):
        parser.error("Require positive limits, RSS <= 11000 MiB and affinity within the first eight cores")
    stem = f"gpu_{args.model}_{args.backend}_fit_{args.report_tag}"
    output = Path("reports")/(stem+".json")
    artifact = Path("build/fit_results")/(stem+".npz")
    if output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous fit evidence")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        assert platform.node().split(".")[0] == "lxplus905"
        for policy in ("spooled", "scoped"):
            for phase in ("smoke", "hhvv"):
                batch_path = Path(f"reports/gpu_{policy}_{phase}_driver.json")
                if batch_path.exists() and json.loads(batch_path.read_text())["status"] == "running":
                    raise RuntimeError(
                        f"The pinned-node benchmark batch must record completion before a fit starts: {batch_path}")
    available = sorted(os.sched_getaffinity(0))
    selected = available[args.core_offset:args.core_offset+args.cores]
    assert len(selected) == args.cores
    os.sched_setaffinity(0, selected)
    os.environ["JAX_PLATFORMS"] = "cuda" if args.backend == "gpu" else "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    import jaxlib
    import iminuit
    from jaxlib import xla_client
    from combine_jax.fitting import fit
    from hhvv_export_cache import load_export

    devices = jax.devices()
    assert jax.default_backend() == args.backend and jax.config.jax_enable_x64
    if args.backend == "gpu":
        assert devices and all(d.platform == "gpu" for d in devices)
        assert Path.cwd() == Path("/tmp/sqian/combine-jax")
        assert platform.node().split(".")[0] == "lxplus905"
    assert "ROOT" not in sys.modules
    jax.config.update("jax_enable_compilation_cache", False)
    started = time.perf_counter()
    cpu_started = time.process_time()
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"][args.model]
    export = Path("build/gpu_exports")/model["cache"]
    likelihood = load_export(export, model["source"])
    point = likelihood.initial.copy()
    expected = float(model["reference"]["cases"][0]["combine_nll"])
    if args.initial_check:
        point, expected = checked_initial(likelihood, model["source"], manifest, args.initial_check)
    mapping_limit = int(Path("/proc/sys/vm/max_map_count").read_text())
    # These native views need their owners alive while fields are read.
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    debug_options = build_options.debug_options
    report = dict(status="running", stage="setup", model=args.model, backend=args.backend,
        host=platform.node(), workdir=str(Path.cwd()), python=platform.python_version(),
        jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        iminuit_version=iminuit.__version__, float64=True, devices=[str(d) for d in devices],
        lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"), cpu_affinity=selected,
        cpu_model=next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                        if line.startswith("model name")), None),
        source=model["source"], source_sha256=hashlib.sha256(b"".join(
            str(p).encode()+b"\0"+p.read_bytes() for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
        export_sha256=digest(export), manifest_sha256=digest(manifest_path),
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        runner_sha256=digest(__file__), export_loader_sha256=digest("scripts/hhvv_export_cache.py"),
        optimizer=dict(errordef=.5, tolerance=1e-4, strategy=args.strategy, max_calls=args.max_calls,
                       migrad_iterate=5, use_simplex=True),
        memory_policy="release", constraint_batch_size=32, hessian_chunk_size=4,
        covariance_requested=not args.no_covariance, rss_limit_mib=args.rss_limit_mib,
        wall_limit_seconds=args.wall_seconds, mapping_limit=mapping_limit,
        mapping_guard=mapping_limit-1024, persistent_cache_enabled=False,
        xla_flags=os.environ.get("XLA_FLAGS"),
        effective_xla_options=dict(disabled_hlo_passes=debug_options.xla_disable_hlo_passes,
            backend_optimization_level=debug_options.xla_backend_optimization_level),
        parameter_names=list(likelihood.parameter_names), initial=point.tolist(),
        discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
        fixed_bounds=[p.name for p in likelihood.parameters if p.minimum == p.maximum],
        initial_check=str(args.initial_check) if args.initial_check else None,
        initial_check_sha256=digest(args.initial_check) if args.initial_check else None,
        scope="Conditional continuous fit at exported category/mask choices; numerical computation only. Separate Combine checks and same-host matched-start measurements are required for agreement or speed claims.")
    monitor = None
    if args.backend == "gpu":
        from gpu_benchmark import MemoryMonitor
        report["gpu_metadata"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader"],
            text=True).strip()
        report["telemetry_source_sha256"] = digest("scripts/gpu_benchmark.py")
        monitor = MemoryMonitor()
    lock = threading.RLock()
    stop = threading.Event()
    artifact.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                cpu_seconds=time.process_time()-cpu_started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if monitor is not None:
                monitor.stage = report["stage"]
                report["gpu_memory"] = monitor.snapshot()
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def watch():
        last_save = started
        while not stop.wait(.25):
            elapsed = time.perf_counter()-started
            rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            mappings = len(Path("/proc/self/maps").read_text().splitlines())
            reason = ("RSS" if rss > args.rss_limit_mib else
                      "wall time" if elapsed > args.wall_seconds else
                      "mapping count" if mappings >= report["mapping_guard"] else None)
            if reason:
                with lock:
                    report.update(status="resource_limit", exception=f"Configured {reason} guard exceeded",
                                  mapping_count=mappings)
                    save()
                os._exit(86)
            if time.perf_counter()-last_save > 20:
                with lock:
                    report["mapping_count"] = mappings
                    save()
                    print(report["stage"], report.get("evaluator_calls", 0),
                          report.get("best_seen_nll"), report["elapsed_seconds"],
                          report["peak_rss_mib"], flush=True)
                last_save = time.perf_counter()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        evaluator = ObservedEvaluator(likelihood.local_evaluator(
            constraint_batch_size=32, hessian_chunk_size=4), report, lock, save, expected)
        result = fit(likelihood, initial=point, evaluator=evaluator, max_calls=args.max_calls,
                     tolerance=1e-4, strategy=args.strategy, covariance=not args.no_covariance,
                     memory_policy="release")
        assert "ROOT" not in sys.modules
        np.savez_compressed(artifact, parameter_names=np.asarray(result.parameter_names),
            initial=point, values=result.values, gradient=result.gradient, hessian=result.hessian,
            covariance=result.covariance if result.covariance is not None else np.empty((0, 0)))
        with lock:
            report.update(stage="finished", root_imported=False, arrays=str(artifact),
                arrays_sha256=digest(artifact), final=dict(
                    values=[float(x) if np.isfinite(x) else None for x in result.values],
                    nll=result.nll if np.isfinite(result.nll) else None,
                    valid_minimum=result.valid_minimum, edm=result.edm if np.isfinite(result.edm) else None,
                    covariance_status=result.covariance_status, fixed=result.fixed,
                    at_bounds=result.at_bounds, function_calls=result.function_calls,
                    gradient_calls=result.gradient_calls),
                timings=result.timings)
            complete = result.valid_minimum and (args.no_covariance or
                        result.covariance_status in ("positive_definite", "all_fixed"))
            report["status"] = ("minimum_computed" if args.no_covariance else "fit_computed") if complete else "fit_incomplete"
            save()
        print(report["status"], result.nll, result.edm, result.covariance_status, flush=True)
    except Exception:
        with lock:
            report.update(status="failed", exception=traceback.format_exc())
            save()
        raise
    finally:
        stop.set()
        watcher.join()
        if monitor is not None:
            monitor.close()


if __name__ == "__main__":
    main()
