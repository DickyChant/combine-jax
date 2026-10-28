"""Measure CPU executable mappings for selected HHVV value/gradient kernels."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import threading
import time
import traceback

import jax
from jaxlib import xla_client
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resources():
    status = Path("/proc/self/status").read_text().splitlines()
    return dict(rss_mib=int(next(s.split()[1] for s in status if s.startswith("VmRSS:")))/1024,
                mappings=len(Path("/proc/self/maps").read_text().splitlines()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-count", type=int, choices=(1, 32), required=True)
    parser.add_argument("--selection", choices=("largest_and_failure", "bbtt"), default="largest_and_failure")
    parser.add_argument("--rss-limit-mib", type=float, default=1000.)
    args = parser.parse_args()
    if not 0 < args.rss_limit_mib <= 1000:
        parser.error("This concurrent probe is limited to 1000 MiB")
    available = sorted(os.sched_getaffinity(0))
    assert len(available) >= 3
    os.sched_setaffinity(0, [available[2]])
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    options = xla_client.CompileOptions()
    build_options = options.executable_build_options
    debug_options = build_options.debug_options
    text = str(debug_options)
    match = re.search(r"^xla_cpu_parallel_codegen_split_count: (\d+)$", text, re.M)
    assert match and int(match[1]) == args.split_count
    assert debug_options.xla_disable_hlo_passes == "" and debug_options.xla_backend_optimization_level == 3
    suffix = "_bbtt" if args.selection == "bbtt" else ""
    output = Path(f"reports/hhvv_cpu_gradient_codegen_split{args.split_count}{suffix}.json")
    directory = Path(f"build/gpu_results/hhvv_cpu_gradient_codegen_split{args.split_count}{suffix}")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve previous code-generation evidence")
    directory.mkdir(parents=True)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"]["hhvv"]
    prior_path = Path("reports/gpu_hhvv_cpu_cachetrim.json")
    prior = json.loads(prior_path.read_text())
    assert prior["compile_progress"]["operation"] == "value_and_grad"
    started = time.perf_counter()
    report = dict(status="running", stage="setup", split_count=args.split_count,
        source=model["source"], jax_version=jax.__version__, float64=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), rss_limit_mib=args.rss_limit_mib,
        xla_flags=os.environ.get("XLA_FLAGS"), effective_xla_options=dict(
            cpu_parallel_codegen_split_count=int(match[1]), disabled_hlo_passes="",
            backend_optimization_level=debug_options.xla_backend_optimization_level),
        manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
        prior_cpu_report=str(prior_path), prior_cpu_report_sha256=digest(prior_path),
        runner_sha256=digest(__file__),
        source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
            for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
        scope="Selected local CPU value/gradient compilation, executable retention and mappings only; no full-model or lxplus/GPU performance acceptance",
        selection=("Largest parameter count, first bbtt term at index 21, and the previous lxplus CPU failure term"
                   if args.selection == "largest_and_failure" else
                   "First bbtt term at index 21, previous lxplus CPU failure term, and its following term"),
        cases=[])
    lock = threading.RLock()
    stop = threading.Event()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def monitor():
        last = started
        while not stop.wait(.1):
            if time.perf_counter()-last > 20:
                save()
                print(report["stage"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
                last = time.perf_counter()
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > args.rss_limit_mib:
                with lock:
                    report.update(status="resource_limit", exception="Configured own-RSS guard exceeded")
                    save()
                os._exit(86)

    save()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(Path("build/gpu_exports")/model["cache"], model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32)
        largest = max(range(len(evaluator.terms)), key=lambda i: len(evaluator.terms[i].indices))
        failed = prior["compile_progress"]["completed_terms"]
        assert prior["current_stage"].endswith(evaluator.terms[failed].name)
        selected = list(dict.fromkeys((largest, 21, failed) if args.selection == "largest_and_failure"
                                      else (21, failed, failed+1)))
        points = np.asarray([c["theta"] for c in model["reference"]["cases"][:2]])
        report["baseline"] = resources()
        kernels = []
        for index in selected:
            term = evaluator.terms[index]
            local_points = points[:, term.indices]
            record = dict(index=index, name=term.name, parameters=len(term.indices), status="running",
                          before_compile=resources())
            report["cases"].append(record)
            report["stage"] = "compile "+term.name
            save()
            begin = time.perf_counter()
            lowered = term.value_and_grad.lower(local_points[0])
            record["lowering_seconds"] = time.perf_counter()-begin
            begin = time.perf_counter()
            compiled = lowered.compile()
            record["compile_seconds"] = time.perf_counter()-begin
            kernels.append(compiled)
            del lowered
            values, gradients = zip(*(compiled(p) for p in local_points), strict=True)
            values, gradients = np.asarray(values), np.asarray(gradients)
            assert np.isfinite(values).all() and np.isfinite(gradients).all()
            record["after_compile"] = resources()
            release_compilation_cache()
            record["retained_after_purge"] = resources()
            actual = [compiled(p) for p in local_points]
            np.testing.assert_array_equal(values, [c[0] for c in actual])
            np.testing.assert_array_equal(gradients, [c[1] for c in actual])
            artifact = directory/f"{index}.npz"
            np.savez_compressed(artifact, parameter_names=np.asarray(term.parameter_names),
                                points=local_points, values=values, gradients=gradients)
            record.update(status="passed", arrays=str(artifact), arrays_sha256=digest(artifact),
                          repeat_after_purge_bitwise_equal=True)
            del compiled, actual
            print(term.name, record["retained_after_purge"], flush=True)
            save()
        report["all_retained"] = resources()
        kernels.clear()
        release_compilation_cache()
        report.update(status="passed", stage="finished", all_discarded=resources())
        save()
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()


if __name__ == "__main__":
    main()
