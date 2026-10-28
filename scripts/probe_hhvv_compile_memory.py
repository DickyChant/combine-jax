"""Check local Hessian finiteness and compiler-cache retention on two HHVV terms."""

import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import re
import threading
import time
import traceback

import jax
import numpy as np

from hhvv_export_cache import load_export


def rss():
    return int(next(s.split()[1] for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmRSS:")))/1024


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hessian-chunk-size", type=int)
    parser.add_argument("--rss-limit-mib", type=float, default=8000.)
    parser.add_argument("--compile-cores", type=int, default=7)
    parser.add_argument("--report-tag", choices=("", "serial", "serial_verified", "serial_verified_full", "pass_trace", "no_fusion_verified", "no_constant_folding", "no_constant_folding_verified", "backend_opt0", "backend_opt0_verified", "bbtt"), default="")
    parser.add_argument("--term-indices", type=int, nargs="+", default=(2,21))
    parser.add_argument("--expect-optimization-level", type=int, choices=(0,1,2,3))
    parser.add_argument("--expect-disabled-passes")
    parser.add_argument("--expect-codegen-split-count", type=int)
    arguments = parser.parse_args()
    if arguments.rss_limit_mib <= 0 or not 1 <= arguments.compile_cores <= 7 or (arguments.hessian_chunk_size is not None and arguments.hessian_chunk_size < 1):
        parser.error("Limits and chunk sizes must be positive")
    available = sorted(os.sched_getaffinity(0))
    assert len(available) >= 2
    os.sched_setaffinity(0,available[1:1+arguments.compile_cores])  # Leave the reference fit's first core free.
    jax.config.update("jax_enable_compilation_cache",False)
    started = time.perf_counter()
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    source = manifest["models"]["hhvv"]["source"]
    report = dict(status="running",source=source,jax_version=jax.__version__,float64=jax.config.jax_enable_x64,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),stage="setup",cases=[],
                  hessian_chunk_size=arguments.hessian_chunk_size,rss_limit_mib=arguments.rss_limit_mib,
                  term_indices=list(arguments.term_indices),
                  xla_flags=os.environ.get("XLA_FLAGS"),persistent_cache_enabled=jax.config.jax_enable_compilation_cache,
                  compiler_logging={key:os.environ.get(key) for key in ("TF_CPP_MIN_LOG_LEVEL","TF_CPP_VMODULE")},
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes() for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
                  scope="Selected local CPU Hessian blocks and cache lifetime only; no full Hessian, GPU or isolated performance claim")
    suffix = "" if arguments.hessian_chunk_size is None else "_chunk"+str(arguments.hessian_chunk_size)
    suffix += "_"+arguments.report_tag if arguments.report_tag else ""
    path = Path("reports")/("hhvv_compile_memory"+suffix+".json")
    if path.exists():
        raise FileExistsError("Preserve previous compilation evidence")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")

    stop = threading.Event()

    def watch():
        last_message = started
        while not stop.wait(.1):
            now = time.perf_counter()
            if now-last_message > 20:
                print(report["stage"],"elapsed",now-started,"RSS MiB",rss(),flush=True)
                last_message = now
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > arguments.rss_limit_mib:
                report.update(status="resource_limit",exception=f"{arguments.rss_limit_mib:,.0f} MiB configured probe guard exceeded")
                save()
                os._exit(86)

    monitor = threading.Thread(target=watch,daemon=True)
    monitor.start()
    try:
        assert report["float64"] and all(d.platform == "cpu" for d in jax.devices())
        from jaxlib import xla_client
        # Retain owners while inspecting nested C++ references.
        options = xla_client.CompileOptions()
        build_options = options.executable_build_options
        debug_options = build_options.debug_options
        report["effective_xla_options"] = dict(backend_optimization_level=debug_options.xla_backend_optimization_level,
                                               disabled_hlo_passes=debug_options.xla_disable_hlo_passes)
        # This field has no Python property in local jaxlib 0.12. Its protobuf
        # debug text still reports the effective parsed value explicitly.
        debug_text = str(debug_options)
        match = re.search(r"^xla_cpu_parallel_codegen_split_count: (\d+)$", debug_text, re.M)
        if arguments.expect_codegen_split_count is not None:
            assert match is not None, "Installed jaxlib does not expose the codegen option in its debug text"
            assert int(match[1]) == arguments.expect_codegen_split_count
        report["effective_xla_options"]["cpu_parallel_codegen_split_count"] = int(match[1]) if match else None
        report["debug_options_text"] = debug_text
        if arguments.expect_optimization_level is not None:
            assert report["effective_xla_options"]["backend_optimization_level"] == arguments.expect_optimization_level
        if arguments.expect_disabled_passes is not None:
            assert report["effective_xla_options"]["disabled_hlo_passes"] == arguments.expect_disabled_passes
        likelihood = load_export("build/gpu_exports/hhvv.pkl",source)
        evaluator = likelihood.local_evaluator(constraint_batch_size=32,hessian_chunk_size=arguments.hessian_chunk_size)
        all_terms = evaluator.terms
        assert all(0 <= index < len(all_terms) for index in arguments.term_indices)
        for index in arguments.term_indices:
            term = all_terms[index]
            point = likelihood.initial[term.indices]
            report["stage"] = "lower "+term.name
            save()
            begin = time.perf_counter()
            operation = "hessian" if arguments.hessian_chunk_size is None else "hessian_columns"
            original = getattr(term,operation)
            inputs = (point,) if arguments.hessian_chunk_size is None else (
                point,np.zeros((min(len(point),arguments.hessian_chunk_size),len(point))))
            lowered = original.lower(*inputs)
            report["active_lowering_seconds"] = time.perf_counter()-begin
            report["active_rss_after_lowering_mib"] = rss()
            report["stage"] = "compile "+term.name
            save()
            compile_begin = time.perf_counter()
            compiled = lowered.compile()
            compile_seconds = time.perf_counter()-compile_begin
            setattr(term,operation,compiled)
            del original,lowered
            evaluator.terms = (term,)

            def execute():
                if arguments.hessian_chunk_size is None:
                    return np.asarray(compiled(point))
                return evaluator.hessian(likelihood.initial)[np.ix_(term.indices,term.indices)]

            before = execute()
            assert np.isfinite(before).all()
            case = dict(name=term.name,parameters=len(point),shape=list(before.shape),
                        lowering_seconds=report["active_lowering_seconds"],compile_seconds=compile_seconds,
                        preparation_and_first_call_seconds=time.perf_counter()-begin,
                        hessian_max_asymmetry=float(abs(before-before.T).max()),rss_before_purge_mib=rss())
            jax.clear_caches(); gc.collect(); ctypes.CDLL(None).malloc_trim(0)
            case["rss_with_compiled_retained_after_purge_mib"] = rss()
            after = execute()
            np.testing.assert_array_equal(before,after)
            case["retained_compiled_bitwise_unchanged"] = True
            setattr(term,operation,None)
            del compiled,before,after,execute
            jax.clear_caches(); gc.collect(); ctypes.CDLL(None).malloc_trim(0)
            case["rss_after_compiled_discard_mib"] = rss()
            report["cases"].append(case)
            save()
            print(case,flush=True)
        report.update(status="passed",stage="finished")
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set(); monitor.join()


if __name__ == "__main__":
    main()
