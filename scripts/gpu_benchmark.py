"""Same-node CPU/GPU benchmark of complete exported likelihoods, without ROOT."""

import argparse
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import threading
import time
import traceback


def process_mapping_count():
    with Path("/proc/self/maps").open() as stream:
        return sum(1 for _ in stream)


class MemoryMonitor:
    """Sample NVML process memory; BFC allocation retention is part of the result."""

    class Memory(C.Structure):
        _fields_ = [(n, C.c_ulonglong) for n in ("total", "free", "used")]

    class Process(C.Structure):
        _fields_ = [("pid", C.c_uint), ("usedGpuMemory", C.c_ulonglong),
                    ("gpuInstanceId", C.c_uint), ("computeInstanceId", C.c_uint)]

    def __init__(self):
        self.nvml = C.CDLL("libnvidia-ml.so.1")
        self.check(self.nvml.nvmlInit_v2())
        self.handle = C.c_void_p()
        self.check(self.nvml.nvmlDeviceGetHandleByIndex_v2(C.c_uint(0), C.byref(self.handle)))
        self.process_peak = self.global_peak = self.samples = 0
        self.pid_seen = False
        self.mapping_peak = process_mapping_count()
        self.error = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.stage = "setup"
        self.started = time.perf_counter()
        self.sample()
        self.baseline_global = self.global_peak
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    @staticmethod
    def check(code):
        if code:
            raise RuntimeError(f"NVML call failed with status {code}")

    def sample(self):
        memory = self.Memory()
        self.check(self.nvml.nvmlDeviceGetMemoryInfo(self.handle, C.byref(memory)))
        count = C.c_uint(128)
        processes = (self.Process*128)()
        self.check(self.nvml.nvmlDeviceGetComputeRunningProcesses_v2(self.handle, C.byref(count), processes))
        used = next((p.usedGpuMemory for p in processes[:count.value] if p.pid == os.getpid()), None)
        with self.lock:
            self.global_peak = max(self.global_peak, memory.used)
            if used is not None:
                self.pid_seen = True
                self.process_peak = max(self.process_peak, used)
            self.samples += 1

    def run(self):
        last_message = self.started
        while not self.stop_event.wait(.05):
            try:
                self.sample()
                now = time.perf_counter()
                if (getattr(self, "memory_limit_callback", None) is not None
                        and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 11500):
                    self.memory_limit_callback()
                if now-last_message > 20:
                    mappings = process_mapping_count()
                    self.mapping_peak = max(self.mapping_peak, mappings)
                    print(f"{self.stage}: elapsed {now-self.started:.1f}s; GPU peak {self.process_peak/2**20:.1f} MiB; RSS peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MiB; mappings {mappings}", flush=True)
                    last_message = now
            except Exception as exc:
                self.error = str(exc)
                return

    def snapshot(self):
        self.sample()
        with self.lock:
            return dict(process_peak_mib=self.process_peak/2**20, process_seen=self.pid_seen,
                        device_baseline_used_mib=self.baseline_global/2**20,
                        device_peak_used_mib=self.global_peak/2**20, samples=self.samples,
                        sample_period_seconds=.05, error=self.error,
                        scope="Sampled NVML process resident memory, including retained allocator pools and compiled executables")

    def close(self):
        self.stop_event.set()
        self.thread.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--constraint-batch-size", type=int, default=1)
    parser.add_argument("--release-stages", action="store_true",
                        help="Release each operation's compiled kernels after timing and validation")
    parser.add_argument("--trim-compilation-cache", action="store_true",
                        help="Clear tracing/compiler caches after each term while retaining its explicit executable")
    parser.add_argument("--spool-hessian-executables", action="store_true",
                        help="Store Hessian executables on disk and load/release them for calls")
    parser.add_argument("--spool-value-gradient", action="store_true",
                        help="Also store value+gradient executables on disk, including loading/cleanup in timings")
    parser.add_argument("--hessian-chunk-size", type=int,
                        help="Compute every Hessian column in exact batches of this size")
    parser.add_argument("--scoped-loads", action="store_true",
                        help="Load each Hessian term once per full call; batch untimed gradient validation")
    parser.add_argument("--cpu-hessian-no-fusion", action="store_true",
                        help="Explicit per-Hessian CPU compiler option; all GPU kernels retain default options")
    parser.add_argument("--report-tag", default="", choices=("", "staged", "cachetrim", "spooled", "scoped"))
    parser.add_argument("--reference-manifest", type=Path, default=Path("build/gpu_exports/manifest.json"))
    arguments = parser.parse_args()
    if arguments.repeat < 1:
        parser.error("repeat must be positive")
    if arguments.constraint_batch_size < 1:
        parser.error("constraint-batch-size must be positive")
    if arguments.trim_compilation_cache != (arguments.report_tag in ("cachetrim", "spooled", "scoped")):
        parser.error("--trim-compilation-cache requires --report-tag cachetrim, spooled or scoped")
    if arguments.spool_hessian_executables != (arguments.report_tag in ("spooled", "scoped")):
        parser.error("--spool-hessian-executables requires the separate --report-tag spooled or scoped")
    if arguments.spool_hessian_executables and not arguments.release_stages:
        parser.error("Spooling requires --release-stages")
    if arguments.spool_value_gradient and not arguments.spool_hessian_executables:
        parser.error("--spool-value-gradient requires the spooled Hessian mode")
    if arguments.hessian_chunk_size is not None and (arguments.hessian_chunk_size < 1 or not arguments.spool_hessian_executables):
        parser.error("A positive --hessian-chunk-size requires spooling")
    if arguments.scoped_loads != (arguments.report_tag == "scoped"):
        parser.error("--scoped-loads requires the separate --report-tag scoped")
    if arguments.scoped_loads and not arguments.spool_value_gradient:
        parser.error("Scoped loading requires --spool-value-gradient")
    if arguments.cpu_hessian_no_fusion and not arguments.scoped_loads:
        parser.error("--cpu-hessian-no-fusion requires the explicit scoped policy")
    if arguments.report_tag in ("spooled", "scoped") and Path(
            f"reports/gpu_{arguments.model}_{arguments.backend}_{arguments.report_tag}.json").exists():
        raise FileExistsError("Preserve the existing disk-backed benchmark report")
    os.environ["JAX_PLATFORMS"] = "cuda" if arguments.backend == "gpu" else "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["JAX_ENABLE_COMPILATION_CACHE"] = "false"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    if hasattr(os, "sched_getaffinity"):
        os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import numpy as np
    from hhvv_export_cache import load_export

    started = time.perf_counter()
    devices = jax.devices()
    assert devices and all(d.platform == arguments.backend for d in devices), devices
    assert jax.config.jax_enable_x64
    assert "ROOT" not in sys.modules
    from jaxlib import xla_client
    # Keep owners alive while reading references to nested C++ options.
    compile_options = xla_client.CompileOptions()
    build_options = compile_options.executable_build_options
    debug_options = build_options.debug_options
    effective_xla_options = dict(backend_optimization_level=debug_options.xla_backend_optimization_level,
                                 disabled_hlo_passes=debug_options.xla_disable_hlo_passes)
    hessian_compiler_options = ({"xla_disable_hlo_passes": "fusion",
                                "xla_cpu_parallel_codegen_split_count": 1}
                               if arguments.cpu_hessian_no_fusion and arguments.backend == "cpu" else {})
    gpu_metadata = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader"], text=True).strip()
    gpu_activity = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,clocks.current.sm,temperature.gpu", "--format=csv,noheader"], text=True).strip()
    gpu_processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader"], text=True).strip()
    cpu_model = next(line.split(":",1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
    monitor = MemoryMonitor()
    report = dict(status="running", model=arguments.model, backend=arguments.backend,
                  host=platform.node(), workdir=str(Path.cwd()), python=sys.version,
                  lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"), jax_version=jax.__version__,
                  jaxlib_version=__import__("jaxlib").__version__, devices=[str(d) for d in devices],
                  float64=jax.config.jax_enable_x64, gpu_metadata=gpu_metadata, cpu_model=cpu_model,
                  xla_flags=os.environ.get("XLA_FLAGS"),effective_xla_options=effective_xla_options,
                  gpu_activity_at_start=gpu_activity, gpu_processes_at_start=gpu_processes,
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  benchmark_script=Path(__file__).name,
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256=hashlib.sha256(b"".join(
                      str(p).encode()+b"\0"+p.read_bytes()
                      for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
                  constraint_batch_size=arguments.constraint_batch_size,
                  release_stages=arguments.release_stages,
                  trim_compilation_cache=arguments.trim_compilation_cache,
                  spool_hessian_executables=arguments.spool_hessian_executables,
                  spool_value_gradient=arguments.spool_value_gradient,
                  hessian_chunk_size=arguments.hessian_chunk_size,
                  scoped_loads=arguments.scoped_loads,
                  cpu_hessian_no_fusion=arguments.cpu_hessian_no_fusion,
                  hessian_compiler_options=hessian_compiler_options,
                  report_tag=arguments.report_tag,
                  execution="Complete model, local parameter kernels on selected device, NumPy accumulation in global order",
                  synchronization="Each local scalar/gradient/Hessian is copied to the host before accumulation",
                  compilation="Separate lower() and compile() for every term; timed calls use returned compiled executables; persistent JAX cache disabled",
                  gpu_allocator=dict(preallocate=False, allocator=os.environ.get("XLA_PYTHON_CLIENT_ALLOCATOR", "default")),
                  mapping_limit=int(Path("/proc/sys/vm/max_map_count").read_text()),
                  timings={}, reference_validation=[])
    if arguments.spool_hessian_executables:
        from spooled_executable import SpooledExecutable
        report.update(spooled_executable_sha256=hashlib.sha256(Path(__file__).with_name("spooled_executable.py").read_bytes()).hexdigest(),
            hessian_execution="All entries of the full Hessian; total call latency includes disk reads/hash verification, deserialization, synchronized execution/host copies, executable/cache release and host accumulation",
            executable_storage="Artifacts created and consumed only by this process; PyTrees stay in memory; files preserved as run evidence")
        if arguments.spool_value_gradient:
            report["value_gradient_execution"] = "Complete value+gradient; total call latency includes disk reads/hash verification, deserialization, synchronized execution/host copies, executable/cache release and host accumulation"
    if arguments.scoped_loads:
        from scoped_executable import hessian_with_scoped_loads, value_gradient_points_with_scoped_loads
        report.update(scoped_executable_sha256=hashlib.sha256(
            Path(__file__).with_name("scoped_executable.py").read_bytes()).hexdigest(),
            hessian_execution="All entries of the full Hessian; one executable load per nonempty term and full matrix call, including all column batches, synchronized copies and cleanup in call latency",
            gradient_validation_execution="Untimed reference batch: one executable load per term for all reference points, including a fresh nominal evaluation. These batched calls are excluded from standalone value+gradient latency.")
    suffix = "_"+arguments.report_tag if arguments.report_tag else ""
    output = Path("reports")/f"gpu_{arguments.model}_{arguments.backend}{suffix}.json"
    output.parent.mkdir(exist_ok=True)
    save_lock = threading.Lock()

    def save():
        with save_lock:
            report["elapsed_seconds"] = time.perf_counter()-started
            report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            report["gpu_memory"] = monitor.snapshot()
            report["current_stage"] = monitor.stage
            mappings = process_mapping_count()
            monitor.mapping_peak = max(monitor.mapping_peak, mappings)
            report["mapping_count"] = mappings
            report["sampled_mapping_peak"] = monitor.mapping_peak
            report["mapping_sampling"] = "Every 20 seconds, at report saves, and after per-term cache release"
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def memory_limit():
        report["status"] = "resource_limit"
        report["exception"] = "CPU RSS exceeded the 11,500 MiB benchmark guard during "+monitor.stage
        # Terminate even while the main thread is inside the native compiler.
        # Completed stages are already saved; normal Python exceptions cannot
        # interrupt a long native compilation promptly.
        try:
            save()
            print(report["exception"], flush=True)
        finally:
            os._exit(86)

    monitor.memory_limit_callback = memory_limit

    try:
        manifest_bytes = arguments.reference_manifest.read_bytes()
        manifest = json.loads(manifest_bytes)
        report.update(reference_manifest=str(arguments.reference_manifest),
                      reference_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())
        model = manifest["models"][arguments.model]
        report["source"] = model["source"]
        report["export_stack"] = {k: manifest[k] for k in ("combine_version", "root_version")}
        report["derivative_scope"] = model["reference"].get("derivative_scope", "All smoke-case gradients versus Combine finite differences; Hessians versus local CPU JAX reference")
        checkpoint = time.perf_counter()
        likelihood = load_export(Path("build/gpu_exports")/model["cache"], model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=arguments.constraint_batch_size,
                                               hessian_chunk_size=arguments.hessian_chunk_size)
        point = likelihood.initial
        if "parameter_names" in model["reference"]:
            np.testing.assert_array_equal(likelihood.parameter_names,model["reference"]["parameter_names"])
        report.update(parameters=len(point), channels=len(likelihood.channels), terms=len(evaluator.terms),
                      max_local_parameters=max(len(t.indices) for t in evaluator.terms),
                      setup_seconds=time.perf_counter()-checkpoint)
        results = {}
        report["reference_validation"] = [{} for _ in model["reference"]["cases"]]
        artifact = Path("build/gpu_results")/f"{arguments.model}_{arguments.backend}{suffix}.npz"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        scales = np.array([p.error if p.error > 0 else max(abs(p.value),1.) for p in likelihood.parameters])
        arrays_to_save = dict(parameter_names=np.array(likelihood.parameter_names), point=point,
                              parameter_scales=scales,
                              validation_points=np.array([c["theta"] for c in model["reference"]["cases"]]),
                              validation_labels=np.array([c.get("label",f"case_{i}") for i,c in enumerate(model["reference"]["cases"])]))
        report["validation_array_scope"] = "NLL and value-gradient outputs at every reference point; full Hessian only at nominal"

        def save_arrays():
            np.savez_compressed(artifact, **arrays_to_save)
            report.update(arrays=str(artifact),
                          arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())

        save()
        for operation in ("value", "value_and_grad", "hessian"):
            kernel_name = "hessian_columns" if operation == "hessian" and arguments.hessian_chunk_size else operation
            spool_operation = arguments.spool_hessian_executables and (operation == "hessian" or
                              operation == "value_and_grad" and arguments.spool_value_gradient)
            stores = []
            compiled = lowered = None
            monitor.stage = "compile "+operation
            lowering_seconds = compile_seconds = 0.
            checkpoint = time.perf_counter()
            term_timings = []
            cache_trim_seconds = 0.
            report["compile_progress"] = dict(operation=operation, completed_terms=0,
                                              term_timings=term_timings)
            for i, term in enumerate(evaluator.terms):
                monitor.stage = f"compile {operation} {i+1}/{len(evaluator.terms)} {term.name}"
                if kernel_name == "hessian_columns" and not len(term.indices):
                    term_timings.append(dict(name=term.name,parameters=0,kernel=kernel_name,
                        lowering_seconds=0.,compile_seconds=0.,skipped_empty_hessian=True,
                        compiler_options={}))
                    report["compile_progress"]["completed_terms"] = i+1
                    continue
                function = getattr(term, kernel_name)
                kernel_arguments = (point[term.indices],)
                if kernel_name == "hessian_columns":
                    size = len(term.indices)
                    kernel_arguments += (np.zeros((min(arguments.hessian_chunk_size,size),size)),)
                begin = time.perf_counter()
                lowered = function.lower(*kernel_arguments)
                term_lowering_seconds = time.perf_counter()-begin
                lowering_seconds += term_lowering_seconds
                begin = time.perf_counter()
                compiler_options = hessian_compiler_options if operation == "hessian" else {}
                compiled = lowered.compile(compiler_options=compiler_options) if compiler_options else lowered.compile()
                term_compile_seconds = time.perf_counter()-begin
                compile_seconds += term_compile_seconds
                term_timings.append(dict(name=term.name, parameters=len(term.indices), kernel=kernel_name,
                                         lowering_seconds=term_lowering_seconds,
                                         compile_seconds=term_compile_seconds,
                                         compiler_options=compiler_options))
                if spool_operation:
                    monitor.stage = f"serialize {operation} {i+1}/{len(evaluator.terms)} {term.name}"
                    executable_path = Path("build/gpu_executables")/f"{arguments.model}_{arguments.backend}{suffix}"/operation/f"{i}.bin"
                    stored, storage_record = SpooledExecutable.write(compiled,executable_path,backend=arguments.backend)
                    term_timings[-1].update(executable=str(executable_path),**storage_record)
                    stores.append((term.name,stored))
                    setattr(term,kernel_name,stored)
                    del compiled
                    compiled = None
                else:
                    setattr(term,kernel_name,compiled)
                if arguments.trim_compilation_cache:
                    monitor.stage = f"trim {operation} {i+1}/{len(evaluator.terms)} {term.name}"
                    trim_started = time.perf_counter()
                    rss_before = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))/1024
                    del function, lowered
                    jax.clear_caches()
                    gc.collect()
                    C.CDLL(None).malloc_trim(0)
                    lowered = None
                    seconds = time.perf_counter()-trim_started
                    cache_trim_seconds += seconds
                    rss_after = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))/1024
                    term_timings[-1].update(cache_trim_seconds=seconds,
                                           rss_before_trim_mib=rss_before, rss_after_trim_mib=rss_after,
                                           mappings_after_trim=process_mapping_count())
                    monitor.mapping_peak = max(monitor.mapping_peak, term_timings[-1]["mappings_after_trim"])
                report["compile_progress"]["completed_terms"] = i+1
                if arguments.spool_hessian_executables and (i+1) % 10 == 0:
                    save()
                if (i+1) % 100 == 0:
                    print(f"{operation}: compiled {i+1}/{len(evaluator.terms)} terms", flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 11500:
                    report["failed_compile_progress"] = dict(operation=operation, completed_terms=i+1,
                                                              term_timings=term_timings)
                    raise MemoryError("CPU RSS exceeded the 11,500 MiB benchmark guard")
            preparation_seconds = time.perf_counter()-checkpoint
            monitor.stage = "execute "+operation
            report.setdefault("preparation_timings", {})[operation] = dict(
                completed_terms=len(term_timings), lowering_seconds=lowering_seconds,
                compile_seconds=compile_seconds, cache_trim_seconds=cache_trim_seconds,
                total_preparation_seconds=preparation_seconds)
            save()
            function = ((lambda theta: hessian_with_scoped_loads(evaluator, theta))
                        if arguments.scoped_loads and operation == "hessian"
                        else getattr(evaluator, operation))
            timed_storage_calls = []
            call_number = 0

            def timed_call():
                nonlocal call_number
                call_number += 1
                monitor.stage = f"execute {operation} call {call_number}/{arguments.repeat+1}"
                previous_counts = [len(store.calls) for _,store in stores]
                begin = time.perf_counter()
                value = function(point)
                elapsed = time.perf_counter()-begin
                if stores:
                    calls = [c for (_,store),first in zip(stores,previous_counts,strict=True) for c in store.calls[first:]]
                    timed_storage_calls.append(dict(total_seconds=elapsed,kernel_calls=len(calls),
                        **{key:sum(c[key] for c in calls) for key in calls[0]}))
                return value,elapsed

            result,first_seconds = timed_call()
            arrays = result if isinstance(result, tuple) else (result,)
            if any(not np.all(np.isfinite(x)) for x in arrays):
                report["nonfinite_counts"] = [int(np.count_nonzero(~np.isfinite(x))) for x in arrays]
                arrays_to_save["nonfinite_"+operation] = arrays[-1]
                save_arrays()
                raise ValueError(f"Nonfinite {operation} at nominal point")
            samples = []
            for _ in range(arguments.repeat):
                warm_result,seconds = timed_call()
                del warm_result
                samples.append(seconds)
            results[operation] = result
            report["timings"][operation] = dict(lowering_seconds=lowering_seconds, compile_seconds=compile_seconds,
                                                cache_trim_seconds=cache_trim_seconds,
                                                total_preparation_seconds=preparation_seconds,
                                                first_execution_seconds=first_seconds, warm_seconds=samples,
                                                median_warm_seconds=float(np.median(samples)),
                                                min_warm_seconds=min(samples), memory_after=monitor.snapshot(),
                                                term_timings=term_timings)
            if stores:
                report["timings"][operation].update(spooled_timed_calls=timed_storage_calls,
                    spooled_timing_scope="First call then each warm call; wall times include all loading and cleanup. Component times sum across local kernels/column batches; validation calls excluded.",
                    serialization_seconds=sum(t.get("serialization_seconds",0.) for t in term_timings),
                    executable_write_seconds=sum(t.get("write_seconds",0.) for t in term_timings),
                    serialized_bytes=sum(t.get("serialized_bytes",0) for t in term_timings))
            if operation == "value":
                arrays_to_save.update(value=float(result), term_values=np.array([
                    float(t.value(point[t.indices])) for t in evaluator.terms]))
                report.update(nll=float(result), nominal_value_absolute_error=abs(
                    float(result)-model["reference"]["cases"][0]["combine_nll"]))
            elif operation == "value_and_grad":
                arrays_to_save.update(vg_value=float(result[0]), gradient=result[1])
                report["value_gradient_nll"] = float(result[0])
            else:
                arrays_to_save["hessian"] = result
            save_arrays()
            report["completed_operations"] = list(results)
            report.pop("compile_progress", None)
            print(operation, {k:v for k,v in report["timings"][operation].items() if k not in ("term_timings", "spooled_timed_calls")}, flush=True)
            save()
            monitor.stage = "validate "+operation
            validation_values, validation_gradients = [], []
            batch_validation = None
            if arguments.scoped_loads and operation == "value_and_grad":
                batch_validation = value_gradient_points_with_scoped_loads(
                    evaluator, np.asarray([c["theta"] for c in model["reference"]["cases"]]))
            for case_index, (case, check) in enumerate(zip(
                    model["reference"]["cases"], report["reference_validation"], strict=True)):
                if operation == "hessian" and "hessian" not in case:
                    continue
                theta = np.asarray(case["theta"])
                validation = ((batch_validation[0][case_index], batch_validation[1][case_index])
                              if batch_validation is not None else
                              result if np.array_equal(theta, point) else function(theta))
                if operation == "value":
                    validation_values.append(validation)
                elif operation == "value_and_grad":
                    validation_values.append(validation[0])
                    validation_gradients.append(validation[1])
                payload = validation if isinstance(validation,tuple) else (validation,)
                nonfinite = [int(np.count_nonzero(~np.isfinite(v))) for v in payload]
                if any(nonfinite):
                    check.setdefault("nonfinite_operations",{})[operation] = nonfinite
                    continue
                if operation == "value":
                    check["nll_absolute_error"] = abs(validation-case["combine_nll"])
                elif operation == "value_and_grad":
                    value, gradient = validation
                    check["value_gradient_nll_absolute_error"] = abs(value-case["combine_nll"])
                    if "combine_fd_gradient" in case:
                        reference = np.asarray(case["combine_fd_gradient"])
                        errors = np.abs(gradient-reference)
                        check.update(gradient_max_absolute_error=float(errors.max(initial=0.)),
                                     gradient_max_relative_error=float((errors/np.maximum(np.abs(reference),1e-8)).max(initial=0.)))
                else:
                    errors = np.abs(validation-np.asarray(case["hessian"]))
                    reference_scales = np.asarray(model["reference"]["parameter_scales"])
                    check["hessian_max_scaled_error"] = float((errors*np.outer(reference_scales,reference_scales)).max(initial=0.))
            if operation == "value":
                arrays_to_save["validation_values"] = np.asarray(validation_values)
            elif operation == "value_and_grad":
                arrays_to_save.update(validation_vg_values=np.asarray(validation_values),
                                      validation_gradients=np.asarray(validation_gradients))
            save_arrays()
            if stores:
                report["timings"][operation]["spooled_all_calls"] = {name:dict(count=len(store.calls),
                    **{key:sum(c[key] for c in store.calls) for key in store.calls[0]}) for name,store in stores}
                if arguments.scoped_loads:
                    report["timings"][operation]["scoped_load_sessions"] = {
                        name: getattr(store, "scoped_loads", []) for name, store in stores}
            if arguments.release_stages:
                monitor.stage = "release "+operation
                cleanup_start = time.perf_counter()
                rss_before = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))/1024
                for term in evaluator.terms:
                    setattr(term, kernel_name, None)
                del compiled, lowered
                jax.clear_caches()
                gc.collect()
                C.CDLL(None).malloc_trim(0)
                rss_after = int(next(line.split()[1] for line in Path("/proc/self/status").read_text().splitlines() if line.startswith("VmRSS:")))/1024
                report["timings"][operation].update(release_seconds=time.perf_counter()-cleanup_start,
                                                     rss_before_release_mib=rss_before,
                                                     rss_after_release_mib=rss_after)
            save()
        value = float(results["value"])
        vg_value, gradient = results["value_and_grad"]
        hessian = results["hessian"]
        report.update(nll=value, value_gradient_nll=float(vg_value),
                      nominal_value_absolute_error=abs(value-model["reference"]["cases"][0]["combine_nll"]),
                      hessian_max_asymmetry=float(np.abs(hessian-hessian.T).max(initial=0.)),
                      arrays=str(artifact), arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                      root_imported="ROOT" in sys.modules)
        assert not report["root_imported"]
        if monitor.error or (arguments.backend == "gpu" and not monitor.pid_seen):
            raise RuntimeError("GPU memory monitoring did not produce valid process measurements")
        failures = [c for c in report["reference_validation"] if c.get("nonfinite_operations") or c["nll_absolute_error"] >= 1e-6 or c["value_gradient_nll_absolute_error"] >= 1e-6 or c.get("gradient_max_relative_error",0.) >= 1e-4 or c.get("hessian_max_scaled_error",0.) >= 1e-6]
        if report["nominal_value_absolute_error"] >= 1e-6:
            failures.append(dict(nominal_value_absolute_error=report["nominal_value_absolute_error"]))
        report["status"] = "reference_failed" if failures else "passed"
        save()
        print("Completed", output, report["status"], flush=True)
        if failures:
            raise AssertionError("Reference validation failed; see report")
    except Exception:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
