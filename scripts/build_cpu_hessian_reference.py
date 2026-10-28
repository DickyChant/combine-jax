"""Assemble a local CPU Hessian with validated per-kernel compiler options.

This is a numerical reference job, not the matched lxplus CPU/GPU benchmark.
Each block is preserved before releasing its executable; no ROOT is imported.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import socket
import sys
import threading
import time
import traceback

import jax
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export
from validate_cpu_no_fusion_hessian import OPTIONS, all_columns, digest, scaled_error


def validate_nominal_refinement(gate, gate_path, refinement_path):
    """Accept only a hash-bound, independently selected nominal FD replacement."""
    from refine_cpu_hessian_fd import STEPS, TOLERANCE, first_converged_triple

    refinement = json.loads(refinement_path.read_text())
    assert gate["status"] == "failed" and refinement["status"] == "passed"
    assert Path(refinement["parent"]) == gate_path
    assert refinement["parent_sha256"] == digest(gate_path)
    assert refinement["runner_sha256"] == digest(Path(__file__).with_name("refine_cpu_hessian_fd.py"))
    for key in ("source", "source_sha256", "jax_version", "jaxlib_version", "float64", "backend"):
        assert refinement[key] == gate[key], key
    assert refinement["default_disabled_hlo_passes"] == ""
    assert refinement["steps"] == list(STEPS)
    assert refinement["selection_completed_before_reading_ad_hessian"]
    failed = [c for c in gate["cases"] if c["status"] != "passed"]
    assert len(failed) == 1 and failed[0]["status"] == "failed"
    case = failed[0]
    assert case["index"] == refinement["term_index"] and case["name"] == refinement["term_name"]
    assert refinement["point_index"] == 0
    for entry in gate["cases"]:
        for index, check in enumerate(entry["finite_difference_checks"]):
            assert check["passed"] or (entry is case and index == 0)
    assert refinement["parent_arrays"] == case["arrays"]
    assert refinement["parent_arrays_sha256"] == case["arrays_sha256"] == digest(case["arrays"])
    assert refinement["arrays_sha256"] == digest(refinement["arrays"])
    with np.load(refinement["arrays"], allow_pickle=False) as arrays, \
            np.load(case["arrays"], allow_pickle=False) as original:
        for key in ("parameter_names", "free_indices", "parameter_scales"):
            np.testing.assert_array_equal(arrays[key], original[key])
        np.testing.assert_array_equal(arrays["point"], original["points"][0])
        np.testing.assert_array_equal(arrays["steps"], STEPS)
        free = arrays["free_indices"]
        scales = arrays["parameter_scales"][free]
        matrices = arrays["fd_hessians"]
        stencils = arrays["gradient_stencils"]
        assert matrices.shape == (len(STEPS), len(free), len(free))
        assert stencils.shape == (len(STEPS), len(free), 4, len(free))
        assert np.isfinite(matrices).all() and np.isfinite(stencils).all()
        for step_index, step in enumerate(STEPS):
            for column in range(len(free)):
                a, b, c, d = stencils[step_index, column]
                reconstructed = (a-8*b+8*c-d)/(12*step*scales[column])
                np.testing.assert_allclose(reconstructed, matrices[step_index, :, column],
                                           rtol=1e-13, atol=1e-13)
        assert scaled_error(matrices[0], original["default_gradient_fd_hessians"][0, -1], scales) < 1e-10
        selection = first_converged_triple(matrices, scales)
        assert selection is not None
        assert json.loads(json.dumps(selection)) == refinement["selection"]
        hessian = original["original_hessians"][0][np.ix_(free, free)]
        np.testing.assert_array_equal(arrays["saved_ad_hessian"], hessian)
        errors = [scaled_error(hessian, m, scales) for m in matrices]
        np.testing.assert_array_equal(errors, refinement["ad_errors_at_every_step"])
        selected = [errors[i] for i in selection["indices"]]
        assert max(selected) < TOLERANCE
        np.testing.assert_array_equal(selected, refinement["selected_ad_errors"])
    return dict(report=str(refinement_path), report_sha256=digest(refinement_path),
                arrays_sha256=refinement["arrays_sha256"], original_gate_status=gate["status"],
                selected_steps=selection["steps"], selected_ad_errors=selected,
                scope="Only the failed nominal FD reference is supplemented; original failure retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "automcstats", "hhvv"), required=True)
    parser.add_argument("--rss-limit-mib", type=float, default=3500.)
    parser.add_argument("--compile-cores", type=int, default=1)
    parser.add_argument("--nominal-fd-refinement", type=Path,
                        help="Explicit supplemental reference for the preserved failed HHVV nominal gate")
    args = parser.parse_args()
    if not 0 < args.rss_limit_mib <= 11000 or not 1 <= args.compile_cores <= 7:
        parser.error("Require RSS in (0, 11000] MiB and 1–7 compiler cores")
    if args.nominal_fd_refinement and args.model != "hhvv":
        parser.error("The supplemental nominal reference applies only to HHVV")
    available = sorted(os.sched_getaffinity(0))
    assert len(available) >= args.compile_cores+1
    os.sched_setaffinity(0, available[1:1+args.compile_cores])
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64 and "ROOT" not in sys.modules
    source_hash = hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()
    gate_path = Path(f"reports/{args.model}_cpu_no_fusion_hessian.json")
    gate = json.loads(gate_path.read_text())
    assert gate["source_sha256"] == source_hash
    assert gate["hessian_compiler_options"] == OPTIONS and gate["jax_version"] == jax.__version__
    for case in gate["cases"]:
        assert digest(case["arrays"]) == case["arrays_sha256"]
    if args.nominal_fd_refinement:
        refinement = validate_nominal_refinement(gate, gate_path, args.nominal_fd_refinement)
    else:
        assert gate["status"] == "passed"
        refinement = None
    output = Path(f"reports/gpu_local_{args.model}_hessian_no_fusion.json")
    directory = Path(f"build/gpu_results/local_{args.model}_hessian_no_fusion")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve previous Hessian evidence")
    directory.mkdir(parents=True)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    model = manifest["models"][args.model]
    export_path = Path("build/gpu_exports")/model["cache"]
    assert gate["source"] == model["source"] and gate["export_sha256"] == digest(export_path)
    started = time.perf_counter()
    report = dict(status="running", stage="setup", model=args.model, source=model["source"],
        host=socket.getfqdn(), backend="cpu", jax_version=jax.__version__,
        jaxlib_version=__import__("jaxlib").__version__, float64=True,
        cpu_affinity=sorted(os.sched_getaffinity(0)), rss_limit_mib=args.rss_limit_mib,
        xla_flags=os.environ.get("XLA_FLAGS"), hessian_compiler_options=OPTIONS,
        constraint_batch_size=32, hessian_chunk_size=4, persistent_cache_enabled=False,
        source_sha256=source_hash, manifest=str(manifest_path), manifest_sha256=digest(manifest_path),
        export_sha256=digest(export_path), validation_gate=str(gate_path), validation_gate_sha256=digest(gate_path),
        nominal_fd_refinement=refinement,
        runner_sha256={str(p): digest(p) for p in (
            Path(__file__), Path("scripts/validate_cpu_no_fusion_hessian.py"), Path("scripts/hhvv_export_cache.py"))},
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        scope="Local CPU nominal Hessian for numerical comparison; compile-then-execute per term; no matched lxplus/GPU performance, fit or independent Combine curvature acceptance",
        completed_operations=[], terms=[])
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
        likelihood = load_export(export_path, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        point = likelihood.initial
        np.testing.assert_array_equal(point, model["reference"]["cases"][0]["theta"])
        scales = np.array([p.error if p.error > 0 else max(abs(p.value), 1.) for p in likelihood.parameters])
        hessian = np.zeros((len(point), len(point)))
        assert len({t.name for t in evaluator.terms}) == len(evaluator.terms)
        report.update(parameters=len(point), channels=len(likelihood.channels),
                      expected_terms=len(evaluator.terms), point=point.tolist())
        for index, term in enumerate(evaluator.terms):
            report["stage"] = f"compile {index+1}/{len(evaluator.terms)} {term.name}"
            save()
            local_point = point[term.indices]
            size = len(local_point)
            begin = time.perf_counter()
            if size:
                lowered = term.hessian_columns.lower(local_point, np.zeros((min(4, size), size)))
                lowering = time.perf_counter()-begin
                begin = time.perf_counter()
                compiled = lowered.compile(compiler_options=OPTIONS)
                compilation = time.perf_counter()-begin
                del lowered
                report["stage"] = f"execute {index+1}/{len(evaluator.terms)} {term.name}"
                begin = time.perf_counter()
                block = all_columns(compiled, local_point)
                execution = time.perf_counter()-begin
                del compiled
            else:
                lowering = compilation = execution = 0.
                block = np.zeros((0, 0))
            hessian[np.ix_(term.indices, term.indices)] += block
            artifact = directory/f"{index}.npz"
            np.savez_compressed(artifact, indices=term.indices, point=local_point, hessian=block)
            begin = time.perf_counter()
            term.reset_kernel("hessian_columns")
            release_compilation_cache()
            cleanup = time.perf_counter()-begin
            record = dict(index=index, name=term.name, parameters=size,
                lowering_seconds=lowering, compile_seconds=compilation,
                first_all_columns_seconds=execution, cleanup_seconds=cleanup,
                max_asymmetry=float(np.max(abs(block-block.T), initial=0.)),
                arrays=str(artifact), arrays_sha256=digest(artifact))
            report["terms"].append(record)
            save()
            if (index+1) % 10 == 0:
                print(f"completed {index+1}/{len(evaluator.terms)} terms", flush=True)
        assert len(report["terms"]) == report["expected_terms"]
        # Reassemble the saved blocks, verifying coverage and each checksum.
        restored = np.zeros_like(hessian)
        for index, (term, record) in enumerate(zip(evaluator.terms, report["terms"], strict=True)):
            assert record["index"] == index and record["name"] == term.name
            assert digest(record["arrays"]) == record["arrays_sha256"]
            with np.load(record["arrays"], allow_pickle=False) as arrays:
                np.testing.assert_array_equal(arrays["indices"], term.indices)
                np.testing.assert_array_equal(arrays["point"], point[term.indices])
                restored[np.ix_(term.indices, term.indices)] += arrays["hessian"]
        np.testing.assert_array_equal(restored, hessian)
        assert np.isfinite(hessian).all()
        artifact = directory/"full.npz"
        np.savez_compressed(artifact, point=point, parameter_names=np.asarray(likelihood.parameter_names),
                            parameter_scales=scales, hessian=hessian)
        report.update(status="computed", stage="finished", completed_operations=["hessian"],
            arrays=str(artifact), arrays_sha256=digest(artifact), saved_block_reassembly_bitwise_equal=True,
            hessian_max_asymmetry=float(abs(hessian-hessian.T).max()),
            timings={key: sum(t[key] for t in report["terms"]) for key in (
                "lowering_seconds", "compile_seconds", "first_all_columns_seconds", "cleanup_seconds")})
        if args.model != "hhvv":
            error = scaled_error(hessian, model["reference"]["cases"][0]["hessian"], scales)
            report.update(saved_cpu_reference_scaled_error=error, status="passed" if error < 1e-10 else "failed")
            assert error < 1e-10
        save()
        print(report["status"], report["timings"], report["peak_rss_mib"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()


if __name__ == "__main__":
    main()
