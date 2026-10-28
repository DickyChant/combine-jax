"""Compute complete AD derivatives at an independently checked native fit point.

This can diagnose a rejected native fit; it cannot promote that fit to a valid
minimum or turn a repaired native covariance into an uncertainty reference.
"""

import argparse
import hashlib
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

from compare_local_cpu_hessian import reassemble_local
from complete_bounded_cpu_covariance import OPTIONS, check_configuration, full_covariance


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core_digest():
    return hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()


def checked_native_point(path):
    """Bind the point check to a terminal native report and its saved arrays."""
    check = json.loads(path.read_bytes())
    assert check["status"] in ("point_checked", "candidate_checked")
    assert check["point_source"] == "final"
    parent_path = Path(check["parent"])
    assert digest(parent_path) == check["parent_sha256"]
    parent = json.loads(parent_path.read_bytes())
    assert parent["stage"] == "finished"
    assert parent["status"] == check["parent_status"]
    assert parent["status"] in ("reference_ready", "fit_incomplete")
    for key in ("model", "source", "parameter_names", "discrete_choices", "channel_masks",
                "combine_version", "root_version", "runtime_flags"):
        assert check[key] == parent[key], key
    assert check["combine_version"] == "11.0.0" and check["root_version"] == "6.40.00"
    assert check["parameter_bounds_checked"] and check["fixed_parameters_unchanged"]
    assert digest(parent["arrays"]) == parent["arrays_sha256"]
    with np.load(parent["arrays"], allow_pickle=False) as saved:
        point = saved["values"].copy()
        np.testing.assert_array_equal(saved["parameter_names"], check["parameter_names"])
    assert point.dtype == np.float64 and point.shape == (len(check["parameter_names"]),)
    assert len(set(check["parameter_names"])) == len(point) and np.isfinite(point).all()
    np.testing.assert_array_equal(point, check["point"])
    np.testing.assert_array_equal(point, parent["final"]["values"])
    probes = {p["label"]: p["nll"] for p in check["probes"]}
    assert len(probes) == len(check["probes"]) == 5
    assert all(p["roofit_evaluation_errors"] == 0 for p in check["probes"])
    nominal, zero, canonical, restored = [probes[key] for key in (
        "canonical_initial", "saved_point_zero_offset", "saved_point_canonical", "restored_initial")]
    assert np.isfinite(list(probes.values())).all()
    errors = dict(saved_zero_point_absolute_error=abs(zero-parent["final"]["nll_zero_point"]),
        canonical_offset_absolute_error=abs(canonical-(nominal+zero)),
        canonical_restoration_absolute_error=abs(restored-nominal))
    for key, value in errors.items():
        assert value == check[key] and value < 1e-6, key
    assert abs(probes["zero_point_initial"]) < 1e-6
    assert abs(nominal-parent["canonical_initial_nll"]) < 1e-6
    assert canonical == check["canonical_candidate_nll"]
    assert zero == check["zero_point_candidate_nll"]
    assert abs(canonical-parent["canonical_fitted_nll"]) < 1e-6
    return check, parent, point


def reassemble_value_gradient(report, point, value, gradient, offset):
    """Check every saved local value/gradient in the same order as assembly."""
    assert len(report["terms"]) == report["expected_terms"]
    restored_value, restored_gradient = float(offset), np.zeros_like(point)
    for index, record in enumerate(report["terms"]):
        assert record["index"] == index and digest(record["arrays"]) == record["arrays_sha256"]
        with np.load(record["arrays"], allow_pickle=False) as block:
            indices = block["indices"]
            assert indices.shape == (record["parameters"],) and np.issubdtype(indices.dtype, np.integer)
            assert len(set(indices)) == len(indices) and np.all((0 <= indices) & (indices < len(point)))
            np.testing.assert_array_equal(block["point"], point[indices])
            local_gradient, local_value = block["gradient"], block["nll"]
            assert local_gradient.shape == indices.shape and local_gradient.dtype == np.float64
            assert local_value.shape == () and local_value.dtype == np.float64
            assert np.isfinite(local_gradient).all() and np.isfinite(local_value)
            restored_value += float(local_value)
            restored_gradient[indices] += local_gradient
    np.testing.assert_array_equal(restored_gradient, gradient)
    np.testing.assert_equal(restored_value, value)
    return dict(value_bitwise_reassembled=True, gradient_bitwise_reassembled=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-check", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-fd-refinement", type=Path)
    parser.add_argument("--rss-limit-mib", type=float, default=3500.)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    args = parser.parse_args()
    if (args.output.parent != Path("reports") or args.output.suffix != ".json"
            or not 0 < args.rss_limit_mib <= 11000 or not 0 < args.wall_seconds < np.inf):
        parser.error("Use reports/*.json, RSS in (0, 11000] MiB and a finite positive wall limit")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This diagnostic is local only")
    directory = Path("build/fit_results")/(args.output.stem+"_blocks")
    artifact = Path("build/fit_results")/(args.output.stem+".npz")
    if any(p.exists() for p in (args.output, directory, artifact)):
        raise FileExistsError("Preserve previous diagnostic evidence")
    check, parent, point = checked_native_point(args.point_check)
    assert parent["model"] in ("counting", "hhvv")
    assert (args.nominal_fd_refinement is not None) == (parent["model"] == "hhvv")
    assert "jax" not in sys.modules and "ROOT" not in sys.modules
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                      JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false")
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    import jax
    import jaxlib
    from jaxlib import xla_client
    from build_cpu_hessian_reference import validate_nominal_refinement
    from combine_jax.local import release_compilation_cache
    from hhvv_export_cache import load_export
    from validate_cpu_no_fusion_hessian import all_columns

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert xla_client.CompileOptions().executable_build_options.debug_options.xla_disable_hlo_passes == ""
    jax.config.update("jax_enable_compilation_cache", False)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][parent["model"]]
    export = Path("build/gpu_exports")/model["cache"]
    assert model["source"] == parent["source"]
    assert all(manifest[k] == parent[k] for k in ("combine_version", "root_version"))
    gate_path = Path(f"reports/{parent['model']}_cpu_no_fusion_hessian.json")
    gate = json.loads(gate_path.read_bytes())
    assert gate["source"] == model["source"] and gate["source_sha256"] == core_digest()
    assert gate["export_sha256"] == digest(export) and gate["hessian_compiler_options"] == OPTIONS
    assert gate["jax_version"] == jax.__version__ and gate["jaxlib_version"] == jaxlib.__version__
    assert all(digest(p) == h for p, h in gate["runner_sha256"].items())
    assert all(digest(c["arrays"]) == c["arrays_sha256"] for c in gate["cases"])
    refinement = validate_nominal_refinement(gate, gate_path, args.nominal_fd_refinement) if args.nominal_fd_refinement else None
    assert refinement or gate["status"] == "passed"
    paths = [args.point_check, Path(check["parent"]), Path(parent["arrays"]), manifest_path,
             export, gate_path, *Path("src/combine_jax").rglob("*.py")]
    paths += [Path("scripts")/p for p in ("diagnose_native_fit_curvature.py",
        "build_cpu_hessian_reference.py", "validate_cpu_no_fusion_hessian.py", "refine_cpu_hessian_fd.py",
        "compare_local_cpu_hessian.py", "complete_bounded_cpu_covariance.py", "hhvv_export_cache.py")]
    if args.nominal_fd_refinement:
        paths.append(args.nominal_fd_refinement)
    frozen = {str(p): digest(p) for p in paths}
    started = time.perf_counter()
    directory.mkdir(parents=True)
    report = dict(status="running", stage="load checked fit-point export", model=parent["model"],
        host=platform.node(), backend="cpu", jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        float64=True, root_imported=False, cpu_affinity=sorted(os.sched_getaffinity(0)),
        source=parent["source"], source_sha256=core_digest(), export_sha256=digest(export),
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        point_check=str(args.point_check), point_check_sha256=digest(args.point_check),
        native_report=check["parent"], native_report_sha256=check["parent_sha256"],
        native_status=parent["status"], native_final_flags={k:v for k,v in parent["final"].items() if k not in ("values", "errors")},
        parameter_names=parent["parameter_names"], point=point.tolist(),
        manifest=str(manifest_path), manifest_sha256=digest(manifest_path), frozen_sources=frozen,
        runner_sha256=digest(__file__), constraint_batch_size=32, hessian_chunk_size=4,
        gradient_compiler_options=OPTIONS, hessian_compiler_options=OPTIONS,
        validation_gate=str(gate_path), validation_gate_sha256=digest(gate_path), nominal_fd_refinement=refinement,
        xla_flags=os.environ.get("XLA_FLAGS"), persistent_cache_enabled=False,
        rss_limit_mib=args.rss_limit_mib, wall_limit_seconds=args.wall_seconds,
        mapping_guard=int(Path("/proc/sys/vm/max_map_count").read_text())-4096,
        completed_operations=[], terms=[],
        scope="Full AD value, gradient and Hessian at the independently checked native point. "
              "Raw free-Hessian inversion is diagnostic only. No minimum, native uncertainty, "
              "profile, Minuit-coordinate covariance or fit-performance acceptance; the parent status is retained.")
    lock, stop = threading.RLock(), threading.Event()
    evaluator = None

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                mapping_count=len(Path("/proc/self/maps").read_text().splitlines()))
            tmp = args.output.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            tmp.replace(args.output)

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
                print(report["stage"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
                last = time.perf_counter()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        check_configuration(likelihood, parent)
        np.testing.assert_array_equal(likelihood.initial, parent["initial"])
        fixed = [p.name for p in likelihood.parameters if p.minimum == p.maximum]
        assert set(fixed) == set(parent["fixed_bounds"])
        for p, x in zip(likelihood.parameters, point, strict=True):
            assert p.minimum <= x <= p.maximum
        free = np.asarray([i for i,p in enumerate(likelihood.parameters) if p.name not in fixed], dtype=int)
        scales = np.asarray([p.error if p.error > 0 else max(abs(p.value), 1.) for p in likelihood.parameters])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        nll, gradient, hessian = float(likelihood.offset), np.zeros_like(point), np.zeros((len(point), len(point)))
        report.update(expected_terms=len(evaluator.terms), parameters=len(point), fixed_bounds=fixed,
                      channels=len(likelihood.channels), likelihood_offset=float(likelihood.offset))
        assert len({t.name for t in evaluator.terms}) == len(evaluator.terms)
        for index, term in enumerate(evaluator.terms):
            with lock:
                report["stage"] = f"value/gradient {index+1}/{len(evaluator.terms)} {term.name}"
                save()
            local_point = point[term.indices]
            begin = time.perf_counter()
            kernel = term.value_and_grad.lower(local_point).compile(compiler_options=OPTIONS)
            raw_value, raw_gradient = kernel(local_point)
            contribution, local_gradient = float(raw_value), np.array(raw_gradient, copy=True)
            assert np.isfinite(contribution) and np.isfinite(local_gradient).all()
            nll += contribution
            gradient[term.indices] += local_gradient
            del kernel, raw_value, raw_gradient
            term.reset_kernel("value_and_grad")
            release_compilation_cache()
            gradient_seconds = time.perf_counter()-begin
            with lock:
                report["stage"] = f"Hessian {index+1}/{len(evaluator.terms)} {term.name}"
                save()
            size = len(local_point)
            if size:
                begin = time.perf_counter()
                lowered = term.hessian_columns.lower(local_point, np.zeros((min(4, size), size)))
                lowering = time.perf_counter()-begin
                begin = time.perf_counter()
                kernel = lowered.compile(compiler_options=OPTIONS)
                compilation = time.perf_counter()-begin
                del lowered
                begin = time.perf_counter()
                block = all_columns(kernel, local_point)
                execution = time.perf_counter()-begin
                del kernel
            else:
                lowering = compilation = execution = 0.
                block = np.empty((0, 0), dtype=np.float64)
            hessian[np.ix_(term.indices, term.indices)] += block
            block_path = directory/f"{index}.npz"
            np.savez_compressed(block_path, indices=term.indices, point=local_point,
                                nll=contribution, gradient=local_gradient, hessian=block)
            begin = time.perf_counter()
            term.reset_kernel("hessian_columns")
            release_compilation_cache()
            cleanup = time.perf_counter()-begin
            with lock:
                report["terms"].append(dict(index=index, name=term.name, parameters=size,
                    gradient_prepare_execute_cleanup_seconds=gradient_seconds,
                    lowering_seconds=lowering, compile_seconds=compilation,
                    first_all_columns_seconds=execution, cleanup_seconds=cleanup,
                    arrays=str(block_path), arrays_sha256=digest(block_path)))
                save()
        error = abs(nll-check["canonical_candidate_nll"])
        assert error < 1e-6, "AD NLL differs from the freshly checked native point"
        assembly = reassemble_local(report, dict(point=point, hessian=hessian))
        assembly.update(reassemble_value_gradient(report, point, nll, gradient, likelihood.offset))
        with lock:
            report.update(stage="diagnose complete raw free Hessian", nll=nll,
                          native_point_nll_absolute_error=error, block_reassembly=assembly)
            save()
        begin = time.perf_counter()
        covariance, diagnostic = full_covariance(hessian, free, scales)
        inversion_seconds = time.perf_counter()-begin
        assert all(digest(p) == h for p,h in frozen.items()) and "ROOT" not in sys.modules
        tmp = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, parameter_names=np.asarray(likelihood.parameter_names), point=point,
            parameter_scales=scales, free_indices=free, value=nll, gradient=gradient, hessian=hessian,
            diagnostic_raw_covariance=np.empty((0, 0)) if covariance is None else covariance)
        tmp.replace(artifact)
        with lock:
            report.update(status="computed", stage="finished", arrays=str(artifact), arrays_sha256=digest(artifact),
                completed_operations=["value_and_grad", "hessian"], raw_covariance_diagnostic=diagnostic,
                timings=dict(raw_hessian_inversion_seconds=inversion_seconds,
                    **{key:sum(t[key] for t in report["terms"]) for key in (
                        "gradient_prepare_execute_cleanup_seconds", "lowering_seconds", "compile_seconds",
                        "first_all_columns_seconds", "cleanup_seconds")}))
            save()
    except BaseException:
        with lock:
            report.update(status="failed", exception=traceback.format_exc())
            save()
        raise
    finally:
        if evaluator is not None:
            evaluator.clear_cache(release_cache=True)
        stop.set()
        watcher.join()
        save()
    print(args.output, report["status"], report.get("raw_covariance_diagnostic"), flush=True)


if __name__ == "__main__":
    main()
