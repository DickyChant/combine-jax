"""Compute a complete fit-point AD Hessian after a validated bounded CPU minimum.

The minimum stays immutable. Every local Hessian block is saved and independently
reassembled before inversion; nonpositive or unstable curvature stays incomplete.
"""

import argparse
import copy
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

from compare_exported_fit import checked_start, load_arrays
from compare_local_cpu_hessian import reassemble_local


OPTIONS = {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def core_digest():
    return hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()


def checked_minimum(path):
    report = json.loads(path.read_bytes())
    assert report["status"] == "minimum_computed" and report["stage"] == "finished"
    assert report["backend"] == "cpu" and report["host"] == platform.node()
    assert report["float64"] and not report["root_imported"]
    assert report["memory_policy"] == "resident_value_spooled_gradient"
    assert report["evaluation_counters"]["closed"] and report["evaluation_counters"]["prepared"]
    assert report["runner_sha256"] == digest("scripts/fit_bounded_cpu.py")
    assert report["source_sha256"] == core_digest()
    assert all(digest(p) == h for p, h in report["frozen_source_files"].items())
    final = report["final"]
    assert final["valid_minimum"] and final["covariance_status"] == "not_requested"
    assert report["optimizer"]["errordef"] == .5
    assert 0 <= report["initial_combine_nll_absolute_error"] < 1e-6
    assert np.isfinite(final["nll"]) and np.isfinite(final["edm"])
    assert abs(final["nll"]-(final["objective_offset"]+final["shifted_nll"])) < 1e-6
    arrays = load_arrays(report)
    checked_start(report, arrays)
    size = len(report["parameter_names"])
    for key in ("values", "gradient"):
        assert arrays[key].shape == (size,) and arrays[key].dtype == np.float64
        assert np.isfinite(arrays[key]).all()
    assert arrays["hessian"].shape == arrays["covariance"].shape == (0, 0)
    checkpoint_path = Path(report["gradient_checkpoint"])
    assert digest(checkpoint_path) == report["gradient_checkpoint_sha256"]
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert checkpoint["source"] == report["source"]
    assert checkpoint["source_sha256"] == report["source_sha256"]
    assert checkpoint["export_sha256"] == report["export_sha256"]
    assert checkpoint["gradient_compiler_options"] == report["gradient_compiler_options"] == OPTIONS
    assert digest(checkpoint["arrays"]) == checkpoint["arrays_sha256"]
    assert len(checkpoint["reference_validation"]) == 26
    assert all(c["nll_absolute_error"] < 1e-6 and c["value_gradient_nll_absolute_error"] < 1e-6
               for c in checkpoint["reference_validation"])
    if report["model"] == "hhvv":
        from compare_gpu_results import verified_refinement
        from gpu_gradient_evidence import random_validation
        nominal = verified_refinement(checkpoint_path, checkpoint)
        random = random_validation(checkpoint_path, checkpoint, digest(checkpoint_path), refined=True)
        assert nominal["status"] == nominal["directions_status"] == random["status"] == "passed"
    else:
        assert checkpoint["status"] == "passed"
        assert all(c["gradient_max_relative_error"] < 1e-4 for c in checkpoint["reference_validation"])
    return report, arrays


def check_configuration(likelihood, report):
    assert list(likelihood.parameter_names) == report["parameter_names"]
    for attribute, key in (("discrete_choices", "discrete_choices"),
                           ("mask_configuration", "channel_masks")):
        # Exported MultiPdf alternatives are tuples; JSON records them as lists.
        assert json.loads(json.dumps(getattr(likelihood, attribute))) == report[key], key


def full_covariance(hessian, free, scales):
    """Invert all free curvature entries, with no truncation or pseudoinverse."""
    hessian, free, scales = np.asarray(hessian), np.asarray(free), np.asarray(scales)
    size = len(scales)
    assert hessian.shape == (size, size) and hessian.dtype == np.float64
    assert free.ndim == 1 and np.issubdtype(free.dtype, np.integer)
    assert len(set(free)) == len(free) and np.all((free >= 0) & (free < size))
    assert np.isfinite(scales).all() and np.all(scales > 0)
    info = dict(status="nonfinite_curvature", free_parameters=len(free),
                method="Complete raw free Hessian inverse; no projection, regularization or pseudoinverse")
    if not np.isfinite(hessian).all():
        return None, info
    if not len(free):
        return np.zeros_like(hessian), dict(info, status="all_fixed")
    matrix = hessian[np.ix_(free, free)]
    scaled = matrix*np.outer(scales[free], scales[free])
    asymmetry = float(abs(scaled-scaled.T).max(initial=0.))
    info["maximum_parameter_scaled_asymmetry"] = asymmetry
    if asymmetry >= 1e-6:
        return None, dict(info, status="asymmetric_curvature")
    # Symmetrization is used only to diagnose eigenvalues. The inverse below
    # uses every original raw entry, and that exact matrix is saved unchanged.
    eigenvalues = np.linalg.eigvalsh(.5*(scaled+scaled.T))
    info.update(minimum_scaled_eigenvalue=float(eigenvalues[0]),
                maximum_scaled_eigenvalue=float(eigenvalues[-1]),
                nonpositive_scaled_eigenvalues=int(np.count_nonzero(eigenvalues <= 0)))
    if eigenvalues[0] <= 0:
        return None, dict(info, status="nonpositive_curvature")
    try:
        inverse = np.linalg.inv(matrix)
        if not np.isfinite(inverse).all() or np.any(np.diag(inverse) <= 0):
            return None, dict(info, status="unstable_inverse")
        sigma = np.sqrt(np.diag(inverse))
        normalized = inverse/np.outer(sigma, sigma)
        covariance_asymmetry = float(abs(normalized-normalized.T).max(initial=0.))
        identity_error = float(abs(matrix@inverse-np.eye(len(free))).max(initial=0.))
        np.linalg.cholesky(.5*(normalized+normalized.T))
    except np.linalg.LinAlgError:
        return None, dict(info, status="unstable_inverse")
    info.update(hessian_covariance_identity_absolute_error=identity_error,
                maximum_covariance_normalized_asymmetry=covariance_asymmetry)
    if identity_error >= 1e-6 or covariance_asymmetry >= 1e-6:
        return None, dict(info, status="unstable_inverse")
    result = np.zeros_like(hessian)
    result[np.ix_(free, free)] = inverse
    return result, dict(info, status="positive_definite")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nominal-fd-refinement", type=Path)
    parser.add_argument("--rss-limit-mib", type=float, default=3500.)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    args = parser.parse_args()
    if (args.output.parent != Path("reports") or args.output.suffix != ".json"
            or not 0 < args.rss_limit_mib <= 11000 or not 0 < args.wall_seconds < np.inf):
        parser.error("Use reports/*.json, RSS in (0, 11000] MiB and a finite positive wall limit")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This covariance continuation is local only")
    directory = Path("build/fit_results")/(args.output.stem+"_blocks")
    artifact = Path("build/fit_results")/(args.output.stem+".npz")
    if any(p.exists() for p in (args.output, directory, artifact)):
        raise FileExistsError("Preserve previous covariance evidence")
    started = time.perf_counter()
    minimum, saved = checked_minimum(args.minimum_report)
    minimum_checksum = digest(args.minimum_report)
    assert (args.nominal_fd_refinement is not None) == (minimum["model"] == "hhvv")
    assert "jax" not in sys.modules and "ROOT" not in sys.modules
    os.environ.update(JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                      JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS="false")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[0]])
    import jax
    import jaxlib
    from jaxlib import xla_client
    from combine_jax.local import release_compilation_cache
    from build_cpu_hessian_reference import validate_nominal_refinement
    from hhvv_export_cache import load_export
    from validate_cpu_no_fusion_hessian import all_columns

    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert minimum["jax_version"] == jax.__version__ and minimum["jaxlib_version"] == jaxlib.__version__
    assert xla_client.CompileOptions().executable_build_options.debug_options.xla_disable_hlo_passes == ""
    jax.config.update("jax_enable_compilation_cache", False)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    assert digest(manifest_path) == minimum["manifest_sha256"]
    manifest = json.loads(manifest_path.read_bytes())
    model = manifest["models"][minimum["model"]]
    export = Path("build/gpu_exports")/model["cache"]
    assert model["source"] == minimum["source"] and digest(export) == minimum["export_sha256"]
    gate_path = Path(f"reports/{minimum['model']}_cpu_no_fusion_hessian.json")
    gate = json.loads(gate_path.read_bytes())
    assert gate["source"] == minimum["source"] and gate["source_sha256"] == core_digest()
    assert gate["export_sha256"] == digest(export) and gate["hessian_compiler_options"] == OPTIONS
    assert gate["jax_version"] == jax.__version__ and gate["jaxlib_version"] == jaxlib.__version__
    assert all(digest(p) == checksum for p, checksum in gate["runner_sha256"].items())
    assert all(digest(c["arrays"]) == c["arrays_sha256"] for c in gate["cases"])
    refinement = validate_nominal_refinement(gate, gate_path, args.nominal_fd_refinement) if args.nominal_fd_refinement else None
    assert refinement or gate["status"] == "passed"
    frozen = {p: digest(p) for p in ("scripts/complete_bounded_cpu_covariance.py",
        "scripts/build_cpu_hessian_reference.py", "scripts/validate_cpu_no_fusion_hessian.py",
        "scripts/refine_cpu_hessian_fd.py", "scripts/compare_local_cpu_hessian.py",
        "scripts/compare_exported_fit.py", "scripts/hhvv_export_cache.py")}
    directory.mkdir(parents=True)
    report = copy.deepcopy(minimum)
    for key in ("arrays", "arrays_sha256", "exception", "preparation_progress"):
        report.pop(key, None)
    report.update(status="running", stage="load fit-point export",
        minimum_report=str(args.minimum_report), minimum_report_sha256=minimum_checksum,
        minimum_runner_sha256=minimum["runner_sha256"], runner_sha256=digest(__file__),
        minimum_elapsed_seconds=minimum["elapsed_seconds"], minimum_peak_rss_mib=minimum["peak_rss_mib"],
        covariance_source_files=frozen, covariance_requested=True, hessian_compiler_options=OPTIONS,
        constraint_batch_size=32, hessian_chunk_size=4,
        covariance_cpu_affinity=sorted(os.sched_getaffinity(0)), rss_limit_mib=args.rss_limit_mib,
        wall_limit_seconds=args.wall_seconds, validation_gate=str(gate_path),
        validation_gate_sha256=digest(gate_path), nominal_fd_refinement=refinement,
        curvature_point=saved["values"].tolist(), point=saved["values"].tolist(), terms=[],
        scope="Conditional minimum plus complete fit-point AD covariance. Minimum and curvature execute in separate local processes; every raw Hessian entry is retained. Native comparison and nuisance profile remain separate gates.",
        elapsed_scope="Sum of the recorded minimum run and this covariance continuation; each includes its initialization/preparation/validation/cleanup after CLI checks. Excludes Python startup, preceding imports and time between the two processes.")
    lock, stop = threading.RLock(), threading.Event()
    mapping_guard = int(Path("/proc/sys/vm/max_map_count").read_text())-4096
    report["mapping_guard"] = mapping_guard
    evaluator = None

    def save():
        with lock:
            report.update(covariance_elapsed_seconds=time.perf_counter()-started,
                covariance_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                mapping_count=len(Path("/proc/self/maps").read_text().splitlines()))
            report["elapsed_seconds"] = minimum["elapsed_seconds"]+report["covariance_elapsed_seconds"]
            report["peak_rss_mib"] = max(minimum["peak_rss_mib"], report["covariance_peak_rss_mib"])
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = started
        while not stop.wait(.25):
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            maps = len(Path("/proc/self/maps").read_text().splitlines())
            reason = ("RSS" if memory > args.rss_limit_mib else "mappings" if maps >= mapping_guard
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
                print(report["stage"], report["covariance_elapsed_seconds"],
                      report["covariance_peak_rss_mib"], flush=True)
                last = time.perf_counter()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        check_configuration(likelihood, minimum)
        point = saved["values"]
        fixed = {p.name: p.minimum for p in likelihood.parameters if p.minimum == p.maximum}
        assert fixed == minimum["final"]["fixed"] and set(fixed) == set(minimum["fixed_bounds"])
        for p, x in zip(likelihood.parameters, point, strict=True):
            assert p.minimum <= x <= p.maximum
        free = np.asarray([i for i, p in enumerate(likelihood.parameters) if p.name not in fixed], dtype=int)
        scales = np.asarray([p.error if p.error > 0 else max(abs(p.value), 1.) for p in likelihood.parameters])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        hessian = np.zeros((len(point), len(point)), dtype=np.float64)
        nll = float(likelihood.offset)
        report.update(expected_terms=len(evaluator.terms), parameters=len(point), channels=len(likelihood.channels))
        for index, term in enumerate(evaluator.terms):
            with lock:
                report["stage"] = f"fit-point NLL {index+1}/{len(evaluator.terms)} {term.name}"
                save()
            local_point = point[term.indices]
            begin = time.perf_counter()
            value_kernel = term.value.lower(local_point).compile()
            contribution = float(value_kernel(local_point))
            assert np.isfinite(contribution)
            nll += contribution
            del value_kernel
            release_compilation_cache()
            value_seconds = time.perf_counter()-begin
            with lock:
                report["stage"] = f"fit-point Hessian {index+1}/{len(evaluator.terms)} {term.name}"
                save()
            size = len(local_point)
            if size:
                begin = time.perf_counter()
                lowered = term.hessian_columns.lower(local_point, np.zeros((min(4, size), size)))
                lowering = time.perf_counter()-begin
                begin = time.perf_counter()
                compiled = lowered.compile(compiler_options=OPTIONS)
                compilation = time.perf_counter()-begin
                del lowered
                begin = time.perf_counter()
                block = all_columns(compiled, local_point)
                execution = time.perf_counter()-begin
                del compiled
            else:
                lowering = compilation = execution = 0.
                block = np.zeros((0, 0), dtype=np.float64)
            hessian[np.ix_(term.indices, term.indices)] += block
            block_path = directory/f"{index}.npz"
            np.savez_compressed(block_path, indices=term.indices, point=local_point, hessian=block, nll=contribution)
            begin = time.perf_counter()
            term.reset_kernel("hessian_columns")
            release_compilation_cache()
            cleanup = time.perf_counter()-begin
            with lock:
                report["terms"].append(dict(index=index, name=term.name, parameters=size,
                    value_prepare_and_execute_seconds=value_seconds, lowering_seconds=lowering,
                    compile_seconds=compilation, first_all_columns_seconds=execution,
                    cleanup_seconds=cleanup, arrays=str(block_path), arrays_sha256=digest(block_path)))
                save()
        nll_error = abs(nll-minimum["final"]["nll"])
        assert nll_error < 1e-6, "Fit-point canonical NLL differs from the validated minimum"
        assembly = reassemble_local(report, dict(point=point, hessian=hessian))
        with lock:
            report.update(stage="invert complete free Hessian", canonical_fit_point_nll=nll,
                          minimum_nll_absolute_error=nll_error, local_reassembly=assembly)
            save()
        begin = time.perf_counter()
        covariance, diagnostic = full_covariance(hessian, free, scales)
        covariance_seconds = time.perf_counter()-begin
        assert core_digest() == minimum["source_sha256"]
        assert digest(args.minimum_report) == minimum_checksum
        assert all(digest(p) == h for p, h in frozen.items())
        assert "ROOT" not in sys.modules
        temporary = artifact.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, parameter_names=np.asarray(likelihood.parameter_names),
            initial=saved["initial"], values=point, point=point, gradient=saved["gradient"],
            parameter_scales=scales, free_indices=free, hessian=hessian,
            covariance=np.empty((0, 0)) if covariance is None else covariance)
        temporary.replace(artifact)
        with lock:
            report["final"]["covariance_status"] = diagnostic["status"]
            report["timings"].update(covariance_inversion_seconds=covariance_seconds,
                **{key: sum(term[key] for term in report["terms"]) for key in (
                    "value_prepare_and_execute_seconds", "lowering_seconds", "compile_seconds",
                    "first_all_columns_seconds", "cleanup_seconds")})
            report.update(status="fit_computed" if covariance is not None else "fit_incomplete",
                stage="finished", arrays=str(artifact), arrays_sha256=digest(artifact),
                covariance_diagnostic=diagnostic, root_imported=False)
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
    print(args.output, report["status"], report.get("covariance_diagnostic"), flush=True)
    raise SystemExit(0 if report["status"] == "fit_computed" else 1)


if __name__ == "__main__":
    main()
