"""Diagnose the failed nominal compiler check with a fixed finer FD grid.

Reference convergence is selected before reading the saved autodiff Hessian.
The original failed validation is never overwritten.
"""

import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback

import jax
from jaxlib import xla_client
import numpy as np

from combine_jax.local import release_compilation_cache
from hhvv_export_cache import load_export


STEPS = (1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 1e-7, 3e-8, 1e-8)
TOLERANCE = 1e-4


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scaled_error(a, b, scales):
    outer = np.outer(scales, scales)
    a, b = a*outer, b*outer
    return float(np.max(abs(a-b)/np.maximum(1., abs(b)), initial=0.))


def first_converged_triple(matrices, scales):
    for first in range(len(matrices)-2):
        indices = (first, first+1, first+2)
        errors = [scaled_error(matrices[i], matrices[j], scales)
                  for i, j in ((first, first+1), (first+1, first+2), (first, first+2))]
        if max(errors) < TOLERANCE:
            return dict(indices=indices, steps=[STEPS[i] for i in indices],
                        reference_pair_errors=errors)
    return None


def main():
    output = Path("reports/hhvv_cpu_no_fusion_hessian_finer_fd.json")
    artifact = Path("build/gpu_results/hhvv_cpu_no_fusion_hessian_finer_fd.npz")
    if output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous finer-grid evidence")
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    jax.config.update("jax_enable_compilation_cache", False)
    assert jax.default_backend() == "cpu" and jax.config.jax_enable_x64
    assert xla_client.CompileOptions().executable_build_options.debug_options.xla_disable_hlo_passes == ""
    parent_path = Path("reports/hhvv_cpu_no_fusion_hessian.json")
    parent = json.loads(parent_path.read_text())
    assert parent["status"] == "failed"
    failed = [c for c in parent["cases"] if c["status"] == "failed"]
    assert len(failed) == 1
    case = failed[0]
    assert not case["finite_difference_checks"][0]["passed"]
    assert all(c["passed"] for c in case["finite_difference_checks"][1:])
    assert digest(case["arrays"]) == case["arrays_sha256"]
    source_hash = hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest()
    assert parent["source_sha256"] == source_hash and parent["jax_version"] == jax.__version__
    with np.load(case["arrays"], allow_pickle=False) as saved:
        point = saved["points"][0].copy()
        names = saved["parameter_names"].copy()
        free = saved["free_indices"].copy()
        scales = saved["parameter_scales"].copy()
        anchor = saved["default_gradient_fd_hessians"][0, -1].copy()
        # Do not access original_hessians until after FD-only selection.
    manifest_path = Path(parent["manifest"])
    assert digest(manifest_path) == parent["manifest_sha256"]
    model = json.loads(manifest_path.read_text())["models"]["hhvv"]
    export = Path("build/gpu_exports")/model["cache"]
    assert digest(export) == parent["export_sha256"]
    started = time.perf_counter()
    report = dict(status="running", stage="setup", model="hhvv", source=parent["source"],
        parent=str(parent_path), parent_sha256=digest(parent_path),
        parent_arrays=case["arrays"], parent_arrays_sha256=case["arrays_sha256"],
        runner_sha256=digest(__file__), source_sha256=source_hash,
        jax_version=jax.__version__, jaxlib_version=__import__("jaxlib").__version__,
        float64=True, backend="cpu", cpu_affinity=sorted(os.sched_getaffinity(0)),
        xla_flags=os.environ.get("XLA_FLAGS"), default_disabled_hlo_passes="",
        rss_limit_mib=3500, wall_limit_seconds=600,
        term_index=case["index"], term_name=case["name"], point_index=0,
        parameters=len(point), free_parameters=len(free), steps=STEPS,
        policy="All free nominal columns at nine fixed steps; first adjacent triple whose three reference-only pair comparisons are below 1e-4. Compare all three with saved AD only after selection; never advance to a later triple because AD failed.",
        scope="Supplemental local compiler-option check only; original failure retained. No full-model, Combine curvature or GPU acceptance.",
        progress=[])
    stop = threading.Event()
    lock = threading.RLock()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def monitor():
        while not stop.wait(.1):
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            elapsed = time.perf_counter()-started
            if memory > report["rss_limit_mib"] or elapsed > report["wall_limit_seconds"]:
                report.update(status="resource_limit",
                              exception="Own RSS or wall-time guard exceeded")
                save()
                os._exit(86)

    save()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        likelihood = load_export(export, model["source"])
        evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
        term = evaluator.terms[case["index"]]
        assert term.name == case["name"]
        np.testing.assert_array_equal(names, term.parameter_names)
        np.testing.assert_array_equal(point, likelihood.initial[term.indices])
        parameters = [likelihood.parameters[i] for i in term.indices]
        report["stage"] = "compile default gradient"
        save()
        begin = time.perf_counter()
        gradient = term.value_and_grad.lower(point).compile()
        report["default_gradient_preparation_seconds"] = time.perf_counter()-begin
        release_compilation_cache()
        matrices, samples = [], []
        for step in STEPS:
            report["stage"] = f"finite differences {step}"
            save()
            matrix = np.empty((len(free), len(free)))
            stencil = np.empty((len(free), 4, len(free)))
            begin = time.perf_counter()
            for column, coordinate in enumerate(free):
                h = step*scales[coordinate]
                parameter = parameters[coordinate]
                assert parameter.minimum <= point[coordinate]-2*h < point[coordinate]+2*h <= parameter.maximum
                for probe, multiplier in enumerate((-2., -1., 1., 2.)):
                    displaced = point.copy()
                    displaced[coordinate] += multiplier*h
                    value, derivative = gradient(displaced)
                    assert np.isfinite(float(value)) and np.isfinite(np.asarray(derivative)).all()
                    stencil[column, probe] = np.asarray(derivative)[free]
                a, b, c, d = stencil[column]
                matrix[:, column] = (a-8*b+8*c-d)/(12*h)
            matrices.append(matrix)
            samples.append(stencil)
            report["progress"].append(dict(step=step, seconds=time.perf_counter()-begin))
            save()
            print("completed step", step, flush=True)
        matrices = np.asarray(matrices)
        report["anchor_scaled_error"] = scaled_error(matrices[0], anchor, scales[free])
        assert report["anchor_scaled_error"] < 1e-10
        selection = first_converged_triple(matrices, scales[free])
        report.update(selection=selection, selection_completed_before_reading_ad_hessian=True)
        save()
        with np.load(case["arrays"], allow_pickle=False) as saved:
            hessian = saved["original_hessians"][0][np.ix_(free, free)].copy()
        errors = [scaled_error(hessian, m, scales[free]) for m in matrices]
        report["ad_errors_at_every_step"] = errors
        if selection is not None:
            selected_errors = [errors[i] for i in selection["indices"]]
            report["selected_ad_errors"] = selected_errors
            report["status"] = "passed" if max(selected_errors) < TOLERANCE else "failed"
        else:
            report["status"] = "reference_unresolved"
        np.savez_compressed(artifact, point=point, parameter_names=names, free_indices=free,
            parameter_scales=scales, steps=STEPS, fd_hessians=matrices,
            gradient_stencils=np.asarray(samples), saved_ad_hessian=hessian)
        report.update(stage="finished", arrays=str(artifact), arrays_sha256=digest(artifact))
        save()
        print(report["status"], report["selection"], report["ad_errors_at_every_step"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()


if __name__ == "__main__":
    main()
