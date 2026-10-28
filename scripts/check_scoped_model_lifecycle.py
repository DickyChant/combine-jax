"""Validate scoped loads on both exported smoke models; no throughput claims."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    args = parser.parse_args()
    output = Path(f"reports/gpu_scoped_{args.backend}_model_lifecycle.json")
    directory = Path(f"build/gpu_results/scoped_{args.backend}_model_lifecycle")
    if output.exists() or directory.exists():
        raise FileExistsError("Preserve prior scoped model checks")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        assert platform.node().split(".")[0] == "lxplus905"
        batch = Path("reports/gpu_spooled_hhvv_driver.json")
        if batch.exists() and json.loads(batch.read_text())["status"] == "running":
            raise RuntimeError("Finish the active benchmark batch before this separate lifecycle check")
    directory.mkdir(parents=True)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    os.environ["JAX_PLATFORMS"] = args.backend
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    import jax
    from combine_jax.local import release_compilation_cache
    from hhvv_export_cache import load_export
    from scoped_executable import hessian_with_scoped_loads, value_gradient_points_with_scoped_loads
    from spooled_executable import SpooledExecutable

    devices = jax.devices()
    assert jax.default_backend() == args.backend and jax.config.jax_enable_x64
    assert "ROOT" not in sys.modules
    if args.backend == "gpu":
        assert Path.cwd() == Path("/tmp/sqian/combine-jax")
        assert all(d.platform == "gpu" for d in devices)
    jax.config.update("jax_enable_compilation_cache", False)
    manifest_path = Path("build/gpu_exports/manifest_random.json")
    manifest = json.loads(manifest_path.read_text())
    started = time.perf_counter()
    report = dict(status="running", backend=args.backend, host=platform.node(),
        workdir=str(Path.cwd()), lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"),
        python=platform.python_version(), jax_version=jax.__version__,
        jaxlib_version=__import__("jaxlib").__version__, float64=True,
        devices=[str(d) for d in devices], cpu_affinity=sorted(os.sched_getaffinity(0)),
        manifest_sha256=digest(manifest_path),
        source_sha256=hashlib.sha256(b"".join(str(p).encode()+b"\0"+p.read_bytes()
            for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
        runner_sha256={str(p): digest(p) for p in [Path(__file__),
            Path("scripts/scoped_executable.py"), Path("scripts/spooled_executable.py")]},
        export_stack={k: manifest[k] for k in ("combine_version", "root_version")},
        scope="Functional scoped-executable validation: both complete smoke models, all 26 NLL/gradient points and three full Hessians. No standalone call latency or performance comparison.",
        models=[])
    if args.backend == "gpu":
        report["gpu_metadata"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader"],
            text=True).strip()

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    save()
    try:
        for label in ("counting", "automcstats"):
            model = manifest["models"][label]
            likelihood = load_export(Path("build/gpu_exports")/model["cache"], model["source"])
            evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
            points = np.asarray([c["theta"] for c in model["reference"]["cases"]])
            baseline = [evaluator.value_and_grad(p) for p in points]
            expected_values = np.asarray([b[0] for b in baseline])
            expected_gradients = np.asarray([b[1] for b in baseline])
            for index, term in enumerate(evaluator.terms):
                lowered = term.value_and_grad.lower(points[0, term.indices])
                compiled = lowered.compile()
                stored, _ = SpooledExecutable.write(compiled,
                    directory/f"{label}_gradient_{index}.bin", backend=args.backend)
                term.value_and_grad = stored
                del compiled, lowered
                release_compilation_cache()
            values, gradients = value_gradient_points_with_scoped_loads(evaluator, points)
            assert np.isfinite(values).all() and np.isfinite(gradients).all()
            np.testing.assert_array_equal(values, expected_values)
            np.testing.assert_array_equal(gradients, expected_gradients)
            nll_errors = [abs(v-c["combine_nll"]) for v, c in zip(values, model["reference"]["cases"], strict=True)]
            gradient_errors = [float(np.max(abs(g-np.asarray(c["combine_fd_gradient"]))/
                np.maximum(abs(np.asarray(c["combine_fd_gradient"])), 1e-8), initial=0.))
                for g, c in zip(gradients, model["reference"]["cases"], strict=True)]
            assert max(nll_errors) < 1e-6 and max(gradient_errors) < 1e-4
            gradient_scopes = [len(t.value_and_grad.scoped_loads) for t in evaluator.terms]
            assert gradient_scopes == [1]*len(evaluator.terms)
            evaluator.clear_cache(release_cache=True)
            hessian_indices = [i for i, c in enumerate(model["reference"]["cases"]) if "hessian" in c]
            assert len(hessian_indices) == 3
            expected_hessians = np.asarray([evaluator.hessian(points[i]) for i in hessian_indices])
            for index, term in enumerate(evaluator.terms):
                size = len(term.indices)
                if not size:
                    continue
                lowered = term.hessian_columns.lower(points[0, term.indices], np.zeros((min(4, size), size)))
                compiled = lowered.compile()
                stored, _ = SpooledExecutable.write(compiled,
                    directory/f"{label}_hessian_{index}.bin", backend=args.backend)
                term.hessian_columns = stored
                del compiled, lowered
                release_compilation_cache()
            hessians = np.asarray([hessian_with_scoped_loads(evaluator, points[i]) for i in hessian_indices])
            np.testing.assert_array_equal(hessians, expected_hessians)
            scales = np.asarray(model["reference"]["parameter_scales"])
            hessian_errors = [float(np.max(abs(h-np.asarray(model["reference"]["cases"][i]["hessian"]))*
                np.outer(scales, scales), initial=0.)) for h, i in zip(hessians, hessian_indices, strict=True)]
            assert max(hessian_errors) < 1e-10
            hessian_scopes = [len(t.hessian_columns.scoped_loads) for t in evaluator.terms if len(t.indices)]
            assert hessian_scopes == [3]*len(hessian_scopes)
            artifact = directory/f"{label}.npz"
            np.savez_compressed(artifact, parameter_names=np.asarray(likelihood.parameter_names),
                points=points, values=values, gradients=gradients, hessian_point_indices=hessian_indices,
                hessians=hessians)
            report["models"].append(dict(model=label, source=model["source"], status="passed",
                reference_points=len(points), hessian_points=len(hessian_indices),
                max_combine_nll_absolute_error=float(max(nll_errors)),
                max_combine_gradient_relative_error=float(max(gradient_errors)),
                max_saved_cpu_hessian_scaled_error=float(max(hessian_errors)),
                retained_and_scoped_bitwise_equal=True,
                gradient_load_scopes_per_term=gradient_scopes,
                hessian_load_scopes_per_nonempty_term=hessian_scopes,
                arrays=str(artifact), arrays_sha256=digest(artifact)))
            evaluator.clear_cache(release_cache=True)
            save()
        assert "ROOT" not in sys.modules
        report.update(status="passed", root_imported=False)
        save()
        print(report["status"], report["models"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
