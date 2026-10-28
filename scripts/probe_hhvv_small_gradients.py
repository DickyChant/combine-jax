"""Diagnose preselected small HHVV derivatives with step-dependent Combine FDs.

Only the 25 small derivatives selected by the independent reference scan are
evaluated. This is a local CPU diagnostic, not the full GPU acceptance gate.
"""

import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from hhvv_export_cache import load_export


def main():
    started = time.perf_counter()
    reference_path = Path("reports/gpu_combine_gradient_refined.json")
    reference = json.loads(reference_path.read_text())
    selected = [c for c in reference["cases"] if c["step_strategy"].startswith("Larger")]
    names = tuple(c["name"] for c in selected)
    likelihood = load_export("build/gpu_exports/hhvv.pkl", reference["source"])
    evaluator = likelihood.local_evaluator(constraint_batch_size=32)
    np.testing.assert_array_equal(likelihood.initial, reference["initial"])
    np.testing.assert_array_equal(likelihood.parameter_names, reference["parameter_names"])
    assert jax.config.jax_enable_x64 and all(d.platform == "cpu" for d in jax.devices())
    gradient = np.zeros(len(selected))
    report = dict(status="running", source=reference["source"], jax_version=jax.__version__,
                  float64=True, devices=[str(d) for d in jax.devices()],
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  reference=str(reference_path), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  selection="The 25 reference-only small-gradient cases scanned with larger steps before AD was available",
                  scope="Local CPU partial-derivative diagnostic only; no full-model gradient, Hessian or GPU acceptance",
                  parameter_names=names, terms=[], cases=[])
    output = Path("reports/hhvv_small_gradients.json")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        for term in evaluator.terms:
            local = np.array([i for i, n in enumerate(term.parameter_names) if n in names], dtype=np.intp)
            if not len(local):
                continue
            destinations = np.array([names.index(term.parameter_names[i]) for i in local])
            point = jnp.asarray(likelihood.initial[term.indices])

            def partial(x):
                return term.function(point.at[local].set(x))

            start = time.perf_counter()
            function = jax.jit(jax.value_and_grad(partial))
            value, derivative = function(point[local])
            value, derivative = float(value), np.asarray(derivative)
            assert np.isfinite(value) and np.isfinite(derivative).all()
            gradient[destinations] += derivative
            report["terms"].append(dict(name=term.name, parameters=[term.parameter_names[i] for i in local],
                                        value=value, partial_gradient=derivative.tolist(),
                                        compile_and_execute_seconds=time.perf_counter()-start))
            save()
            print(term.name, "parameters", len(local), "elapsed", report["elapsed_seconds"], flush=True)
            if report["peak_rss_mib"] > 11500:
                raise MemoryError("11,500 MiB RSS guard exceeded")
            del function, value, derivative, point, partial
            jax.clear_caches()
            gc.collect()
            ctypes.CDLL(None).malloc_trim(0)
        for i, case in enumerate(selected):
            probes = []
            for p in case["probes"]:
                error = float(abs(gradient[i]-p["derivative"]))
                probes.append(dict(step=p["step"], combine=p["derivative"], absolute_error=error,
                                   relative_error=error/max(abs(p["derivative"]), 1e-8)))
            report["cases"].append(dict(name=case["name"], ad=float(gradient[i]), probes=probes))
        report.update(status="diagnostic_complete", note="Every step is retained; no closest-step selection or tolerance relaxation")
        save()
        print(report["status"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
