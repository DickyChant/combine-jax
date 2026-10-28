"""Validate DoubleCB/ErfExp values, integrals and density derivatives in HHVV."""

from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import numpy as np

from hhvv_setup import configure_hhvv
from combine_jax.graph import parameters, walk
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_root, open_workspace
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def main():
    start = time.perf_counter()
    root, registry = load_root(), default_registry()
    rng = np.random.default_rng(8196)
    report = dict(source="inputs/workspace_MX2000.root", seed=8196,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="DoubleCB/ErfExp class values, direct-x integrals and density-projection gradients; no HHVV NLL claim",
                  nodes=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        pdfs = [n for n in walk(workspace.obj("ModelConfig").GetPdf()).values()
                if n.ClassName() in ("RooDoubleCB", "RooErfExpPdf")]
        for index, pdf in enumerate(pdfs):
            graph = registry.export([pdf])
            name = str(pdf.GetName()); node = graph[name]
            axis_name = node.attrs["fields"][0] if node.class_name == "RooDoubleCB" else node.inputs[0]
            axis = workspace.var(axis_name)
            if not axis:
                raise ValueError(f"{name}: class probe needs a directly assignable x")
            # Conditional shapes keep their other event coordinates at saved
            # values; those are observables, never varied as nuisance parameters.
            observables = {str(v.GetName()) for v in workspace.obj("ModelConfig").GetObservables()}
            graph = registry.export([pdf], [axis_name])
            info = tuple(p for p in parameters(pdf, observables) if p.maximum > p.minimum)
            names = tuple(p.name for p in info); variables = [workspace.var(n) for n in names]
            initial = np.array([p.value for p in info]); original_axis = float(axis.getVal())
            scales = np.array([min(p.error or 1., (p.maximum-p.minimum)/8) for p in info])
            steps = scales*2e-4
            lower, upper = np.array([p.minimum for p in info]), np.array([p.maximum for p in info])
            if np.any(initial-2*steps < lower) or np.any(initial+2*steps > upper):
                raise ValueError(f"{name}: nominal point lacks two-sided finite-difference room")
            points = [initial, *[np.clip(initial+rng.uniform(-2, 2, len(info))*scales, lower+3*steps, upper-3*steps) for _ in range(2)]]
            coordinates = np.linspace(axis.getMin(), axis.getMax(), 19)
            weights = np.linspace(.7, 1.3, len(coordinates))
            norm = root.RooArgSet(axis); integral = pdf.createIntegral(norm)
            def evaluate(theta):
                ctx = Evaluation(graph, registry, names, theta, {axis_name: coordinates})
                return ctx.value(name), ctx.integral(name, (axis_name,)), ctx.density(name, (axis_name,))
            evaluate = jax.jit(evaluate)
            gradient = jax.jit(jax.grad(lambda theta: evaluate(theta)[2] @ weights))
            def reference(theta):
                for variable, value in zip(variables, theta, strict=True): variable.setVal(float(value))
                raw, density = [], []
                for coordinate in coordinates:
                    axis.setVal(float(coordinate)); raw.append(pdf.getVal()); density.append(pdf.getVal(norm))
                return np.array(raw), integral.getVal(), np.array(density)
            result = dict(name=name, class_name=node.class_name, axis=axis_name, parameters=names,
                          coordinates=coordinates.tolist(), projection_weights=weights.tolist(), fd_steps=steps.tolist(), cases=[])
            try:
                for point in points:
                    actual, ref = evaluate(point), reference(point)
                    grad = np.asarray(gradient(point))
                    fd = finite_difference(lambda t: reference(t)[2] @ weights, point, steps)
                    assert all(np.all(np.isfinite(v)) for v in (*actual, *ref, grad, fd)), name
                    absolute = np.abs(grad-fd); relative = absolute/np.maximum(np.abs(fd), 1e-8)
                    result["cases"].append(dict(theta=point.tolist(),
                        raw_max_absolute_error=float(np.max(np.abs(actual[0]-ref[0]))),
                        density_max_absolute_error=float(np.max(np.abs(actual[2]-ref[2]))),
                        integral_absolute_error=abs(float(actual[1])-ref[1]),
                        integral_relative_error=abs(float(actual[1])-ref[1])/max(abs(ref[1]),1e-9),
                        jax_gradient=grad.tolist(), combine_fd_gradient=fd.tolist(),
                        gradient_max_absolute_error=float(np.max(absolute, initial=0)),
                        gradient_max_relative_error=float(np.max(relative, initial=0))))
            finally:
                for variable, value in zip(variables, initial, strict=True): variable.setVal(float(value))
                axis.setVal(original_axis)
            report["nodes"].append(result)
            if (index+1) % 30 == 0:
                print(f"Compared {index+1}/{len(pdfs)} resonance PDFs", flush=True)
    cases = [c for n in report["nodes"] for c in n["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]))
    for metric in ("raw_max_absolute_error", "density_max_absolute_error", "integral_absolute_error", "integral_relative_error",
                   "gradient_max_absolute_error", "gradient_max_relative_error"):
        report[metric] = max(c[metric] for c in cases)
    Path("reports/hhvv_resonance.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k:v for k,v in report.items() if k != "nodes"}, flush=True)
    assert report["tested_nodes"] == 216
    assert report["raw_max_absolute_error"] < 1e-8
    assert report["density_max_absolute_error"] < 1e-8
    assert report["integral_relative_error"] < 1e-8
    assert report["gradient_max_relative_error"] < 1e-4


if __name__ == "__main__":
    main()
