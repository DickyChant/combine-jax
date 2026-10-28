"""Class-level HHVV comparisons for floating templates, powers and fractions."""

from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
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
    rng = np.random.default_rng(8197)
    report = dict(source="inputs/workspace_MX2000.root", seed=8197,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="RooPower, RooParametricStepFunction, RooParametricHist2D and RooRecursiveFraction class comparisons; no HHVV NLL claim",
                  nodes=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        config = workspace.obj("ModelConfig")
        observables = {str(v.GetName()) for v in config.GetObservables()}
        classes = ("RooPower", "RooParametricStepFunction", "RooParametricHist2D", "RooRecursiveFraction")
        objects = [n for n in walk(config.GetPdf()).values() if n.ClassName() in classes]
        for index, obj in enumerate(objects):
            graph = registry.export([obj], observables)
            name = str(obj.GetName()); node = graph[name]
            axes = node.inputs[:1] if node.class_name == "RooPower" else node.attrs.get("axes", ())
            variables = [workspace.var(n) for n in axes]
            original_axes = [float(v.getVal()) for v in variables]
            if node.class_name == "RooPower":
                coordinates = {axes[0]: np.linspace(variables[0].getMin(), variables[0].getMax(), 21)}
            elif node.class_name == "RooParametricStepFunction":
                edge = node.attrs["edges"]
                coordinates = {axes[0]: (edge[:-1]+edge[1:])/2}
            elif node.class_name == "RooParametricHist2D":
                centers = [(e[:-1]+e[1:])/2 for e in node.attrs["edges"]]
                coordinates = dict(zip(axes, [v.reshape(-1) for v in np.meshgrid(*centers, indexing="ij")], strict=True))
            else:
                coordinates = {}
            info = tuple(p for p in parameters(obj, observables) if p.maximum > p.minimum)
            names = tuple(p.name for p in info); pars = [workspace.var(n) for n in names]
            initial = np.array([p.value for p in info])
            scales = np.array([min(p.error or max(.1*abs(p.value), .01), (p.maximum-p.minimum)/8) for p in info])
            # Some saved fit errors are ~1e-8 at a fraction's range boundary.
            # A step proportional only to those errors loses the subtraction
            # to roundoff; retain a physical-coordinate step floor.
            steps = np.maximum(2e-4*scales, 2e-5*np.maximum(np.abs(initial),1.))
            lower, upper = np.array([p.minimum for p in info]), np.array([p.maximum for p in info])
            points = [initial, *[np.clip(initial+rng.uniform(-2,2,len(info))*scales, lower, upper) for _ in range(2)]]
            norm = root.RooArgSet(*variables)
            integral = obj.createIntegral(norm) if axes else None
            size = len(next(iter(coordinates.values()))) if coordinates else 1
            weights = np.linspace(.7,1.3,size)
            def evaluate(theta):
                ctx = Evaluation(graph,registry,names,theta,coordinates)
                raw = jnp.atleast_1d(ctx.value(name))
                return (raw,ctx.integral(name,axes),ctx.density(name,axes)) if axes else (raw,)
            evaluate = jax.jit(evaluate)
            gradient = jax.jit(jax.grad(lambda t: jnp.atleast_1d(evaluate(t)[-1]) @ weights))
            def reference(theta):
                for v,p in zip(pars,theta,strict=True): v.setVal(float(p))
                raw,density=[],[]
                for row in range(size):
                    for v in variables: v.setVal(float(coordinates[v.GetName()][row]))
                    raw.append(float(obj.getVal()))
                    if axes: density.append(float(obj.getVal(norm)))
                return (np.array(raw),float(integral.getVal()),np.array(density)) if axes else (np.array(raw),)
            result = dict(name=name,class_name=node.class_name,axes=axes,parameters=names,
                          coordinates={k:v.tolist() for k,v in coordinates.items()},fd_steps=steps.tolist(),
                          bounds=[lower.tolist(),upper.tolist()],cases=[])
            try:
                for point in points:
                    actual,ref=evaluate(point),reference(point)
                    grad=np.asarray(gradient(point))
                    fd=finite_difference(lambda t:reference(t)[-1] @ weights,point,steps,bounds=(lower,upper))
                    assert all(np.all(np.isfinite(v)) for v in (*actual,*ref,grad,fd)),name
                    errors=[float(np.max(np.abs(a-b))) for a,b in zip(actual,ref,strict=True)]
                    result["cases"].append(dict(theta=point.tolist(),raw_max_absolute_error=errors[0],
                        integral_absolute_error=errors[1] if axes else 0.,density_max_absolute_error=errors[-1],
                        jax_gradient=grad.tolist(),combine_fd_gradient=fd.tolist(),
                        gradient_max_absolute_error=float(np.max(np.abs(grad-fd),initial=0)),
                        gradient_max_relative_error=float(np.max(np.abs(grad-fd)/np.maximum(np.abs(fd),1e-8),initial=0))))
            finally:
                for v,p in zip(pars,initial,strict=True):v.setVal(float(p))
                for v,p in zip(variables,original_axes,strict=True):v.setVal(p)
            report["nodes"].append(result)
            if (index+1)%20==0:print(f"Compared {index+1}/{len(objects)} remaining shape nodes",flush=True)
    report.update(tested_nodes=len(report["nodes"]),elapsed_seconds=time.perf_counter()-start,
                  finite_difference="Five-point central, or range-aware five-point one-sided at parameter boundaries")
    cases=[c for n in report["nodes"] for c in n["cases"]]
    for key in ("raw_max_absolute_error","integral_absolute_error","density_max_absolute_error","gradient_max_absolute_error","gradient_max_relative_error"):
        report[key]=max(c[key] for c in cases)
    Path("reports/hhvv_remaining_shapes.json").write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print({k:v for k,v in report.items() if k!="nodes"},flush=True)
    assert report["tested_nodes"]==105
    assert report["raw_max_absolute_error"]<1e-8
    assert report["integral_absolute_error"]<1e-8
    assert report["density_max_absolute_error"]<1e-8
    assert report["gradient_max_relative_error"]<1e-4


if __name__=="__main__":main()
