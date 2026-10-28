"""Compare all MX2000 CMSHistErrorPropagator values and integrals to v11.

These are explicit-nuisance class checks, not a whole-HHVV likelihood comparison.
"""

from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from hhvv_setup import configure_hhvv

from combine_jax.bin_statistics import geometry
from combine_jax.graph import UnsupportedModelError, parameters, walk
from combine_jax.registry import Evaluation
from combine_jax.root_io import open_workspace, load_root
from combine_jax.translators import default_registry


def main():
    start = time.perf_counter()
    root, registry = load_root(), default_registry()
    rng = np.random.default_rng(6018)
    report = dict(source="inputs/workspace_MX2000.root", seed=6018,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="CMSHistErrorPropagator explicit-nuisance class kernels only", nodes=[], unsupported=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        propagators = [n for n in walk(workspace.pdf("model_s")).values() if n.ClassName() == "CMSHistErrorPropagator"]
        for index, func in enumerate(propagators):
            try:
                graph = registry.export([func], ["CMS_th1x"])
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(func.GetName()), reason=str(exc)))
                continue
            node = graph[str(func.GetName())]
            attrs = geometry(node, Evaluation(graph, registry, (), ()))
            observable = workspace.var(node.attrs["observable_name"])
            coords = (attrs["edges"][1:]+attrs["edges"][:-1])/2
            info = parameters(func, [observable.GetName()])
            names = tuple(p.name for p in info)
            variables = [workspace.var(name) for name in names]
            initial = np.array([p.value for p in info])
            original_x = float(observable.getVal())
            root_integral = func.createIntegral(root.RooArgSet(observable))
            points = [initial]
            for _ in range(2):
                points.append(np.array([np.clip(p.value + rng.uniform(-2, 2)*(p.error or 1.),
                                                p.minimum, p.maximum) for p in info]))
            result = dict(name=str(func.GetName()), parameters=list(names),
                          bin_types=node.attrs["bin_types"], cases=[])
            try:
                for point in points:
                    for variable, val in zip(variables, point, strict=True):
                        variable.setVal(float(val))
                    context = Evaluation(graph, registry, names, point, {observable.GetName(): coords})
                    values = np.asarray(context.value(node.name))
                    ref_values = []
                    for coord in coords:
                        observable.setVal(float(coord)); ref_values.append(float(func.getVal()))
                    ref_values = np.asarray(ref_values)
                    integral = float(context.integral(node.name, (observable.GetName(),)))
                    reference_integral = float(root_integral.getVal())
                    result["cases"].append(dict(theta=point.tolist(),
                        max_value_absolute_error=float(np.max(np.abs(values-ref_values))),
                        max_value_relative_error=float(np.max(np.abs(values-ref_values)/np.maximum(np.abs(ref_values), 1e-9))),
                        jax_integral=integral, combine_integral=reference_integral,
                        integral_absolute_error=abs(integral-reference_integral)))
            finally:
                for variable, val in zip(variables, initial, strict=True): variable.setVal(float(val))
                observable.setVal(original_x)
            report["nodes"].append(result)
            if (index+1) % 30 == 0:
                print(f"Compared {index+1}/{len(propagators)} error propagators", flush=True)
    cases = [case for node in report["nodes"] for case in node["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]),
                  max_value_absolute_error=max(c["max_value_absolute_error"] for c in cases),
                  max_value_relative_error=max(c["max_value_relative_error"] for c in cases),
                  max_integral_absolute_error=max(c["integral_absolute_error"] for c in cases))
    Path("reports/hhvv_errors.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k not in ("nodes", "unsupported")}, flush=True)
    assert not report["unsupported"]
    assert report["tested_nodes"] == 182
    assert report["max_value_relative_error"] < 1e-10
    assert report["max_integral_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
