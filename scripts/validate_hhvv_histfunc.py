"""Compare every MX2000 CMSHistFunc at nominal and seeded nuisance points.

This checks the class kernels, not the HHVV likelihood or autoMCStats profiling.
"""

from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from hhvv_setup import configure_hhvv

from combine_jax.graph import UnsupportedModelError, parameters, walk
from combine_jax.registry import Evaluation
from combine_jax.root_io import open_workspace, load_root
from combine_jax.translators import default_registry


def main():
    start = time.perf_counter()
    root = load_root()
    registry = default_registry()
    rng = np.random.default_rng(8161)
    report = dict(source="inputs/workspace_MX2000.root", seed=8161,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="CMSHistFunc values and integrals only; no HHVV NLL claim", nodes=[], unsupported=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        histograms = [node for node in walk(workspace.pdf("model_s")).values()
                      if node.ClassName() == "CMSHistFunc"]
        for index, func in enumerate(histograms):
            observable = func.getXVar()
            obsname = str(observable.GetName())
            try:
                graph = registry.export([func], [obsname])
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(func.GetName()), reason=str(exc)))
                continue
            node = graph[str(func.GetName())]
            edges = node.attrs["edges"]
            coords = (edges[1:] + edges[:-1])/2
            param = parameters(func, [obsname])
            param_names = tuple(p.name for p in param)
            initial = np.array([p.value for p in param])
            variables = [workspace.var(name) for name in param_names]
            original_x = float(observable.getVal())
            integral = func.createIntegral(root.RooArgSet(observable))
            points = [initial]
            for _ in range(2):
                points.append(np.array([np.clip(p.value + rng.uniform(-2, 2)*(p.error or 1.),
                                                p.minimum, p.maximum) for p in param]))
            result = dict(name=str(func.GetName()), parameters=list(param_names), cases=[])
            try:
                for point in points:
                    for variable, val in zip(variables, point, strict=True):
                        variable.setVal(float(val))
                    context = Evaluation(graph, registry, param_names, point, {obsname: coords})
                    values = np.asarray(context.value(node.name))
                    ref_values = []
                    for coord in coords:
                        observable.setVal(float(coord)); ref_values.append(float(func.getVal()))
                    ref_values = np.asarray(ref_values)
                    jax_integral = float(context.integral(node.name, (obsname,)))
                    ref_integral = float(integral.getVal())
                    absolute = np.max(np.abs(values-ref_values))
                    relative = np.max(np.abs(values-ref_values) / np.maximum(np.abs(ref_values), 1e-9))
                    result["cases"].append(dict(theta=point.tolist(), max_value_absolute_error=float(absolute),
                                                max_value_relative_error=float(relative),
                                                jax_integral=jax_integral, combine_integral=ref_integral,
                                                integral_absolute_error=abs(jax_integral-ref_integral)))
            finally:
                for variable, val in zip(variables, initial, strict=True):
                    variable.setVal(float(val))
                observable.setVal(original_x)
            report["nodes"].append(result)
            if (index+1) % 100 == 0:
                print(f"Compared {index+1}/{len(histograms)} CMSHistFunc nodes", flush=True)
    cases = [case for node in report["nodes"] for case in node["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]),
                  max_value_absolute_error=max(c["max_value_absolute_error"] for c in cases),
                  max_value_relative_error=max(c["max_value_relative_error"] for c in cases),
                  max_integral_absolute_error=max(c["integral_absolute_error"] for c in cases))
    Path("reports/hhvv_histfunc.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print({key: value for key, value in report.items() if key not in ("nodes", "unsupported")}, flush=True)
    assert not report["unsupported"]
    assert report["tested_nodes"] == 546
    assert report["max_value_relative_error"] < 1e-12
    assert report["max_integral_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
