"""Compare all MX2000 one-dimensional fast vertical kernels against Combine v11."""

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
    root, registry = load_root(), default_registry()
    rng = np.random.default_rng(8192)
    report = dict(source="inputs/workspace_MX2000.root", seed=8192,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="FastVerticalInterpHistPdf and FastVerticalInterpHistPdf2; component values, no full HHVV NLL claim",
                  integral_reference="Exact bin quadrature of v11 values over observable bounds",
                  nodes=[], unsupported=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        histograms = [n for n in walk(workspace.pdf("model_s")).values()
                      if n.ClassName() in ("FastVerticalInterpHistPdf", "FastVerticalInterpHistPdf2")]
        for index, func in enumerate(histograms):
            observables = tuple(func.getObservables(workspace.obj("ModelConfig").GetObservables()))
            if len(observables) != 1:
                raise UnsupportedModelError(f"{func.GetName()}: expected one observable")
            observable, obsname = observables[0], str(observables[0].GetName())
            try:
                graph = registry.export([func], [obsname])
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(func.GetName()), reason=str(exc)))
                continue
            node = graph[str(func.GetName())]
            edges = node.attrs["edges"]
            coords = (edges[:-1]+edges[1:])/2
            overlaps = np.maximum(0., np.minimum(edges[1:], observable.getMax())
                                  - np.maximum(edges[:-1], observable.getMin()))
            info = parameters(func, [obsname])
            names = tuple(p.name for p in info)
            variables = [workspace.var(name) for name in names]
            initial = np.array([p.value for p in info])
            original_x = float(observable.getVal())
            normset = root.RooArgSet(observable)
            points = [initial]
            for _ in range(2):
                points.append(np.array([np.clip(p.value + rng.uniform(-2, 2)*(p.error or 1.),
                                                p.minimum, p.maximum) for p in info]))
            result = dict(name=str(func.GetName()), class_name=str(func.ClassName()), parameters=list(names),
                          active_bins=node.attrs["active"], full_bins=len(coords),
                          logarithmic=node.attrs["logarithmic"], cases=[])
            try:
                for point in points:
                    for variable, val in zip(variables, point, strict=True):
                        variable.setVal(float(val))
                    context = Evaluation(graph, registry, names, point, {obsname: coords})
                    values = np.asarray(context.density(node.name, (obsname,)))
                    ref_values = []
                    for coord in coords:
                        observable.setVal(float(coord)); ref_values.append(float(func.getVal(normset)))
                    ref_values = np.asarray(ref_values)
                    integral = float(context.integral(node.name, (obsname,)))
                    reference_integral = float(ref_values @ overlaps)
                    result["cases"].append(dict(theta=point.tolist(),
                        max_value_absolute_error=float(np.max(np.abs(values-ref_values))),
                        max_value_relative_error=float(np.max(np.abs(values-ref_values)/np.maximum(np.abs(ref_values), 1e-9))),
                        jax_integral=integral, combine_integral=reference_integral,
                        integral_absolute_error=abs(integral-reference_integral)))
            finally:
                for variable, val in zip(variables, initial, strict=True): variable.setVal(float(val))
                observable.setVal(original_x)
            report["nodes"].append(result)
            if (index+1) % 50 == 0:
                print(f"Compared {index+1}/{len(histograms)} fast vertical PDFs", flush=True)
    cases = [case for node in report["nodes"] for case in node["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]),
                  max_value_absolute_error=max(c["max_value_absolute_error"] for c in cases),
                  max_value_relative_error=max(c["max_value_relative_error"] for c in cases),
                  max_integral_absolute_error=max(c["integral_absolute_error"] for c in cases))
    Path("reports/hhvv_vertical.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k not in ("nodes", "unsupported")}, flush=True)
    assert not report["unsupported"]
    assert report["tested_nodes"] == 361
    assert report["max_value_relative_error"] < 1e-10
    assert report["max_integral_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
