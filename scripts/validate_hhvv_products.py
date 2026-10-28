"""Compare supported HHVV product shapes and explicitly inventory remaining gaps."""

from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from hhvv_setup import configure_hhvv
from combine_jax.graph import UnsupportedModelError, parameters, walk
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_root, open_workspace
from combine_jax.translators import default_registry


def main():
    start = time.perf_counter()
    root, registry = load_root(), default_registry()
    rng = np.random.default_rng(8195)
    report = dict(source="inputs/workspace_MX2000.root", seed=8195,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="Supported RooProdPdf shape values; unsupported graphs listed; no HHVV NLL claim",
                  nodes=[], unsupported=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        config = workspace.obj("ModelConfig")
        products = [n for n in walk(config.GetPdf()).values() if n.ClassName() == "RooProdPdf"]
        report["total_product_nodes"] = len(products)
        for index, pdf in enumerate(products):
            norm = pdf.getObservables(config.GetObservables())
            axes = sorted(norm, key=lambda v: str(v.GetName()))
            names = tuple(str(v.GetName()) for v in axes)
            try:
                graph = registry.export([pdf], names)
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(pdf.GetName()), reason=str(exc)))
                continue
            info = parameters(pdf, names)
            param_names = tuple(p.name for p in info)
            variables = [workspace.var(n) for n in param_names]
            initial = np.array([p.value for p in info])
            original_axes = [float(v.getVal()) for v in axes]
            coordinates = {str(v.GetName()): rng.uniform(v.getMin(), v.getMax(), 25) for v in axes}
            points = [initial, *[np.array([np.clip(p.value+rng.uniform(-2,2)*(p.error or 1.), p.minimum, p.maximum)
                                          for p in info]) for _ in range(2)]]
            result = dict(name=str(pdf.GetName()), axes=names, parameters=param_names,
                          coordinates={k: v.tolist() for k, v in coordinates.items()}, cases=[])
            try:
                for point in points:
                    for variable, value in zip(variables, point, strict=True): variable.setVal(float(value))
                    context = Evaluation(graph, registry, param_names, point, coordinates)
                    values = np.asarray(context.density(str(pdf.GetName()), names))
                    reference = []
                    for row in range(25):
                        for variable in axes: variable.setVal(float(coordinates[str(variable.GetName())][row]))
                        reference.append(float(pdf.getVal(norm)))
                    reference = np.array(reference)
                    assert np.all(np.isfinite(values)) and np.all(np.isfinite(reference)), pdf.GetName()
                    result["cases"].append(dict(theta=point.tolist(), jax=values.tolist(), combine=reference.tolist(),
                        max_value_absolute_error=float(np.max(np.abs(values-reference))),
                        max_value_relative_error=float(np.max(np.abs(values-reference)/np.maximum(np.abs(reference), 1e-9)))))
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(pdf.GetName()), reason=str(exc)))
                continue
            finally:
                for variable, value in zip(variables, initial, strict=True): variable.setVal(float(value))
                for variable, value in zip(axes, original_axes, strict=True): variable.setVal(value)
            report["nodes"].append(result)
            if (index+1) % 30 == 0:
                print(f"Visited {index+1}/{len(products)} products; compared {len(report['nodes'])}", flush=True)
    cases = [case for node in report["nodes"] for case in node["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]),
                  max_value_absolute_error=max(c["max_value_absolute_error"] for c in cases),
                  max_value_relative_error=max(c["max_value_relative_error"] for c in cases))
    Path("reports/hhvv_products.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k:v for k,v in report.items() if k not in ("nodes", "unsupported")}, flush=True)
    print(f"Unsupported graphs: {len(report['unsupported'])}", flush=True)
    assert report["tested_nodes"] + len(report["unsupported"]) == report["total_product_nodes"]
    assert not report["unsupported"]
    assert report["tested_nodes"] == 354
    assert report["max_value_relative_error"] < 1e-8


if __name__ == "__main__":
    main()
