"""Validate all MX2000 2D/3D fast vertical component values against Combine v11."""

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
    root.gInterpreter.Declare(r'''
      namespace combine_jax_validation {
      std::vector<double> sampleGrid(RooAbsPdf const& pdf, RooArgList const& axes,
                                     std::vector<std::vector<double>> const& centers) {
        RooArgSet norm(axes);
        std::vector<RooRealVar*> variables;
        size_t size = 1;
        for (int d=0; d<axes.size(); ++d) {
          variables.push_back(dynamic_cast<RooRealVar*>(axes.at(d)));
          size *= centers[d].size();
        }
        std::vector<double> result(size);
        for (size_t i=0; i<size; ++i) {
          size_t index = i;
          for (int d=axes.size()-1; d>=0; --d) {
            variables[d]->setVal(centers[d][index % centers[d].size()]);
            index /= centers[d].size();
          }
          result[i] = pdf.getVal(&norm);
        }
        return result;
      }
      }
    ''')
    rng = np.random.default_rng(8193)
    report = dict(source="inputs/workspace_MX2000.root", seed=8193,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="2D/3D fast vertical component values only; no full HHVV NLL claim",
                  integral_reference="Exact bin quadrature of v11 values over observable bounds",
                  nodes=[], unsupported=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        classes = ("FastVerticalInterpHistPdf2D", "FastVerticalInterpHistPdf2D2", "FastVerticalInterpHistPdf3D")
        histograms = [n for n in walk(workspace.pdf("model_s")).values() if n.ClassName() in classes]
        for index, pdf in enumerate(histograms):
            axes = [pdf.x(), pdf.y()] + ([pdf.z()] if pdf.ClassName().endswith("3D") else [])
            names = tuple(str(v.GetName()) for v in axes)
            try:
                graph = registry.export([pdf], names)
            except UnsupportedModelError as exc:
                report["unsupported"].append(dict(name=str(pdf.GetName()), reason=str(exc)))
                continue
            node = graph[str(pdf.GetName())]
            centers, widths = [], []
            for axis, edge in zip(axes, node.attrs["edges"], strict=True):
                low, high = np.maximum(edge[:-1], axis.getMin()), np.minimum(edge[1:], axis.getMax())
                keep = high > low
                centers.append((low[keep]+high[keep])/2)
                widths.append(high[keep]-low[keep])
            if any(len(c) == 0 for c in centers):
                raise UnsupportedModelError(f"{node.name}: no histogram bins within observable bounds")
            coordinates = [v.reshape(-1) for v in np.meshgrid(*centers, indexing="ij")]
            volumes = np.prod(np.stack(np.meshgrid(*widths, indexing="ij")), axis=0).reshape(-1)
            info = parameters(pdf, names)
            param_names = tuple(p.name for p in info)
            variables = [workspace.var(n) for n in param_names]
            initial = np.array([p.value for p in info])
            original_axes = [float(v.getVal()) for v in axes]
            points = [initial]
            for _ in range(2):
                points.append(np.array([np.clip(p.value+rng.uniform(-2, 2)*(p.error or 1.), p.minimum, p.maximum)
                                        for p in info]))
            result = dict(name=node.name, class_name=node.class_name, parameters=list(param_names),
                          shape=node.attrs["shape"], conditional=node.attrs["conditional"],
                          logarithmic=node.attrs["logarithmic"], cases=[])
            try:
                root_axes = root.RooArgList(*axes)
                for point in points:
                    for variable, value in zip(variables, point, strict=True): variable.setVal(float(value))
                    context = Evaluation(graph, registry, param_names, point, dict(zip(names, coordinates, strict=True)))
                    values = np.asarray(context.density(node.name, names))
                    ref_values = np.asarray(root.combine_jax_validation.sampleGrid(pdf, root_axes, [c.tolist() for c in centers]))
                    integral = float(context.integral(node.name, names))
                    reference_integral = float(ref_values @ volumes)
                    assert np.all(np.isfinite(values)) and np.all(np.isfinite(ref_values)), node.name
                    result["cases"].append(dict(theta=point.tolist(),
                        max_value_absolute_error=float(np.max(np.abs(values-ref_values))),
                        max_value_relative_error=float(np.max(np.abs(values-ref_values)/np.maximum(np.abs(ref_values), 1e-9))),
                        jax_integral=integral, combine_integral=reference_integral,
                        integral_absolute_error=abs(integral-reference_integral)))
            finally:
                for variable, value in zip(variables, initial, strict=True): variable.setVal(float(value))
                for variable, value in zip(axes, original_axes, strict=True): variable.setVal(value)
            report["nodes"].append(result)
            if (index+1) % 30 == 0:
                print(f"Compared {index+1}/{len(histograms)} multidimensional morphers", flush=True)
    cases = [case for node in report["nodes"] for case in node["cases"]]
    report.update(elapsed_seconds=time.perf_counter()-start, tested_nodes=len(report["nodes"]),
                  max_value_absolute_error=max(c["max_value_absolute_error"] for c in cases),
                  max_value_relative_error=max(c["max_value_relative_error"] for c in cases),
                  max_integral_absolute_error=max(c["integral_absolute_error"] for c in cases))
    Path("reports/hhvv_vertical_nd.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k not in ("nodes", "unsupported")}, flush=True)
    assert not report["unsupported"]
    assert report["tested_nodes"] == 370
    assert report["max_value_relative_error"] < 1e-10
    assert report["max_integral_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
