"""Check all HHVV splines at configured mass and at argument-level probe points."""

from importlib.metadata import version
import json
from pathlib import Path
import time

import jax
import numpy as np

from hhvv_setup import configure_hhvv
from combine_jax.graph import walk
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_root, open_workspace, stored_member
from combine_jax.translators import default_registry


def main():
    start = time.perf_counter()
    root, registry = load_root(), default_registry()
    rng = np.random.default_rng(8194)
    report = dict(source="inputs/workspace_MX2000.root", seed=8194,
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  scope="Nominal configured values plus argument-level spline probes; no full NLL claim",
                  derivative_reference="ROOT::Math::Interpolator Deriv/Deriv2 with the same stored table and algorithm",
                  nodes=[])
    with open_workspace(report["source"]) as workspace:
        report["configuration"] = configure_hhvv(workspace)
        splines = [n for n in walk(workspace.pdf("model_s")).values() if n.ClassName() == "RooSpline1D"]
        for index, spline in enumerate(splines):
            graph = registry.export([spline])
            node = graph[str(spline.GetName())]
            argument = tuple(spline.servers())[0]
            if argument.ClassName() != "RooRealVar":
                raise ValueError(f"{node.name}: class-probe argument must be directly assignable")
            original = float(argument.getVal())
            knots = node.attrs["knots"]
            lo, hi = max(knots[0], argument.getMin()), min(knots[-1], argument.getMax())
            points = [original, *rng.uniform(lo, hi, 2)]
            interpolator = root.Math.Interpolator(knots.tolist(), stored_member(spline, "y_"),
                                                 getattr(root.Math.Interpolation, "k"+node.attrs["algorithm"]))
            def value(x):
                return Evaluation(graph, registry, node.inputs, [x]).value(node.name)
            derivative, second = jax.grad(value), jax.hessian(value)
            result = dict(name=node.name, argument=node.inputs[0], algorithm=node.attrs["algorithm"],
                          knots=len(knots), cases=[])
            try:
                for coordinate in points:
                    argument.setVal(float(coordinate))
                    actual, reference = float(value(coordinate)), float(spline.getVal())
                    grad, ref_grad = float(derivative(coordinate)), float(interpolator.Deriv(float(coordinate)))
                    hess, ref_hess = float(second(coordinate)), float(interpolator.Deriv2(float(coordinate)))
                    result["cases"].append(dict(argument=float(coordinate), jax=actual, combine=reference,
                        absolute_error=abs(actual-reference), jax_derivative=grad, reference_derivative=ref_grad,
                        derivative_absolute_error=abs(grad-ref_grad), derivative_relative_error=abs(grad-ref_grad)/max(abs(ref_grad), 1e-8),
                        jax_second_derivative=hess, reference_second_derivative=ref_hess,
                        second_derivative_absolute_error=abs(hess-ref_hess)))
            finally:
                argument.setVal(original)
            report["nodes"].append(result)
            if (index+1) % 50 == 0:
                print(f"Compared {index+1}/{len(splines)} splines", flush=True)
    cases = [c for n in report["nodes"] for c in n["cases"]]
    report.update(tested_nodes=len(report["nodes"]), elapsed_seconds=time.perf_counter()-start,
                  max_value_absolute_error=max(c["absolute_error"] for c in cases),
                  max_derivative_absolute_error=max(c["derivative_absolute_error"] for c in cases),
                  max_derivative_relative_error=max(c["derivative_relative_error"] for c in cases),
                  max_second_derivative_absolute_error=max(c["second_derivative_absolute_error"] for c in cases))
    Path("reports/hhvv_splines.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print({k: v for k, v in report.items() if k != "nodes"}, flush=True)
    assert report["tested_nodes"] == 284
    assert report["max_value_absolute_error"] < 1e-8
    assert report["max_derivative_relative_error"] < 1e-6
    assert report["max_second_derivative_absolute_error"] < 1e-8


if __name__ == "__main__":
    main()
