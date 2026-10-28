"""Independent, offset-subtracted Combine curvature probes for the nominal Hessian.

The directions are frozen by the earlier gradient reference. Step selection uses
only reference convergence, before any Hessian arrays are read.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarser", action="store_true",
                        help="Prepend fixed larger steps to the preserved original reference grid")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    parent_path = Path("reports/gpu_combine_directions.json")
    parent = json.loads(parent_path.read_text())
    assert parent["status"] == "reference_ready"
    likelihood = load_export("build/gpu_exports/hhvv.pkl", parent["source"])
    np.testing.assert_array_equal(likelihood.initial, parent["initial"])
    np.testing.assert_array_equal(likelihood.parameter_names, parent["parameter_names"])
    initial = likelihood.initial
    low = np.array([p.minimum for p in likelihood.parameters])
    high = np.array([p.maximum for p in likelihood.parameters])
    steps = (1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6)
    previous = None
    if arguments.coarser:
        previous_path = Path("reports/gpu_combine_curvature.json")
        previous_bytes = previous_path.read_bytes()
        previous = json.loads(previous_bytes)
        assert previous["status"] in ("reference_ready","reference_uncertain")
        assert previous["source"] == parent["source"]
        assert previous["parent_reference_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
        assert previous["steps"] == list(steps)
        steps = (1e-2,5e-3,2e-3)
    report = dict(status="running", source=parent["source"],
                  combine_version=parent["combine_version"], root_version=parent["root_version"],
                  runtime_flags=REFERENCE_FLAGS, parameter_names=parent["parameter_names"],
                  initial=parent["initial"], steps=list(steps),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  parent_reference=str(parent_path),
                  parent_reference_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  policy="All 18 frozen directions; five-point central curvature with channel zero points at nominal; first three consecutive steps with both adjacent relative changes <1e-4 are selected without consulting AD. AD must later agree with all three. Unconverged directions remain unresolved.",
                  scope="Independent nominal directional Hessian reference; no all-entry Hessian, random derivative, fit or performance acceptance",
                  cases=[])
    suffix = "_coarser" if arguments.coarser else ""
    output = Path(f"reports/gpu_combine_curvature{suffix}.json")
    if previous:
        report.update(extends_reference=str(previous_path),
                      extends_reference_sha256=hashlib.sha256(previous_bytes).hexdigest(),
                      steps=list(steps)+previous["steps"],
                      extension_scope="Three fixed larger steps for every original direction, followed by the unchanged original probes; first converged triple selected without AD. New probes explicitly check RooFit evaluation-error counts.")
    if output.exists():
        raise FileExistsError("Do not overwrite the original curvature reference")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        fit_path = Path("reports/hhvv_combine_fit.json")
        fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
        other = fit.get("peak_rss_mib", 0.) if fit.get("status") == "running" else 0.
        if own+other > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed the 11,000 MiB guard")

    root = None
    old_error_mode = None
    try:
        root = load_caching_nll()
        assert str(root.gROOT.GetVersion()) == parent["root_version"]
        assert version("higgs-combine-tool") == parent["combine_version"]
        old_error_mode = root.RooAbsReal.evalErrorLoggingMode()
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == parent["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
            reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint(); reference.clearZeroPoint()

            def value(theta):
                guard()
                if np.any(theta < low) or np.any(theta > high):
                    raise ValueError("Curvature probe leaves the allowed range")
                for variable, coordinate in zip(variables, theta, strict=True):
                    variable.setVal(float(coordinate))
                root.RooAbsReal.clearEvalErrorLog()
                result = float(reference.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                if not np.isfinite(result) or errors:
                    raise ValueError(f"Invalid curvature reference NLL: value={result}, RooFit error count={errors}")
                return result

            nominal = value(initial)
            report["canonical_nominal_nll"] = nominal
            assert abs(nominal-parent["combine_nll"]) < 1e-6
            reference.setZeroPoint()
            zero = value(initial)
            report["nominal_zero_point_nll"] = zero
            assert abs(zero) < 1e-6
            save()
            for index,direction in enumerate(parent["cases"]):
                assert direction["stencil"] == "central"
                vector = np.asarray(direction["direction"])
                case = dict(label=direction["label"], direction=vector.tolist(), probes=[])
                report["cases"].append(case)
                for step in steps:
                    values = [value(initial+t*step*vector) for t in (-2,-1,1,2)]
                    deltas = np.array(values,dtype=np.longdouble)-np.longdouble(zero)
                    curvature = np.dot(deltas,np.array([-1,16,16,-1]))/(12*np.longdouble(step)**2)
                    case["probes"].append(dict(step=step, zero_point_nll_values=values, curvature=float(curvature),
                                               roofit_evaluation_errors=0))
                    save()
                if previous:
                    old_case = previous["cases"][index]
                    assert old_case["label"] == case["label"] and old_case["direction"] == case["direction"]
                    case["probes"].extend(old_case["probes"])
                probes = case["probes"]
                triples = []
                for first in range(len(probes)-2):
                    triple = probes[first:first+3]
                    changes = [abs(a["curvature"]-b["curvature"])/max(abs(b["curvature"]),1e-8)
                               for a,b in zip(triple,triple[1:])]
                    if max(changes) < 1e-4:
                        triples.append(dict(first_probe_index=first, steps=[p["step"] for p in triple],
                                            adjacent_relative_changes=changes))
                case["reference_status"] = "converged" if triples else "unresolved"
                case["selected_triple"] = triples[0] if triples else None
                print(case["label"], case["reference_status"], [p["curvature"] for p in probes], flush=True)
                save()
            value(initial)
            reference.clearZeroPoint()
            report["canonical_restoration_absolute_error"] = abs(value(initial)-nominal)
            assert report["canonical_restoration_absolute_error"] < 1e-6
            del reference
        report["unresolved_directions"] = [c["label"] for c in report["cases"] if c["reference_status"] != "converged"]
        report["status"] = "reference_ready" if not report["unresolved_directions"] else "reference_uncertain"
        save()
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        if root is not None and old_error_mode is not None:
            root.RooAbsReal.setEvalErrorLoggingMode(old_error_mode)


if __name__ == "__main__":
    main()
