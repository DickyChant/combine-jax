"""Stable ROOT-bin FD references for the 24 frozen small random derivatives."""

import argparse
from collections import Counter
import gzip
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
import resource
import time
import traceback

import numpy as np

from combine_jax.discrete import active_dependencies
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from stable_root_bins import RootBins, constraint_delta, poisson_delta, LD


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-tag", default="")
    arguments = parser.parse_args()
    assert re.fullmatch(r"[a-z0-9_]*",arguments.report_tag)
    suffix = "_"+arguments.report_tag if arguments.report_tag else ""
    output = Path(f"reports/gpu_random_stable_fd{suffix}.json")
    if output.exists():
        raise FileExistsError(str(output))
    os.sched_setaffinity(0,[sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    reference_path = Path("reports/gpu_combine_random_gradient_extended.json")
    reference_bytes = reference_path.read_bytes()
    original = json.loads(reference_bytes)
    likelihood = load_export("build/gpu_exports/hhvv.pkl",original["source"])
    np.testing.assert_array_equal(likelihood.parameter_names,original["parameter_names"])
    dependencies = {c.name:active_dependencies(likelihood.nodes,[c.pdf]) for c in likelihood.channels}
    root = load_caching_nll()
    assert str(root.gROOT.GetVersion()) == original["root_version"]
    assert version("higgs-combine-tool") == original["combine_version"]
    assert np.finfo(LD).nmant >= 63
    report = dict(status="running",source=original["source"],reference=str(reference_path),
                  reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
                  root_version=str(root.gROOT.GetVersion()),combine_version=version("higgs-combine-tool"),
                  runtime_flags=REFERENCE_FLAGS,cpu_affinity=sorted(os.sched_getaffinity(0)),
                  parameter_names=original["parameter_names"],longdouble_mantissa_bits=np.finfo(LD).nmant,
                  script_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                 (Path(__file__),Path("scripts/stable_root_bins.py"),Path("scripts/probe_hhvv_stable_fd.py"))},
                  selection="Exactly the 24 roundoff-sensitive coordinates in the frozen extended reference; all three original steps retained; no AD arrays consulted",
                  method="ROOT component bin expectations, long-double summation and stable Poisson differences; exact Gaussian/Poisson constraint differences; five-point FD",
                  policy="All three stable FD references must converge with adjacent relative changes below 1e-4; AD must later pass all three. Uncertain cases remain explicit.",
                  scope="Reference generation only, on verified flat-bin unprofiled channels; no AD acceptance or isolated performance claim",
                  points=[])

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        fit = json.loads(Path("reports/hhvv_combine_fit.json").read_text())
        total = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if fit["status"] == "running":
            total += fit["peak_rss_mib"]
        if total > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed 11,000 MiB")

    old_error_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == original["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(),workspace.data("data_obs"),model.GetNuisanceParameters())
            nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()

            def value():
                root.RooAbsReal.clearEvalErrorLog()
                result = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                assert np.isfinite(result) and errors == 0, (result,errors)
                return result

            for index in original["points"]:
                guard()
                blob = Path(index["artifact"]).read_bytes()
                assert hashlib.sha256(blob).hexdigest() == index["artifact_sha256"]
                raw = gzip.decompress(blob)
                assert hashlib.sha256(raw).hexdigest() == index["raw_sha256"]
                old = json.loads(raw)
                selected = [c for c in old["cases"] if c["status"] == "roundoff_sensitive"]
                if not selected:
                    continue
                for variable,coordinate in zip(variables,old["theta"],strict=True):
                    variable.setVal(float(coordinate))
                nll.clearZeroPoint()
                canonical = value()
                assert abs(canonical-old["canonical_nll"]) < 1e-6
                nll.setZeroPoint()
                zero = value()
                assert abs(zero) < 1e-6
                point = dict(point_number=old["point_number"],attempt=old["attempt"],theta=old["theta"],
                             canonical_nll=canonical,canonical_parent_absolute_error=abs(canonical-old["canonical_nll"]),
                             cases=[],channels=[])
                report["points"].append(point)
                selected_names = {c["name"] for c in selected}
                channels = [c for c in likelihood.channels if selected_names & dependencies[c.name]]
                predictors = {}
                for channel in channels:
                    prediction = RootBins(root,workspace,likelihood,channel)
                    prediction.check_flatness()
                    baseline = prediction.predict()
                    point["channels"].append(prediction.report)
                    predictors[channel.name] = (prediction,baseline)
                for case in selected:
                    variable = variables[case["index"]]
                    coordinate = variable.getVal()
                    affected = [c.name for c in channels if case["name"] in dependencies[c.name]]
                    result = dict(name=case["name"],index=case["index"],coordinate=coordinate,
                                  channels=affected,probes=[])
                    point["cases"].append(result)
                    assert len(case["probes"]) == 3
                    for old_probe in case["probes"]:
                        step = old_probe["step"]
                        assert old_probe["stencil"] == "central" and old_probe["status"] == "valid"
                        values, checks = [], []
                        for multiple in (-2,-1,1,2):
                            shifted = coordinate+multiple*step
                            variable.setVal(shifted)
                            actual = value()-zero
                            stable = constraint_delta(likelihood,workspace,case["name"],coordinate,shifted)
                            for name in affected:
                                prediction,baseline = predictors[name]
                                stable += poisson_delta(prediction.predict(),baseline,prediction.weights)
                            errors = int(root.RooAbsReal.numEvalErrors())
                            assert errors == 0 and np.isfinite(stable)
                            absolute_error = abs(float(stable)-actual)
                            checks.append(dict(coordinate=shifted,combine_delta=actual,stable_delta=float(stable),
                                               absolute_error=absolute_error))
                            assert absolute_error < 1e-6, (case["name"],checks[-1])
                            values.append(stable)
                        derivative = float((8*(values[2]-values[1])-(values[3]-values[0]))/(12*LD(step)))
                        result["probes"].append(dict(step=step,derivative=derivative,
                                                    original_full_nll_fd=old_probe["derivative"],delta_checks=checks))
                        variable.setVal(coordinate)
                        assert abs(value()-zero) < 1e-6
                    derivatives = [p["derivative"] for p in result["probes"]]
                    changes = [abs(a-b)/max(abs(b),1e-8) for a,b in zip(derivatives,derivatives[1:])]
                    result.update(adjacent_relative_changes=changes,
                                  status="reference_ready" if max(changes) < 1e-4 else "reference_uncertain")
                    print(point["point_number"],case["name"],result["status"],derivatives,flush=True)
                    guard(); save()
                nll.clearZeroPoint()
                point["canonical_restoration_absolute_error"] = abs(value()-canonical)
                assert point["canonical_restoration_absolute_error"] < 1e-6
            del nll
        cases = [c for p in report["points"] for c in p["cases"]]
        assert len(cases) == 24
        report["status_counts"] = dict(Counter(c["status"] for c in cases))
        report["status"] = "reference_ready" if all(c["status"] == "reference_ready" for c in cases) else "reference_uncertain"
        report["max_combine_delta_absolute_error"] = max(d["absolute_error"] for c in cases for p in c["probes"] for d in p["delta_checks"])
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_error_mode)


if __name__ == "__main__":
    main()
