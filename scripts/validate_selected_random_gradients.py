"""Fixed-grid Combine diagnostics for explicitly selected random-gradient failures.

Selection is made after the original GPU comparison. This generator receives
only case identities and fixed probe rules, never AD derivative values or arrays.
It preserves the original reference and comparison reports.
"""

from collections import Counter
import gzip
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
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    selection_path = Path("reports/gpu_random_gradient_failure_selection.json")
    raw = selection_path.read_bytes()
    selection = json.loads(raw)
    reference_path = Path(selection["reference"])
    reference_raw = reference_path.read_bytes()
    assert hashlib.sha256(reference_raw).hexdigest() == selection["reference_sha256"]
    reference = json.loads(reference_raw)
    assert reference["source"] == selection["source"]
    likelihood = load_export("build/gpu_exports/hhvv.pkl", reference["source"])
    np.testing.assert_array_equal(likelihood.parameter_names, reference["parameter_names"])
    output = Path("reports/gpu_random_gradient_failure_fd.json")
    if output.exists():
        raise FileExistsError(str(output))
    report = dict(status="running", source=reference["source"],
                  selection=str(selection_path), selection_sha256=hashlib.sha256(raw).hexdigest(),
                  reference=str(reference_path), reference_sha256=selection["reference_sha256"],
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  combine_version=version("higgs-combine-tool"), root_version=reference["root_version"],
                  runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
                  fractions=selection["fractions"], policy=selection["policy"],
                  scope=selection["scope"]+" Reference diagnostics only; no AD acceptance.", points=[])

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        fit = json.loads(Path("reports/hhvv_combine_fit.json").read_text())
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if fit["status"] == "running":
            peak += fit["peak_rss_mib"]
        if peak > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed 11,000 MiB")

    root = load_caching_nll()
    assert str(root.gROOT.GetVersion()) == reference["root_version"]
    assert report["combine_version"] == reference["combine_version"]
    old_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == reference["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
            nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()

            def value():
                guard()
                root.RooAbsReal.clearEvalErrorLog()
                result = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                if not np.isfinite(result) or errors:
                    raise ValueError(f"Invalid Combine NLL: value={result}, RooFit error count={errors}")
                return result

            for index in reference["points"]:
                selected = [c for c in selection["cases"] if c["point_number"] == index["point_number"]]
                if not selected:
                    continue
                blob = Path(index["artifact"]).read_bytes()
                assert hashlib.sha256(blob).hexdigest() == index["artifact_sha256"]
                raw = gzip.decompress(blob)
                assert hashlib.sha256(raw).hexdigest() == index["raw_sha256"]
                old = json.loads(raw)
                for variable, coordinate in zip(variables, old["theta"], strict=True):
                    variable.setVal(float(coordinate))
                nll.clearZeroPoint()
                canonical = value()
                assert abs(canonical-old["canonical_nll"]) < 1e-6
                nll.setZeroPoint()
                zero = value()
                assert abs(zero) < 1e-6
                point = dict(point_number=old["point_number"], attempt=old["attempt"], theta=old["theta"],
                             canonical_nll=canonical, canonical_parent_absolute_error=abs(canonical-old["canonical_nll"]), cases=[])
                report["points"].append(point)
                for selected_case in selected:
                    i = selected_case["index"]
                    parameter, variable, previous = likelihood.parameters[i], variables[i], old["cases"][i]
                    assert parameter.name == previous["name"] == selected_case["name"]
                    assert selected_case["attempt"] == old["attempt"]
                    coordinate = old["theta"][i]
                    scale = parameter.error if parameter.error > 0 else max(abs(parameter.value), 1.)
                    case = dict(name=parameter.name, index=i, coordinate=coordinate, scale=scale,
                                original_reference=previous, probes=[])
                    point["cases"].append(case)
                    left, right = coordinate-parameter.minimum, parameter.maximum-coordinate
                    for fraction in selection["fractions"]:
                        step = fraction*scale
                        central = min(left, right) >= 2*step
                        direction = 1. if central or right >= left else -1.
                        if not central:
                            step = min(step, max(left, right)/5)
                        probe = dict(fraction=fraction, step=step, stencil="central" if central else "forward",
                                     direction=direction, zero_point_nll_values=[])
                        try:
                            assert step > 0 and coordinate+direction*step != coordinate
                            for multiple in ((-2,-1,1,2) if central else (1,2,3,4)):
                                shifted = coordinate+direction*multiple*step
                                assert parameter.minimum <= shifted <= parameter.maximum
                                variable.setVal(float(shifted))
                                probe["zero_point_nll_values"].append(value())
                            values = np.array(probe["zero_point_nll_values"], dtype=np.longdouble)
                            derivative = ((8*(values[2]-values[1])-(values[3]-values[0]))/(12*np.longdouble(step)) if central else
                                          np.dot([48,-36,16,-3], values-np.longdouble(zero))/(12*direction*np.longdouble(step)))
                            probe.update(status="valid", derivative=float(derivative), roofit_evaluation_errors=0)
                        except Exception as exc:
                            probe.update(status="invalid", error=str(exc))
                        finally:
                            variable.setVal(float(coordinate))
                            assert abs(value()-zero) < 1e-6
                        case["probes"].append(probe)
                    case.update(status="reference_uncertain", selected_first_probe_index=None)
                    for first in range(len(case["probes"])-2):
                        triple = case["probes"][first:first+3]
                        if not all(p["status"] == "valid" for p in triple):
                            continue
                        values = [p["derivative"] for p in triple]
                        spread = (max(values)-min(values))/max(min(abs(v) for v in values), 1e-8)
                        if spread < 1e-5:
                            case.update(status="reference_ready", selected_first_probe_index=first,
                                        selected_relative_spread=spread)
                            break
                    print(point["point_number"], case["name"], case["status"],
                          [(p["step"], p.get("derivative")) for p in case["probes"]], flush=True)
                    save()
                nll.clearZeroPoint()
                point["canonical_restoration_absolute_error"] = abs(value()-canonical)
                assert point["canonical_restoration_absolute_error"] < 1e-6
            del nll
        cases = [c for p in report["points"] for c in p["cases"]]
        assert len(cases) == len(selection["cases"])
        report["status_counts"] = dict(Counter(c["status"] for c in cases))
        report["status"] = "reference_ready" if all(c["status"] == "reference_ready" for c in cases) else "reference_uncertain"
        save()
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
