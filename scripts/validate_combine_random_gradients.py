"""Independent v11 FD references at the 25 saved random HHVV points.

No AD arrays are consulted. Uncertain, tiny or invalid finite differences remain
explicitly unresolved. Detailed probes are losslessly compressed per point.
"""

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extend-unresolved", action="store_true",
                        help="Preserve the original run and append smaller fixed steps only to unresolved references")
    arguments = parser.parse_args()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,[available[1]])
    started = time.perf_counter()
    parent_path = Path("reports/hhvv_random_values.json")
    parent = json.loads(parent_path.read_text())
    assert parent["status"] == "passed" and parent["valid_random_points"] == 25
    likelihood = load_export("build/gpu_exports/hhvv.pkl",parent["source"])
    np.testing.assert_array_equal(parent["parameter_names"],likelihood.parameter_names)
    np.testing.assert_array_equal(parent["initial"],likelihood.initial)
    fractions = (1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5,5e-6)
    previous = None
    if arguments.extend_unresolved:
        previous_path = Path("reports/gpu_combine_random_gradient.json")
        previous_bytes = previous_path.read_bytes()
        previous = json.loads(previous_bytes)
        assert previous["status"] == "reference_uncertain"
        assert previous["source"] == parent["source"] and len(previous["points"]) == 25
        assert previous["parent_reference_sha256"] == hashlib.sha256(parent_path.read_bytes()).hexdigest()
        fractions = (2e-6,1e-6,5e-7,2e-7,1e-7)
    report = dict(status="running",source=parent["source"],combine_version=version("higgs-combine-tool"),
                  root_version=parent["root_version"],runtime_flags=REFERENCE_FLAGS,
                  parameter_names=parent["parameter_names"],cpu_affinity=sorted(os.sched_getaffinity(0)),
                  parent_reference=str(parent_path),parent_reference_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fractions=list(fractions),
                  policy="For each nonfixed coordinate, start with three fixed steps and stop at the first three consecutive valid FD references whose two adjacent relative changes are <1e-4; no AD values used. Selected derivatives below 1e-4 in magnitude remain roundoff-sensitive, including identical rounded zero estimates. Invalid RooFit evaluations are recorded and smaller steps may be tried. AD must later agree with all three selected references.",
                  scope="Independent random-point gradient references only; no AD acceptance or isolated timing claim",
                  points=[])
    tag = "_extended" if previous else ""
    output = Path(f"reports/gpu_combine_random_gradient{tag}.json")
    if previous:
        report.update(extends_reference=str(previous_path),
                      extends_reference_sha256=hashlib.sha256(previous_bytes).hexdigest(),
                      fractions=previous["fractions"]+list(fractions),
                      extension_scope="Append the fixed smaller-step grid only to the five unresolved references; no AD arrays read. Tiny references remain uncertain. Original artifacts preserved.")
    if output.exists():
        raise FileExistsError("Preserve the existing random-gradient reference run")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        other = json.loads(Path("reports/hhvv_combine_fit.json").read_text())
        total = own+(other.get("peak_rss_mib",0.) if other.get("status") == "running" else 0.)
        if total > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed 11,000 MiB")

    root = load_caching_nll()
    assert str(root.gROOT.GetVersion()) == parent["root_version"]
    assert report["combine_version"] == parent["combine_version"]
    old_error_mode = root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == parent["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            model = workspace.obj("ModelConfig")
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(),workspace.data("data_obs"),model.GetNuisanceParameters())
            reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint(); reference.clearZeroPoint()

            def value():
                root.RooAbsReal.clearEvalErrorLog()
                result = float(reference.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                if not np.isfinite(result) or errors:
                    raise ValueError(f"Invalid Combine evaluation: value={result}, RooFit error count={errors}")
                return result

            for point_number,parent_case in enumerate(parent["cases"][1:],start=1):
                guard()
                theta = np.asarray(parent_case["theta"])
                previous_point = None
                if previous:
                    index = previous["points"][point_number-1]
                    blob = Path(index["artifact"]).read_bytes()
                    assert hashlib.sha256(blob).hexdigest() == index["artifact_sha256"]
                    raw = gzip.decompress(blob)
                    assert hashlib.sha256(raw).hexdigest() == index["raw_sha256"]
                    previous_point = json.loads(raw)
                    np.testing.assert_array_equal(previous_point["theta"],theta)
                    if not any(c["status"] == "unresolved" for c in previous_point["cases"]):
                        report["points"].append(index)
                        continue
                for variable,coordinate in zip(variables,theta,strict=True):
                    variable.setVal(float(coordinate))
                reference.clearZeroPoint()
                canonical = value()
                assert abs(canonical-parent_case["combine_nll"]) < 1e-6
                reference.setZeroPoint()
                zero = value()
                assert abs(zero) < 1e-6
                point = dict(point_number=point_number,attempt=parent_case["attempt"],theta=theta.tolist(),
                             canonical_nll=canonical,canonical_parent_absolute_error=abs(canonical-parent_case["combine_nll"]),
                             zero_point_nll=zero,cases=[])
                if previous_point:
                    point["cases"] = previous_point["cases"]
                report["current_point"] = point_number
                save()
                for index,(parameter,variable,coordinate) in enumerate(zip(likelihood.parameters,variables,theta,strict=True)):
                    if previous_point:
                        case = point["cases"][index]
                        assert case["name"] == parameter.name and case["coordinate"] == coordinate
                        if case["status"] != "unresolved":
                            continue
                    else:
                        case = dict(index=index,name=parameter.name,coordinate=float(coordinate),probes=[])
                    if parameter.minimum == parameter.maximum:
                        case["status"] = "fixed_bounds"
                    else:
                        scale = parameter.error if parameter.error > 0 else max(abs(parameter.value),1.)
                        left,right = coordinate-parameter.minimum,parameter.maximum-coordinate
                        for fraction in fractions:
                            step = fraction*scale
                            central = min(left,right) >= 2*step
                            direction = 1. if central or right >= left else -1.
                            if not central:
                                step = min(step,max(left,right)/5)
                            probe = dict(step=step,stencil="central" if central else "forward",direction=direction,
                                         zero_point_nll_values=[])
                            try:
                                assert step > 0 and coordinate+direction*step != coordinate
                                for multiple in ((-2,-1,1,2) if central else (1,2,3,4)):
                                    shifted = coordinate+direction*multiple*step
                                    assert parameter.minimum <= shifted <= parameter.maximum
                                    variable.setVal(float(shifted))
                                    probe["zero_point_nll_values"].append(value())
                                values = probe["zero_point_nll_values"]
                                derivative = ((8*(values[2]-values[1])-(values[3]-values[0]))/(12*step) if central else
                                              sum(c*(v-zero) for c,v in zip((48,-36,16,-3),values,strict=True))/(12*direction*step))
                                probe.update(status="valid",derivative=derivative)
                            except Exception as exc:
                                probe.update(status="invalid",error=str(exc))
                            finally:
                                variable.setVal(float(coordinate))
                                value()
                            case["probes"].append(probe)
                            triple = case["probes"][-3:]
                            if len(triple) == 3 and all(p["status"] == "valid" for p in triple):
                                changes = [abs(a["derivative"]-b["derivative"])/max(abs(b["derivative"]),1e-8)
                                           for a,b in zip(triple,triple[1:])]
                                if max(changes) < 1e-4:
                                    case.update(status="converged" if min(abs(p["derivative"]) for p in triple) >= 1e-4 else "roundoff_sensitive",
                                                selected_first_probe_index=len(case["probes"])-3,
                                                adjacent_relative_changes=changes)
                                    break
                        case.setdefault("status","unresolved")
                    if not previous_point:
                        point["cases"].append(case)
                    if (index+1) % 200 == 0:
                        guard()
                        print("point",point_number,"coordinate",index+1,"elapsed",time.perf_counter()-started,flush=True)
                reference.clearZeroPoint()
                point["canonical_restoration_absolute_error"] = abs(value()-canonical)
                assert point["canonical_restoration_absolute_error"] < 1e-6
                statuses = sorted({c["status"] for c in point["cases"]})
                point["status_counts"] = {s:sum(c["status"] == s for c in point["cases"]) for s in statuses}
                point["unresolved_names"] = [c["name"] for c in point["cases"] if c["status"] not in ("fixed_bounds","converged")]
                raw = (json.dumps(point,indent=2,allow_nan=False)+"\n").encode()
                artifact = Path("reports")/f"gpu_combine_random_gradient{tag}_{point_number:02d}.json.gz"
                if artifact.exists():
                    raise FileExistsError(str(artifact))
                artifact.write_bytes(gzip.compress(raw,mtime=0))
                report["points"].append(dict(point_number=point_number,attempt=parent_case["attempt"],
                    artifact=str(artifact),artifact_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    raw_sha256=hashlib.sha256(raw).hexdigest(),status_counts=point["status_counts"],
                    unresolved_names=point["unresolved_names"],canonical_restoration_absolute_error=point["canonical_restoration_absolute_error"]))
                save()
                print("point complete",point_number,point["status_counts"],flush=True)
            assert len(report["points"]) == 25
            report["status"] = "reference_ready" if all(not p["unresolved_names"] for p in report["points"]) else "reference_uncertain"
            del reference
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_error_mode)


if __name__ == "__main__":
    main()
