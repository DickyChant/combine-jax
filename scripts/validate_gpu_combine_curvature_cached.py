"""Freeze all 18 Combine curvature references using validated cached-density differences.

Directions, steps, channel coverage and convergence rule are fixed before running.
No JAX derivative or Hessian arrays are read. Original failed references survive.
"""

import argparse
import copy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from cached_root_intensities import CachedRootIntensities
from caching_nll_diagnostics import channel_nlls
from combine_jax.discrete import active_dependencies
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from compare_gpu_curvature import first_converged_triple
from compare_gpu_cached_curvature import FINER_STEPS, audit_reference
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from probe_gpu_curvature_channels import curvature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--finer-grid", action="store_true",
                        help="Preserve the original 12 steps, repeat their final anchor and append three fixed finer steps for every direction")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    base_path = Path("reports/gpu_combine_curvature_cached.json")
    output = base_path.with_name("gpu_combine_curvature_cached_finer.json") if arguments.finer_grid else base_path
    if output.exists():
        raise FileExistsError("Preserve existing curvature references")
    base = audit_reference(base_path)[0] if arguments.finer_grid else None
    original_path = Path("reports/gpu_combine_curvature_coarser.json")
    original = json.loads(original_path.read_text())
    pilot_path = Path("reports/gpu_curvature_cached_all.json")
    pilot = json.loads(pilot_path.read_text())
    audit_path = Path("reports/gpu_curvature_cached_all_audit.json")
    audit = json.loads(audit_path.read_text())
    assert audit["status"] == "passed" and audit["all_channels_accounted_for"]
    assert audit["report_sha256"] == hashlib.sha256(pilot_path.read_bytes()).hexdigest()
    assert pilot["reference_sha256"] == hashlib.sha256(original_path.read_bytes()).hexdigest()
    helper = Path("scripts/cached_root_intensities.py")
    assert pilot["runner_sha256"][str(helper)] == hashlib.sha256(helper.read_bytes()).hexdigest()
    assert original["status"] == "reference_uncertain" and len(original["cases"]) == 18
    likelihood = load_export("build/gpu_exports/hhvv.pkl", original["source"])
    initial = np.asarray(original["initial"])
    np.testing.assert_array_equal(likelihood.initial, initial)
    np.testing.assert_array_equal(likelihood.parameter_names, original["parameter_names"])
    low = np.array([p.minimum for p in likelihood.parameters])
    high = np.array([p.maximum for p in likelihood.parameters])
    report = dict(status="running", source=original["source"],
        combine_version=version("higgs-combine-tool"), root_version=original["root_version"],
        runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
        parent_reference=str(original_path), parent_reference_sha256=hashlib.sha256(original_path.read_bytes()).hexdigest(),
        pilot=str(pilot_path), pilot_sha256=hashlib.sha256(pilot_path.read_bytes()).hexdigest(),
        pilot_audit=str(audit_path), pilot_audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),helper,Path("scripts/caching_nll_diagnostics.py"),
                       Path("scripts/stable_root_intensities.py"),Path("scripts/probe_hhvv_stable_fd.py"),
                       Path("scripts/compare_gpu_curvature.py"),Path("scripts/probe_gpu_curvature_channels.py"),
                       Path("scripts/compare_gpu_cached_curvature.py"))},
        parameter_names=original["parameter_names"], initial=initial.tolist(), steps=original["steps"],
        policy="All 18 frozen directions and all 12 original steps. Stable long-double NLL differences use actual live cached densities/coefficients in the 274 validated RooAddPdf channels. Every excluded channel and nonchannel residual stays in the full NLL unchanged. Select the first three consecutive steps whose adjacent relative curvature changes are both <1e-4, denominator max(abs(next curvature),1e-8), without consulting AD. Future AD comparisons must pass all three. Unconverged cases remain unresolved.",
        scope="Independent nominal directional curvature references; no full Hessian, GPU, fit or isolated timing acceptance",
        excluded_cached_channels=pilot["excluded_cached_channels"], cases=[])
    if base is not None:
        assert base["parent_reference_sha256"] == report["parent_reference_sha256"]
        report.update(extends_cached_reference=str(base_path),
            extends_cached_reference_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
            steps=base["steps"]+FINER_STEPS,
            policy=base["policy"]+" Fixed extension: retain every original sample, exactly repeat the final 2e-6 anchor in each of all 18 directions, and append steps 1e-6, 5e-7, 2e-7. Apply the unchanged first-converged-triple rule to the whole sequence. No AD arrays are read.")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        fit_path = Path("reports/hhvv_combine_fit.json")
        fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
        if own+(fit.get("peak_rss_mib",0.) if fit.get("status") == "running" else 0.) > 11000:
            raise MemoryError("Combined local reference jobs exceed 11,000 MiB")
        if time.perf_counter()-started > 2400:
            raise TimeoutError("Cached curvature reference exceeds 2,400 s")

    root = load_caching_nll()
    assert str(root.gROOT.GetVersion()) == report["root_version"]
    assert report["combine_version"] == original["combine_version"]
    old_mode = root.RooAbsReal.evalErrorLoggingMode()
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
            parts, layout = channel_nlls(root,nll)
            assert len(parts) == len(likelihood.channels) == 459
            group, excluded = [], []
            for index,(part,channel) in enumerate(zip(parts,likelihood.channels,strict=True)):
                assert str(part.GetName()) == channel.name
                dependencies = active_dependencies(likelihood.nodes,[channel.pdf])
                reason = ("masked" if channel.masked else
                    "non-RooAddPdf" if str(part.pdf().ClassName()) != "RooAddPdf" else
                    "MultiPdf" if any(likelihood.nodes[n].class_name == "RooMultiPdf" for n in dependencies) else
                    "analytic-bin profile" if any(p.propagator in dependencies for p in likelihood.profiles) else None)
                if reason:
                    excluded.append(dict(index=index,channel=channel.name,reason=reason))
                else:
                    group.append((index,CachedRootIntensities(root,part)))
            assert excluded == pilot["excluded_cached_channels"]
            assert [i for i,_ in group] == [c["index"] for c in pilot["stable_cached_channels"]]
            report["reflected_layout"] = layout
            baselines = None

            def sample(theta):
                guard()
                assert np.all(theta >= low) and np.all(theta <= high)
                for variable,coordinate in zip(variables,theta,strict=True):
                    variable.setVal(float(coordinate))
                root.RooAbsReal.clearEvalErrorLog()
                full = float(nll.getVal())
                channels = np.array([p.getVal() for p in parts],dtype=np.longdouble)
                assert np.isfinite(full) and np.isfinite(channels).all()
                result = dict(full_nll=full)
                if baselines is not None:
                    stable = np.array([p.delta(b) for (_,p),b in zip(group,baselines,strict=True)],dtype=np.longdouble)
                    cached = np.array([channels[i]-zero_channels[i] for i,_ in group])
                    maximum = float(np.max(abs(stable-cached),initial=0.))
                    assert maximum < 1e-6
                    result.update(stable_delta=float(np.longdouble(full)-np.longdouble(zero["full_nll"])+np.sum(stable-cached,dtype=np.longdouble)),
                                  stable_channel_delta_max_absolute_error=maximum)
                assert float(nll.getVal()) == full and int(root.RooAbsReal.numEvalErrors()) == 0
                result["roofit_evaluation_errors"] = 0
                return result,channels

            canonical,_ = sample(initial)
            assert abs(canonical["full_nll"]-original["canonical_nominal_nll"]) < 1e-6
            nll.setZeroPoint()
            zero,zero_channels = sample(initial)
            assert abs(zero["full_nll"]) < 1e-6
            baselines = [p.predict() for _,p in group]
            channels = [dict(index=i,channel=likelihood.channels[i].name,**p.report) for i,p in group]
            assert channels == pilot["stable_cached_channels"]
            report.update(canonical_nominal_nll=canonical["full_nll"],nominal_zero_point_nll=zero["full_nll"],
                          stable_cached_channels=channels)
            save()
            for case_index, original_case in enumerate(original["cases"]):
                direction = np.asarray(original_case["direction"])
                case = dict(label=original_case["label"],direction=direction.tolist(),probes=[])
                report["cases"].append(case)
                old_probes = original_case["probes"]
                if base is not None:
                    prior_case = base["cases"][case_index]
                    assert prior_case["label"] == case["label"] and prior_case["direction"] == case["direction"]
                    case.update(copy.deepcopy(prior_case))
                    step = base["steps"][-1]
                    samples = [dict(multiple=k,**sample(initial+k*step*direction)[0]) for k in (-2,-1,1,2)]
                    case["repeated_anchor"] = dict(step=step,samples=samples,
                        curvature=float(curvature([s["stable_delta"] for s in samples],0.,step)))
                    for current, previous in zip(samples,prior_case["probes"][-1]["samples"],strict=True):
                        for key in ("multiple","full_nll","stable_delta"):
                            assert current[key] == previous[key], "Frozen anchor changed"
                    old_probes = [dict(step=step) for step in FINER_STEPS]
                for old_probe in old_probes:
                    step = old_probe["step"]
                    samples = [dict(multiple=k,**sample(initial+k*step*direction)[0]) for k in (-2,-1,1,2)]
                    probe = dict(step=step,samples=samples,
                        curvature=float(curvature([s["stable_delta"] for s in samples],0.,step)))
                    if base is None:
                        cached_values = [s["full_nll"] for s in samples]
                        error = float(np.max(abs(np.array(cached_values)-old_probe["zero_point_nll_values"])))
                        assert error < 1e-6
                        probe.update(original_curvature=old_probe["curvature"],original_sample_max_absolute_error=error)
                    case["probes"].append(probe)
                    save()
                case["selected_triple"] = first_converged_triple(case["probes"])
                case["reference_status"] = "converged" if case["selected_triple"] else "unresolved"
                if base is None and case["label"] == pilot["label"]:
                    case["pilot_curvature_max_absolute_error"] = max(abs(a["curvature"]-b["stable_full_curvature"])
                        for a,b in zip(case["probes"],pilot["probes"],strict=True))
                print(case["label"],case["reference_status"],case["selected_triple"],flush=True)
                save()
            baselines = None
            sample(initial)
            nll.clearZeroPoint()
            report["canonical_restoration_absolute_error"] = abs(sample(initial)[0]["full_nll"]-canonical["full_nll"])
            assert report["canonical_restoration_absolute_error"] < 1e-6
            group.clear(); parts.clear()
            del nll
        report["unresolved_directions"] = [c["label"] for c in report["cases"] if c["selected_triple"] is None]
        report["status"] = "reference_uncertain" if report["unresolved_directions"] else "reference_ready"
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
