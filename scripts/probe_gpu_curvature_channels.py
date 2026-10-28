"""Locate step-dependent curvature in the actual Combine channel NLLs.

Select the unresolved direction with the largest relative curvature range using
only saved Combine references. No AD gradient or Hessian is read.
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

from caching_nll_diagnostics import channel_nlls
from cached_root_intensities import CachedRootIntensities
from combine_jax.discrete import active_dependencies
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from stable_root_intensities import RootIntensities


def curvature(values, zero, step):
    deltas = np.asarray(values,dtype=np.longdouble)-np.asarray(zero,dtype=np.longdouble)
    return np.tensordot(np.array([-1,16,16,-1],dtype=np.longdouble),deltas,axes=1)/(12*np.longdouble(step)**2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--stable-leading",action="store_true",
                        help="Repeat the unchanged grid with stable ROOT-density differences in the previously leading channel")
    modes.add_argument("--stable-cached-small",action="store_true",
                        help="Use live cached components in the channel with the largest range over the final three steps")
    modes.add_argument("--stable-cached-all",action="store_true",
                        help="Use stable cached-density differences in every supported channel; retain other contributions explicitly")
    parser.add_argument("--output",type=Path,help="Separate report path; existing evidence cannot be overwritten")
    args = parser.parse_args()
    os.sched_setaffinity(0, [sorted(os.sched_getaffinity(0))[1]])
    started = time.perf_counter()
    reference_path = Path("reports/gpu_combine_curvature_coarser.json")
    raw = reference_path.read_bytes()
    reference = json.loads(raw)
    assert reference["status"] == "reference_uncertain"
    candidates = [c for c in reference["cases"] if c["selected_triple"] is None]
    def score(case):
        values = [p["curvature"] for p in case["probes"]]
        return (max(values)-min(values))/max(abs(float(np.median(values))),1e-8)
    selected = max(candidates,key=score)
    likelihood = load_export("build/gpu_exports/hhvv.pkl",reference["source"])
    np.testing.assert_array_equal(likelihood.initial,reference["initial"])
    np.testing.assert_array_equal(likelihood.parameter_names,reference["parameter_names"])
    initial = np.asarray(reference["initial"])
    direction = np.asarray(selected["direction"])
    low = np.array([p.minimum for p in likelihood.parameters])
    high = np.array([p.maximum for p in likelihood.parameters])
    original_path = Path("reports/gpu_curvature_channel_diagnostic.json")
    original = None
    if args.stable_leading or args.stable_cached_small or args.stable_cached_all:
        original_raw = original_path.read_bytes()
        original = json.loads(original_raw)
        assert original["status"] == "diagnostic_complete"
        assert original["reference_sha256"] == hashlib.sha256(raw).hexdigest()
        assert original["label"] == selected["label"]
    output = args.output or (Path("reports/gpu_curvature_cached_all.json") if args.stable_cached_all else
                             Path("reports/gpu_curvature_channel_cached_small.json") if args.stable_cached_small else
                             Path("reports/gpu_curvature_channel_stable.json") if original else original_path)
    if output.exists():
        raise FileExistsError(str(output))
    report = dict(status="running", source=reference["source"],
        reference=str(reference_path), reference_sha256=hashlib.sha256(raw).hexdigest(),
        combine_version=version("higgs-combine-tool"), root_version=reference["root_version"],
        runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path("scripts/caching_nll_diagnostics.py"),
                       Path("scripts/stable_root_intensities.py"),Path("scripts/cached_root_intensities.py"),
                       Path("scripts/probe_hhvv_stable_fd.py"))},
        selection_policy="Among unresolved original directions, choose the largest (max-min)/abs(median) curvature over the full saved grid; ties retain original order. No AD values are read.",
        selection_scores={c["label"]:score(c) for c in candidates}, label=selected["label"],
        direction=selected["direction"], initial=reference["initial"],
        parameter_names=reference["parameter_names"],
        scope="Diagnostic decomposition of actual Combine channel NLLs at unchanged probes; not a refined reference, Hessian acceptance or isolated timing baseline",
        probes=[])
    if original:
        report.update(channel_selection_from=str(original_path),
            channel_selection_from_sha256=hashlib.sha256(original_raw).hexdigest(),
            channel_selection_policy="The channel with the largest curvature range in the previous complete diagnostic; unchanged direction and steps, no AD input")
        if args.stable_cached_small:
            report["channel_selection_policy"] = "Largest channel curvature range over the final three saved steps; ties use the original channel order; unchanged direction and full grid, no AD input"
        if args.stable_cached_all:
            report["channel_selection_policy"] = "Every unmasked RooAddPdf channel without MultiPdf or external analytic-bin profiles; all excluded channel NLLs and nonchannel residual remain in the full result unchanged"

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        fit_path = Path("reports/hhvv_combine_fit.json")
        fit = json.loads(fit_path.read_text()) if fit_path.exists() else {}
        if fit.get("status") == "running":
            peak += fit["peak_rss_mib"]
        if peak > 11000:
            raise MemoryError("Combined local reference-job RSS peaks exceed 11,000 MiB")
        if time.perf_counter()-started > 1200:
            raise TimeoutError("Channel diagnostic exceeds its 1,200 s guard")

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
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(),workspace.data("data_obs"),model.GetNuisanceParameters())
            nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            parts, layout = channel_nlls(root,nll)
            assert len(parts) == len(likelihood.channels) == 459
            report.update(reflected_layout=layout, channels=[dict(pdf=str(p.pdf().GetName()),
                pdf_class=str(p.pdf().ClassName()), nll_name=str(p.GetName())) for p in parts])
            predictor = baseline = None
            cached_group, cached_baselines = [], None
            if args.stable_cached_all:
                report.update(stable_cached_channels=[], excluded_cached_channels=[])
                for index,(part,channel) in enumerate(zip(parts,likelihood.channels,strict=True)):
                    assert str(part.GetName()) == channel.name
                    dependencies = active_dependencies(likelihood.nodes,[channel.pdf])
                    reason = ("masked" if channel.masked else
                              "non-RooAddPdf" if str(part.pdf().ClassName()) != "RooAddPdf" else
                              "MultiPdf" if any(likelihood.nodes[n].class_name == "RooMultiPdf" for n in dependencies) else
                              "analytic-bin profile" if any(p.propagator in dependencies for p in likelihood.profiles) else None)
                    if reason:
                        report["excluded_cached_channels"].append(dict(index=index,channel=channel.name,reason=reason))
                    else:
                        cached_group.append((index,CachedRootIntensities(root,part)))
                assert cached_group
            if original and not args.stable_cached_all:
                chosen = original["largest_channel_curvature_ranges"][0]
                if args.stable_cached_small:
                    ranges = np.ptp([p["channel_curvatures"] for p in original["probes"][-3:]],axis=0)
                    channel_index = int(np.argmax(ranges))
                    chosen = original["channels"][channel_index]
                    report["small_step_channel_ranges"] = ranges.tolist()
                    report["channel_selection_steps"] = [p["step"] for p in original["probes"][-3:]]
                channel = next(c for c in likelihood.channels if c.name == chosen["nll_name"])
                assert not channel.masked
                dependencies = active_dependencies(likelihood.nodes,[channel.pdf])
                assert not any(p.propagator in dependencies for p in likelihood.profiles)
                channel_index = next(i for i,p in enumerate(parts) if str(p.GetName()) == channel.name)
                assert report["channels"][channel_index]["pdf"] == chosen["pdf"]
                predictor = CachedRootIntensities(root,parts[channel_index]) if args.stable_cached_small else RootIntensities(root,workspace.pdf(channel.pdf),
                    [workspace.var(n) for n in channel.observables],
                    np.column_stack([channel.data.coordinates[n] for n in channel.observables]),channel.data.weights)
                report["stable_channel"] = dict(index=channel_index,channel=channel.name,**predictor.report)

            def evaluate(theta):
                guard()
                assert np.all(theta >= low) and np.all(theta <= high)
                for variable,coordinate in zip(variables,theta,strict=True):
                    variable.setVal(float(coordinate))
                root.RooAbsReal.clearEvalErrorLog()
                full = float(nll.getVal())
                values = [float(p.getVal()) for p in parts]
                again = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                assert full == again and np.isfinite([full,*values]).all() and errors == 0
                residual = float(np.longdouble(full)-np.sum(values,dtype=np.longdouble))
                result = dict(full_nll=full,channel_nlls=values,nonchannel_residual=residual,
                              full_nll_after_channels=again,roofit_evaluation_errors=errors)
                if baseline is not None:
                    delta = float(predictor.delta(baseline))
                    assert np.isfinite(delta) and int(root.RooAbsReal.numEvalErrors()) == 0
                    after = float(nll.getVal())
                    assert full == after
                    result.update(stable_channel_delta=delta,full_nll_after_density_read=after,
                        stable_channel_delta_absolute_error=abs(delta-(values[channel_index]-zero["channel_nlls"][channel_index])))
                if cached_baselines is not None:
                    stable = np.array([p.delta(b) for (_,p),b in zip(cached_group,cached_baselines,strict=True)],dtype=np.longdouble)
                    cached = np.array([np.longdouble(values[i])-np.longdouble(zero["channel_nlls"][i]) for i,_ in cached_group])
                    correction = np.sum(stable-cached,dtype=np.longdouble)
                    after = float(nll.getVal())
                    assert after == full and int(root.RooAbsReal.numEvalErrors()) == 0
                    result.update(stable_cached_deltas=np.asarray(stable,dtype=float).tolist(),
                        stable_full_delta=float(np.longdouble(full)-np.longdouble(zero["full_nll"])+correction),
                        stable_cached_max_delta_error=float(np.max(abs(stable-cached),initial=0.)),
                        full_nll_after_cached_reads=after)
                return result

            canonical = evaluate(initial)
            assert abs(canonical["full_nll"]-reference["canonical_nominal_nll"]) < 1e-6
            report["canonical_nominal_nll"] = canonical["full_nll"]
            nll.setZeroPoint()
            zero = evaluate(initial)
            assert abs(zero["full_nll"]) < 1e-6
            report["zero_point"] = zero
            if predictor:
                baseline = predictor.predict()
                report["stable_channel"].update(predictor.report)
            if cached_group:
                cached_baselines = [p.predict() for _,p in cached_group]
                report["stable_cached_channels"] = [dict(index=i,channel=likelihood.channels[i].name,**p.report) for i,p in cached_group]
            save()
            for original_probe in selected["probes"]:
                step = original_probe["step"]
                samples = [dict(multiple=k,**evaluate(initial+k*step*direction)) for k in (-2,-1,1,2)]
                values = [s["full_nll"] for s in samples]
                components = [s["channel_nlls"] for s in samples]
                full_curvature = float(curvature(values,zero["full_nll"],step))
                channel_curvatures = curvature(components,zero["channel_nlls"],step)
                residual_curvature = float(curvature([s["nonchannel_residual"] for s in samples],zero["nonchannel_residual"],step))
                probe = dict(step=step,samples=samples,curvature=full_curvature,
                    original_curvature=original_probe["curvature"],
                    original_sample_max_absolute_error=float(np.max(abs(np.array(values)-original_probe["zero_point_nll_values"]))),
                    channel_curvatures=np.asarray(channel_curvatures,dtype=float).tolist(),
                    nonchannel_curvature=residual_curvature,
                    curvature_sum_absolute_error=float(abs(np.sum(channel_curvatures)+np.longdouble(residual_curvature)-full_curvature)))
                if predictor:
                    probe["stable_channel_curvature"] = float(curvature([s["stable_channel_delta"] for s in samples],0.,step))
                if cached_group:
                    probe["stable_full_curvature"] = float(curvature([s["stable_full_delta"] for s in samples],0.,step))
                    probe["stable_cached_curvatures"] = np.asarray(curvature([s["stable_cached_deltas"] for s in samples],0.,step),dtype=float).tolist()
                report["probes"].append(probe)
                assert probe["original_sample_max_absolute_error"] < 1e-6
                print(report["label"],step,full_curvature,"sample reproduction",probe["original_sample_max_absolute_error"],flush=True)
                save()
            evaluate(initial)
            baseline = None
            cached_baselines = None
            nll.clearZeroPoint()
            report["canonical_restoration_absolute_error"] = abs(evaluate(initial)["full_nll"]-canonical["full_nll"])
            assert report["canonical_restoration_absolute_error"] < 1e-6
            parts.clear()
            del nll
        table = np.array([p["channel_curvatures"] for p in report["probes"]])
        spreads = np.ptp(table,axis=0)
        report["largest_channel_curvature_ranges"] = [dict(index=int(i),**report["channels"][i],
            absolute_range=float(spreads[i]),curvatures=table[:,i].tolist()) for i in np.argsort(spreads)[-20:][::-1]]
        if predictor:
            maximum = max(s["stable_channel_delta_absolute_error"] for p in report["probes"] for s in p["samples"])
            report.update(stable_channel_delta_max_absolute_error=maximum,
                          stable_channel_delta_validation="passed" if maximum < 1e-6 else "failed",
                          stable_channel_curvature_range=float(np.ptp([p["stable_channel_curvature"] for p in report["probes"]])))
        if cached_group:
            maximum = max(s["stable_cached_max_delta_error"] for p in report["probes"] for s in p["samples"])
            report.update(stable_cached_delta_max_absolute_error=maximum,
                          stable_cached_delta_validation="passed" if maximum < 1e-6 else "failed",
                          stable_full_curvature_range=float(np.ptp([p["stable_full_curvature"] for p in report["probes"]])),
                          stable_full_small_step_curvatures=[p["stable_full_curvature"] for p in report["probes"][-3:]])
        report["status"] = "diagnostic_complete"
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
