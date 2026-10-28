"""Check floor transitions at the unchanged leading-channel curvature probes."""

import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np

from combine_jax.discrete import active_dependencies
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from root_histogram_floors import floor_view_class
from stable_root_intensities import RootIntensities
from probe_gpu_curvature_channels import curvature


def main():
    os.sched_setaffinity(0,[sorted(os.sched_getaffinity(0))[1]])
    started=time.perf_counter()
    reference_path=Path("reports/gpu_curvature_channel_stable.json")
    reference_raw=reference_path.read_bytes(); reference=json.loads(reference_raw)
    assert reference["status"] == "diagnostic_complete"
    assert reference["stable_channel_delta_validation"] == "passed"
    likelihood=load_export("build/gpu_exports/hhvv.pkl",reference["source"])
    np.testing.assert_array_equal(likelihood.parameter_names,reference["parameter_names"])
    initial=np.asarray(reference["initial"]); direction=np.asarray(reference["direction"])
    np.testing.assert_array_equal(initial,likelihood.initial)
    channel=next(c for c in likelihood.channels if c.name==reference["stable_channel"]["channel"])
    dependencies=active_dependencies(likelihood.nodes,[channel.pdf])
    assert not channel.masked and not any(p.propagator in dependencies for p in likelihood.profiles)
    output=Path("reports/gpu_curvature_histogram_floors.json")
    artifact=Path("build/gpu_results/hhvv_curvature_histogram_floors.npz")
    if output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous histogram-floor diagnostics")
    report=dict(status="running",source=reference["source"],reference=str(reference_path),
        reference_sha256=hashlib.sha256(reference_raw).hexdigest(),
        combine_version=version("higgs-combine-tool"),root_version=reference["root_version"],
        runtime_flags=REFERENCE_FLAGS,cpu_affinity=sorted(os.sched_getaffinity(0)),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path("scripts/root_histogram_floors.py"),Path("scripts/stable_root_intensities.py"))},
        channel=channel.name,direction_label=reference["label"],floor=1e-9,
        policy="Use the preceding diagnostic's channel, direction and complete step grid; inspect all direct additive 2D stored morphs using installed Combine Meld/CropUnderflows/Normalize methods on copies. No AD arrays, likelihood modifications or new step selection.",
        scope="Histogram-floor diagnostic for one channel and one direction; no Hessian or full curvature-reference acceptance",
        samples=[],probes=[])

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=output.with_suffix(".json.tmp"); temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");temp.replace(output)

    def guard():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        fit_path=Path("reports/hhvv_combine_fit.json")
        fit=json.loads(fit_path.read_text()) if fit_path.exists() else {}
        if fit.get("status")=="running": peak+=fit["peak_rss_mib"]
        if peak>11000: raise MemoryError("Combined local reference-job RSS exceeds 11,000 MiB")
        if time.perf_counter()-started>600: raise TimeoutError("Histogram-floor diagnostic exceeds 600 s")

    root=load_caching_nll(); view_class=floor_view_class()
    assert str(root.gROOT.GetVersion())==reference["root_version"]
    assert report["combine_version"]==reference["combine_version"]
    old_mode=root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    raw_arrays=[]; rebuilt_arrays=[]; root_arrays=[]; points=[]
    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace)==reference["source"]["configuration"]
            variables=[workspace.var(n) for n in likelihood.parameter_names]
            for variable,value in zip(variables,initial,strict=True): variable.setVal(float(value))
            predictor=RootIntensities(root,workspace.pdf(channel.pdf),
                [workspace.var(n) for n in channel.observables],
                np.column_stack([channel.data.coordinates[n] for n in channel.observables]),channel.data.weights)
            baseline=predictor.predict()
            components=[]
            for pdf in predictor.functions:
                if str(pdf.ClassName())!="FastVerticalInterpHistPdf2D2":
                    continue
                assert pdf.smoothAlgo()>=0 and not pdf.conditional()
                node=likelihood.nodes[str(pdf.GetName())]
                assert node.class_name=="FastVerticalInterpHistPdf2D2"
                axes=[workspace.var(n) for n in node.attrs["axes"]]
                coordinates=np.array(np.meshgrid(*[(e[:-1]+e[1:])/2 for e in node.attrs["edges"]],indexing="ij")).reshape(2,-1).T
                components.append((pdf,view_class(pdf),axes,coordinates))
            assert len(components)==12
            report["components"]=[str(p.GetName()) for p,_,_,_ in components]
            report["other_components"]=[dict(name=str(p.GetName()),class_name=str(p.ClassName()))
                                        for p in predictor.functions if str(p.GetName()) not in report["components"]]
            nominal_mask=None

            def evaluate(theta,step,multiple,original=None):
                nonlocal nominal_mask
                guard()
                for variable,value in zip(variables,theta,strict=True): variable.setVal(float(value))
                root.RooAbsReal.clearEvalErrorLog()
                delta=float(predictor.delta(baseline))
                raw_bins=[]; rebuilt_bins=[]; original_bins=[]
                for pdf,view,axes,coordinates in components:
                    raw,rebuilt,cached=np.asarray(view.bins(),dtype=float)
                    np.testing.assert_array_equal(rebuilt,cached)
                    norm=root.RooArgSet(*axes); actual=[]
                    for row in coordinates:
                        for axis,value in zip(axes,row,strict=True): axis.setVal(float(value))
                        actual.append(float(pdf.getVal(norm)))
                    np.testing.assert_allclose(rebuilt,actual,atol=1e-14,rtol=1e-13)
                    raw_bins.append(raw);rebuilt_bins.append(rebuilt);original_bins.append(actual)
                raw_bins=np.asarray(raw_bins); rebuilt_bins=np.asarray(rebuilt_bins); original_bins=np.asarray(original_bins)
                assert np.isfinite(raw_bins).all() and np.isfinite(original_bins).all()
                mask=raw_bins<1e-9
                if nominal_mask is None: nominal_mask=mask.copy()
                changed=np.argwhere(mask!=nominal_mask)
                sample=dict(step=step,multiple=multiple,stable_delta=delta,
                    floor_active_bins=int(mask.sum()),floor_transitions_from_nominal=len(changed),
                    changed_bins=[dict(component=int(c),bin=int(b),nominal_raw=float(raw_arrays[0][c,b]),raw=float(raw_bins[c,b])) for c,b in changed],
                    root_density_max_absolute_error=float(np.max(abs(rebuilt_bins-original_bins))),
                    roofit_evaluation_errors=int(root.RooAbsReal.numEvalErrors()))
                assert sample["roofit_evaluation_errors"]==0
                if original is not None:
                    sample["previous_stable_delta_absolute_error"]=abs(delta-original["stable_channel_delta"])
                    assert sample["previous_stable_delta_absolute_error"]<1e-6
                raw_arrays.append(raw_bins);rebuilt_arrays.append(rebuilt_bins);root_arrays.append(original_bins);points.append(theta.copy())
                report["samples"].append(sample)
                return sample

            evaluate(initial,0.,0)
            save()
            for previous in reference["probes"]:
                step=previous["step"]
                samples=[evaluate(initial+s["multiple"]*step*direction,step,s["multiple"],s) for s in previous["samples"]]
                probe=dict(step=step,stable_channel_curvature=float(curvature([s["stable_delta"] for s in samples],0.,step)),
                    previous_stable_channel_curvature=previous["stable_channel_curvature"],
                    floor_transitions=[s["floor_transitions_from_nominal"] for s in samples])
                report["probes"].append(probe)
                print(step,probe["stable_channel_curvature"],probe["floor_transitions"],flush=True)
                save()
            for variable,value in zip(variables,initial,strict=True): variable.setVal(float(value))
            report["restored_stable_delta"]=float(predictor.delta(baseline))
            assert abs(report["restored_stable_delta"])<1e-6
            components.clear()
        artifact.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(artifact,parameter_names=likelihood.parameter_names,initial=initial,direction=direction,
            components=report["components"],points=np.array(points),raw_bins=np.array(raw_arrays),
            rebuilt_bins=np.array(rebuilt_arrays),root_bins=np.array(root_arrays))
        report.update(status="diagnostic_complete",arrays=str(artifact),arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
            root_density_max_absolute_error=max(s["root_density_max_absolute_error"] for s in report["samples"]),
            previous_stable_delta_max_absolute_error=max(s.get("previous_stable_delta_absolute_error",0.) for s in report["samples"]))
        save()
    except Exception:
        report.update(status="failed",exception=traceback.format_exc());save();raise
    finally:
        root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__=="__main__":
    main()
