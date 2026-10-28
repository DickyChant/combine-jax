"""Inspect native bin NLL contributions and floor states in an uncertain channel.

Select the leading channel of the largest adjacent-step change in the saved
native-only component diagnostic; use the same direction and all twelve steps.
"""

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
from stable_root_intensities import RootIntensities, extended_delta
from probe_hhvv_stable_fd import log_remainder
from probe_gpu_curvature_channels import curvature


def main():
    assert os.sched_getaffinity(0) == {5}
    started=time.perf_counter()
    reference_path=Path("reports/gpu_combine_curvature_unresolved_components.json")
    reference_raw=reference_path.read_bytes(); components_reference=json.loads(reference_raw)
    assert components_reference["status"] == "diagnostic_complete"
    parent_path=Path(components_reference["parent_reference"])
    parent_raw=parent_path.read_bytes(); parent=json.loads(parent_raw)
    assert hashlib.sha256(parent_raw).hexdigest()==components_reference["parent_reference_sha256"]
    worst=max(components_reference["adjacent_step_component_changes"],key=lambda r:r["relative_change"])
    selected=worst["largest_components"][0]
    channel_index=selected["index"]
    assert channel_index<459 and selected["mode"]=="cached_stable_delta"
    reference=dict(source=components_reference["source"],root_version=components_reference["root_version"],
        combine_version=components_reference["combine_version"],initial=parent["initial"],
        parameter_names=parent["parameter_names"],label=components_reference["label"],
        direction=components_reference["direction"],stable_channel=dict(channel=selected["name"]),
        probes=[dict(step=p["step"],stable_channel_curvature=p["component_curvatures"][channel_index],
                     samples=[dict(multiple=v["multiple"],stable_channel_delta=v["component_deltas"][channel_index])
                              for v in p["samples"]]) for p in components_reference["probes"]])
    likelihood=load_export("build/gpu_exports/hhvv.pkl",reference["source"])
    np.testing.assert_array_equal(likelihood.parameter_names,reference["parameter_names"])
    initial=np.asarray(reference["initial"]); direction=np.asarray(reference["direction"])
    np.testing.assert_array_equal(initial,likelihood.initial)
    channel=next(c for c in likelihood.channels if c.name==reference["stable_channel"]["channel"])
    dependencies=active_dependencies(likelihood.nodes,[channel.pdf])
    assert not channel.masked and not any(p.propagator in dependencies for p in likelihood.profiles)
    output=Path("reports/gpu_unresolved_curvature_bins.json")
    artifact=Path("build/gpu_results/hhvv_unresolved_curvature_bins.npz")
    if output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous histogram-floor diagnostics")
    report=dict(status="running",source=reference["source"],reference=str(reference_path),
        reference_sha256=hashlib.sha256(reference_raw).hexdigest(),
        parent_reference=str(parent_path),parent_reference_sha256=hashlib.sha256(parent_raw).hexdigest(),
        selected_channel_index=channel_index,selection_pair=worst["steps"],
        selection_policy="Choose the largest relative adjacent-step change in the complete native component record, then its largest absolute component change. No AD input or new step selection.",
        combine_version=version("higgs-combine-tool"),root_version=reference["root_version"],
        runtime_flags=REFERENCE_FLAGS,cpu_affinity=sorted(os.sched_getaffinity(0)),
        runner_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      (Path(__file__),Path("scripts/root_histogram_floors.py"),Path("scripts/stable_root_intensities.py"))},
        channel=channel.name,direction_label=reference["label"],floor=1e-9,
        policy="Use the preceding diagnostic's channel, direction and complete step grid; inspect all direct additive 2D stored morphs using installed Combine Meld/CropUnderflows/Normalize methods on copies. No AD arrays, likelihood modifications or new step selection.",
        scope="Native bin-contribution and histogram-floor diagnostic for one channel/direction; no Hessian or full curvature-reference acceptance. This separate core-5 probe overlaps the ongoing local fits and has an external combined-memory guard.",
        samples=[],probes=[])

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=output.with_suffix(".json.tmp"); temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");temp.replace(output)

    def guard():
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        if peak>3500: raise MemoryError("Native bin diagnostic exceeds 3500 MiB RSS")
        if time.perf_counter()-started>600: raise TimeoutError("Histogram-floor diagnostic exceeds 600 s")

    root=load_caching_nll(); view_class=floor_view_class()
    assert str(root.gROOT.GetVersion())==reference["root_version"]
    assert report["combine_version"]==reference["combine_version"]
    old_mode=root.RooAbsReal.evalErrorLoggingMode()
    root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
    raw_arrays=[]; rebuilt_arrays=[]; root_arrays=[]; points=[]
    intensity_arrays=[]; expected_arrays=[]; bin_delta_arrays=[]; coefficient_arrays=[]
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
            assert components, "No additive 2D morphs in the selected channel"
            assert len({len(c) for _,_,_,c in components})==1, "Different template grid sizes unsupported"
            report["components"]=[str(p.GetName()) for p,_,_,_ in components]
            report["other_components"]=[dict(name=str(p.GetName()),class_name=str(p.ClassName()))
                                        for p in predictor.functions if str(p.GetName()) not in report["components"]]
            report["data_rows"]=len(predictor.weights)
            report["mixture_components"]=[str(p.GetName()) for p in predictor.functions]
            report["row_observables"]=list(channel.observables)
            report["bin_contribution_order"]="Expected-yield difference, followed by -weight*log(intensity/nominal intensity) at each exported channel row; zero-weight rows contribute zero"
            report["longdouble_mantissa_bits"]=int(np.finfo(np.longdouble).nmant)
            nominal_mask=None

            def evaluate(theta,step,multiple,original=None):
                nonlocal nominal_mask
                guard()
                for variable,value in zip(variables,theta,strict=True): variable.setVal(float(value))
                root.RooAbsReal.clearEvalErrorLog()
                expected,intensities=predictor.predict()
                delta=float(extended_delta(expected,baseline[0],intensities,baseline[1],predictor.weights))
                counted=predictor.weights>0
                ratio=np.where(counted,intensities-baseline[1],np.longdouble(0))/np.where(counted,baseline[1],np.longdouble(1))
                bin_deltas=np.r_[np.array([expected-baseline[0]],dtype=np.longdouble),
                    predictor.weights*(-ratio+log_remainder(ratio))]
                reconstruction=float(abs(np.sum(bin_deltas,dtype=np.longdouble)-np.longdouble(delta)))
                assert reconstruction<1e-9
                intensity_arrays.append(intensities.copy()); expected_arrays.append(expected)
                bin_delta_arrays.append(bin_deltas.copy())
                coefficient_arrays.append(np.asarray([c.getVal() for c in predictor.coefficients],dtype=float))
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
                    bin_delta_sum_absolute_error=reconstruction,
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
                bin_curvature=np.asarray(curvature(bin_delta_arrays[-4:],0.,step),dtype=float)
                sum_error=abs(float(np.sum(bin_curvature,dtype=np.longdouble))-probe["stable_channel_curvature"])
                assert sum_error<max(1e-4,abs(probe["stable_channel_curvature"])*1e-8)
                probe.update(bin_curvatures=bin_curvature.tolist(),bin_curvature_sum_absolute_error=sum_error)
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
            rebuilt_bins=np.array(rebuilt_arrays),root_bins=np.array(root_arrays),
            weights=np.asarray(predictor.weights),positions=predictor.positions,
            expected=np.asarray(expected_arrays),intensities=np.asarray(intensity_arrays),
            bin_deltas=np.asarray(bin_delta_arrays),coefficients=np.asarray(coefficient_arrays))
        changes=[]
        for before,after in zip(report["probes"],report["probes"][1:]):
            differences=np.asarray(after["bin_curvatures"])-np.asarray(before["bin_curvatures"])
            indices=np.argsort(-abs(differences),kind="stable")[:15]
            changes.append(dict(steps=[before["step"],after["step"]],
                total_curvature_change=after["stable_channel_curvature"]-before["stable_channel_curvature"],
                leading_bins=[dict(index=int(i),row_index=int(i-1) if i else None,
                    coordinates=predictor.positions[i-1].tolist() if i else None,
                    weight=float(predictor.weights[i-1]) if i else None,
                    curvature_change=float(differences[i])) for i in indices]))
        report["adjacent_bin_changes"]=changes
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
