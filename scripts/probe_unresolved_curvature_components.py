"""Resolve the component error budget of the first uncertain native direction.

Repeat the original twelve steps and preserve all native channel and residual
contributions. This diagnostic reads no AD arrays and changes no reference gate.
"""

from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import resource
import threading
import time
import traceback

import numpy as np

from cached_root_intensities import CachedRootIntensities
from cached_root_real_sum import CachedRootRealSum
from caching_nll_diagnostics import channel_nlls
from combine_jax.discrete import active_dependencies
from combine_jax.root_io import REFERENCE_FLAGS, load_caching_nll, open_workspace
from compare_gpu_cached_curvature import audit_reference
from compare_gpu_curvature import first_converged_triple
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv
from probe_gpu_curvature_channels import curvature


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    # A separate owned-process wrapper guards this probe and both live fit trees.
    assert os.sched_getaffinity(0) == {5}
    output = Path("reports/gpu_combine_curvature_unresolved_components.json")
    if output.exists():
        raise FileExistsError("Preserve previous curvature pilot evidence")
    gate_path = Path("reports/cached_root_real_sum_tests.json")
    gate = json.loads(gate_path.read_text())
    assert gate["status"] == "passed"
    assert all(digest(p) == checksum for p, checksum in gate["sources"].items())
    parent_path = Path("reports/gpu_combine_curvature_cached.json")
    parent, parent_audit = audit_reference(parent_path)
    selected = next(c for c in parent["cases"] if c["reference_status"] == "unresolved")
    assert len(parent["steps"]) == len(selected["probes"]) == 12
    prior_path = Path("reports/gpu_combine_curvature_real_sum_pilot.json")
    prior = json.loads(prior_path.read_bytes())
    assert prior["status"] == "pilot_checked" and prior["label"] == selected["label"]
    assert prior["parent_reference_sha256"] == digest(parent_path)
    assert prior["selected_triple"] is None
    started = time.perf_counter()
    report = dict(status="running", stage="setup", source=parent["source"],
        parent_reference=str(parent_path), parent_reference_sha256=digest(parent_path),
        prior_pilot=str(prior_path), prior_pilot_sha256=digest(prior_path),
        parent_audit=parent_audit, validation_gate=str(gate_path), validation_gate_sha256=digest(gate_path),
        combine_version=version("higgs-combine-tool"), root_version=parent["root_version"],
        runtime_flags=REFERENCE_FLAGS, cpu_affinity=sorted(os.sched_getaffinity(0)),
        rss_limit_mib=3500, wall_limit_seconds=1200,
        runner_sha256={str(p): digest(p) for p in (
            Path(__file__), Path("scripts/cached_root_real_sum.py"), Path("scripts/cached_root_intensities.py"),
            Path("scripts/caching_nll_diagnostics.py"), Path("scripts/stable_root_intensities.py"),
            Path("scripts/probe_gpu_curvature_channels.py"), Path("scripts/compare_gpu_curvature.py"))},
        policy="First unresolved direction in the frozen reference, every original step and signed sample. Compare all native NLL samples within 1e-6. Keep every excluded channel and constraint residual unchanged. Do not consult AD or change the convergence rule.",
        scope="Native-only component diagnosis for the first unresolved direction. All channels and the native nonchannel residual remain included; no AD comparison or reference promotion. This local probe overlaps the ongoing fits on a separate CPU core and is not an isolated timing measurement.",
        label=selected["label"], direction=selected["direction"], steps=parent["steps"],
        probes=[])
    stop = threading.Event()
    lock = threading.RLock()

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            temporary = output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(output)

    def monitor():
        while not stop.wait(.1):
            if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 3500
                    or time.perf_counter()-started > 1200):
                with lock:
                    report.update(status="resource_limit", exception="Pilot memory or wall-time guard exceeded")
                    save()
                os._exit(86)

    save()
    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    root = None
    old_mode = None
    try:
        likelihood = load_export("build/gpu_exports/hhvv.pkl", parent["source"])
        initial = np.asarray(parent["initial"])
        np.testing.assert_array_equal(initial, likelihood.initial)
        np.testing.assert_array_equal(parent["parameter_names"], likelihood.parameter_names)
        low = np.array([p.minimum for p in likelihood.parameters])
        high = np.array([p.maximum for p in likelihood.parameters])
        root = load_caching_nll()
        assert str(root.gROOT.GetVersion()) == report["root_version"]
        assert report["combine_version"] == parent["combine_version"]
        old_mode = root.RooAbsReal.evalErrorLoggingMode()
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == parent["source"]["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            for variable, value in zip(variables, initial, strict=True):
                variable.setVal(float(value))
            model = workspace.obj("ModelConfig")
            nll = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"),
                                               model.GetNuisanceParameters())
            nll.setChannelMasks(model.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint()
            nll.clearZeroPoint()
            parts, layout = channel_nlls(root, nll)
            assert len(parts) == len(likelihood.channels) == 459
            group, excluded = [], []
            for index, (part, channel) in enumerate(zip(parts, likelihood.channels, strict=True)):
                assert str(part.GetName()) == channel.name and not channel.masked
                dependencies = active_dependencies(likelihood.nodes, [channel.pdf])
                cls = str(part.pdf().ClassName())
                if any(likelihood.nodes[n].class_name == "RooMultiPdf" for n in dependencies):
                    excluded.append(dict(index=index, channel=channel.name, reason="MultiPdf"))
                elif cls == "RooRealSumPdf":
                    group.append((index, CachedRootRealSum(root, part)))
                elif cls == "RooAddPdf":
                    group.append((index, CachedRootIntensities(root, part)))
                else:
                    raise NotImplementedError(f"Unexpected channel class {cls}: {channel.name}")
            assert len(group) == 456 and len(excluded) == 3
            report.update(reflected_layout=layout, excluded_cached_channels=excluded)
            canonical = float(nll.getVal())
            assert abs(canonical-parent["canonical_nominal_nll"]) < 1e-6
            nll.setZeroPoint()
            zero = float(nll.getVal())
            assert abs(zero) < 1e-6
            zero_channels = np.asarray([p.getVal() for p in parts], dtype=np.longdouble)
            baselines = [p.predict() for _, p in group]
            report.update(canonical_nominal_nll=canonical, nominal_zero_point_nll=zero,
                stable_cached_channels=[dict(index=i, channel=likelihood.channels[i].name, **p.report) for i, p in group])
            assert sum(p.report["class_name"] == "RooRealSumPdf" for _, p in group) == 182
            report["component_names"] = [c.name for c in likelihood.channels]+["native_nonchannel_residual"]
            report["component_modes"] = [
                "cached_stable_delta" if i in {index for index, _ in group} else "native_channel_delta"
                for i in range(len(parts))]+["native_full_minus_channels_delta"]
            save()

            def evaluate(point):
                assert np.all(point >= low) and np.all(point <= high)
                for variable, value in zip(variables, point, strict=True):
                    variable.setVal(float(value))
                root.RooAbsReal.clearEvalErrorLog()
                full = float(nll.getVal())
                channels = np.asarray([p.getVal() for p in parts], dtype=np.longdouble)
                stable = np.asarray([p.delta(b) for (_, p), b in zip(group, baselines, strict=True)], dtype=np.longdouble)
                original = np.asarray([channels[i]-zero_channels[i] for i, _ in group], dtype=np.longdouble)
                correction = np.sum(stable-original, dtype=np.longdouble)
                errors = abs(stable-original)
                maximum = float(np.max(errors, initial=0.))
                count = int(root.RooAbsReal.numEvalErrors())
                assert np.isfinite(full) and np.isfinite(stable).all()
                assert maximum < 1e-6 and abs(correction) < 1e-6 and count == 0
                assert float(nll.getVal()) == full
                component_deltas = np.empty(len(parts)+1, dtype=np.longdouble)
                component_deltas[:-1] = channels-zero_channels
                component_deltas[-1] = np.longdouble(full)-np.longdouble(zero)-np.sum(component_deltas[:-1], dtype=np.longdouble)
                for (index, _), delta in zip(group, stable, strict=True):
                    component_deltas[index] = delta
                stable_delta = float(np.longdouble(full)-np.longdouble(zero)+correction)
                reconstruction = float(np.sum(component_deltas, dtype=np.longdouble))
                assert abs(reconstruction-stable_delta) < 1e-9
                return dict(full_nll=full, stable_delta=stable_delta,
                    component_deltas=np.asarray(component_deltas, dtype=float).tolist(),
                    component_reconstruction_absolute_error=abs(reconstruction-stable_delta),
                    stable_channel_delta_max_absolute_error=maximum,
                    stable_full_correction=float(correction), roofit_evaluation_errors=count,
                    largest_channel_correction_index=group[int(np.argmax(errors))][0])

            direction = np.asarray(selected["direction"])
            for step, previous in zip(parent["steps"], selected["probes"], strict=True):
                report["stage"] = f"curvature {selected['label']} step {step}"
                samples = []
                for multiple, old in zip((-2, -1, 1, 2), previous["samples"], strict=True):
                    sample = evaluate(initial+multiple*step*direction)
                    sample.update(multiple=multiple, original_sample_absolute_error=abs(sample["full_nll"]-old["full_nll"]))
                    assert sample["original_sample_absolute_error"] < 1e-6
                    samples.append(sample)
                probe = dict(step=step, samples=samples, original_curvature=previous["curvature"],
                              curvature=float(curvature([s["stable_delta"] for s in samples], 0., step)))
                component_curvatures = np.asarray(curvature([s["component_deltas"] for s in samples], 0., step), dtype=float)
                discrepancy = abs(float(np.sum(component_curvatures, dtype=np.longdouble))-probe["curvature"])
                assert discrepancy < max(1e-4, 1e-8*abs(probe["curvature"]))
                prior_probe = prior["probes"][len(report["probes"])]
                assert prior_probe["step"] == step
                prior_error = max(abs(a["stable_delta"]-b["stable_delta"])
                                  for a, b in zip(samples, prior_probe["samples"], strict=True))
                assert prior_error < 1e-9
                probe.update(component_curvatures=component_curvatures.tolist(),
                    component_curvature_sum_absolute_error=discrepancy,
                    prior_pilot_max_stable_delta_difference=prior_error)
                report["probes"].append(probe)
                save()
                print(selected["label"], step, probe["curvature"], flush=True)
            restored = evaluate(initial)
            assert abs(restored["full_nll"]-zero) < 1e-6 and abs(restored["stable_delta"]) < 1e-6
            nll.clearZeroPoint()
            restoration_error = abs(float(nll.getVal())-canonical)
            assert restoration_error < 1e-6
            triple = first_converged_triple(report["probes"])
            changes = []
            for before, after in zip(report["probes"], report["probes"][1:]):
                delta = np.asarray(after["component_curvatures"])-np.asarray(before["component_curvatures"])
                total = after["curvature"]-before["curvature"]
                order = np.argsort(-abs(delta), kind="stable")[:15]
                changes.append(dict(steps=[before["step"], after["step"]],
                    full_curvature_change=total,
                    relative_change=abs(total)/max(abs(after["curvature"]), 1e-8),
                    component_change_sum_error=abs(float(np.sum(delta, dtype=np.longdouble))-total),
                    largest_components=[dict(index=int(i), name=report["component_names"][i],
                        mode=report["component_modes"][i], curvature_change=float(delta[i])) for i in order]))
            report.update(adjacent_step_component_changes=changes)
            report.update(status="diagnostic_complete", stage="finished", selected_triple=triple,
                curvature_reference_status="converged" if triple else "unresolved",
                canonical_restoration_absolute_error=restoration_error)
            save()
            print(report["status"], report["curvature_reference_status"], flush=True)
            del group, parts, nll
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise
    finally:
        stop.set()
        watcher.join()
        if root is not None and old_mode is not None:
            root.RooAbsReal.clearEvalErrorLog()
            root.RooAbsReal.setEvalErrorLoggingMode(old_mode)


if __name__ == "__main__":
    main()
