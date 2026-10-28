"""Diagnose tiny HHVV derivatives using ROOT bin predictions and stable NLL deltas.

This independent reference is restricted to verified piecewise-constant extended
RooAddPdf channels with unit Gaussian constraints. It retains comparisons against
the actual Combine NLL and never replaces the raw reference reports.
"""

import argparse
import hashlib
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


LD = np.longdouble


def log_remainder(x):
    """x-log1p(x), avoiding subtraction for small x, in long-double arithmetic."""
    x = np.asarray(x, dtype=LD)
    small = abs(x) < LD('.125')
    y = np.where(small, x, 0)
    result = y*y/2
    power = y*y
    for k in range(3, 32):
        power *= -y
        result += power/k
    return np.where(small, result, x-np.log1p(x))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, help="Compare a full saved benchmark gradient with all 30 small bbVV mcstat references and three small single-bin signal-stat references")
    arguments = parser.parse_args()
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    started = time.perf_counter()
    prior_path = arguments.benchmark or Path("reports/hhvv_small_gradients.json")
    prior = json.loads(prior_path.read_text())
    reference_path = Path("reports/gpu_combine_gradient.json" if arguments.benchmark else "reports/gpu_combine_gradient_refined.json")
    refined = json.loads(reference_path.read_text())
    assert refined["source"] == prior["source"]
    likelihood = load_export("build/gpu_exports/hhvv.pkl", prior["source"])
    if arguments.benchmark:
        selected_cases = [c for c in refined["cases"] if c["status"] == "reference_ready" and (
            (c["name"].startswith("CMS_XHYbbWW_boosted_failmXbin") and abs(c["probes"][-1]["derivative"]) < 1e-4)
            or c["name"] in [f"CMSRun2_stat_Sig_CR_LPggF_Bin{i}" for i in (1,2,3)])]
        assert len(selected_cases) == 33
        artifact = Path(prior["arrays"])
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == prior["arrays_sha256"]
        with np.load(artifact, allow_pickle=False) as arrays:
            np.testing.assert_array_equal(arrays["point"], likelihood.initial)
            np.testing.assert_array_equal(arrays["parameter_names"], likelihood.parameter_names)
            assert np.isfinite(arrays["gradient"]).all()
            prior["cases"] = [dict(name=c["name"], ad=float(arrays["gradient"][c["index"]])) for c in selected_cases]
        prior["parameter_names"] = [c["name"] for c in selected_cases]
    selected = {c["name"]: c for c in refined["cases"] if c["name"] in prior["parameter_names"]}
    terms = [c for c in likelihood.channels if any(n in active_dependencies(likelihood.nodes, [c.pdf]) for n in selected)]
    assert np.finfo(LD).nmant >= 63
    root = load_caching_nll()
    report = dict(status="running", source=prior["source"], root_version=str(root.gROOT.GetVersion()),
                  combine_version=refined["combine_version"], runtime_flags=REFERENCE_FLAGS,
                  cpu_affinity=sorted(os.sched_getaffinity(0)), longdouble_mantissa_bits=np.finfo(LD).nmant,
                  selection=("All 30 bbVV mcstat cases with raw abs(FD)<1e-4, plus all three LPggF control-bin signal-stat derivatives; original two steps retained" if arguments.benchmark
                             else "Same 25 reference-selected small gradients; fixed step list from the prior scan"),
                  ad_report=str(prior_path), ad_report_sha256=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                  reference=str(reference_path), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  method="ROOT component bin probabilities, long-double normalization/summation, stable Poisson NLL difference plus exact unit Gaussian difference; five-point FD",
                  scope="Independent diagnostic on verified flat-bin channels only; raw Combine FD results remain unchanged",
                  channels=[], cases=[])
    output = arguments.benchmark.with_name(arguments.benchmark.stem+"_stable_fd.json") if arguments.benchmark else Path("reports/hhvv_stable_fd.json")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == prior["source"]["configuration"]
            model = workspace.obj("ModelConfig")
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"), model.GetNuisanceParameters())
            reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint(); reference.clearZeroPoint()
            report["canonical_nll"] = float(reference.getVal())
            reference.setZeroPoint()
            zero = float(reference.getVal())
            channel_inputs = {}
            for channel in terms:
                node = likelihood.nodes[channel.pdf]
                assert node.class_name == "RooAddPdf" and node.attrs["extended"]
                assert len(channel.observables) == 1 and not channel.masked
                observable = workspace.var(channel.observables[0])
                axis = root.RooArgSet(observable)
                binning = observable.getBinning()
                positions = channel.data.coordinates[channel.observables[0]]
                bins = [binning.binNumber(float(x)) for x in positions]
                assert len(set(bins)) == binning.numBins() == len(positions)
                widths = np.array([binning.binWidth(i) for i in bins], dtype=LD)
                pdfs = [workspace.pdf(n) for n in node.attrs["pdfs"]]
                assert all(p.ClassName() in ("FastVerticalInterpHistPdf2", "RooParametricStepFunction", "RooUniform") for p in pdfs)
                coefs = [workspace.function(n) or workspace.var(n) for n in node.attrs["coefs"]]
                max_flatness_error = 0.
                for pdf in pdfs:
                    for i in bins:
                        lo, hi = binning.binLow(i), binning.binHigh(i)
                        observable.setVal(lo+.25*(hi-lo)); a = float(pdf.getVal(axis))
                        observable.setVal(lo+.75*(hi-lo)); b = float(pdf.getVal(axis))
                        max_flatness_error = max(max_flatness_error, abs(a-b))
                assert max_flatness_error < 1e-12
                entry = dict(name=channel.name, component_classes=[str(p.ClassName()) for p in pdfs],
                             bins=bins, widths=[float(x) for x in widths], max_flatness_error=max_flatness_error,
                             max_component_normalization_error=0., max_total_yield_sum_error=0.)
                report["channels"].append(entry)

                def predict(observable=observable, axis=axis, positions=positions, widths=widths,
                            pdfs=pdfs, coefs=coefs, entry=entry):
                    means = np.zeros(len(positions), dtype=LD)
                    expected = LD(0)
                    for pdf, coef in zip(pdfs, coefs, strict=True):
                        probabilities = []
                        for x in positions:
                            observable.setVal(float(x))
                            probabilities.append(LD(pdf.getVal(axis)))
                        probabilities = np.array(probabilities)*widths
                        norm = probabilities.sum()
                        entry["max_component_normalization_error"] = max(entry["max_component_normalization_error"], float(abs(norm-1)))
                        assert abs(norm-1) < 1e-10 and np.all(probabilities >= 0)
                        coefficient = LD(coef.getVal())
                        expected += coefficient
                        means += coefficient*probabilities/norm
                    entry["max_total_yield_sum_error"] = max(entry["max_total_yield_sum_error"], float(abs(means.sum()-expected)))
                    assert np.all(means > 0)
                    return means

                baseline = predict()
                channel_inputs[channel.name] = (predict, baseline, np.asarray(channel.data.weights, dtype=LD))
                entry["nominal_bin_means"] = [float(x) for x in baseline]
                entry["weights"] = channel.data.weights.tolist()
            ad = {c["name"]: c["ad"] for c in prior["cases"]}
            for name, case in selected.items():
                variable = workspace.var(name)
                theta0 = variable.getVal()
                assert theta0 == 0.
                channels = [c for c in terms if name in active_dependencies(likelihood.nodes, [c.pdf])]
                constraints = [likelihood.nodes[c] for c in likelihood.constraints if name in active_dependencies(likelihood.nodes, [c])]
                assert len(channels) == len(constraints) == 1
                constraint = constraints[0]
                assert constraint.class_name == "SimpleGaussianConstraint"
                assert constraint.inputs[0] == name
                assert likelihood.nodes[constraint.inputs[1]].attrs["value"] == 0.
                assert likelihood.nodes[constraint.inputs[2]].attrs["value"] == 1.
                predict, baseline, weights = channel_inputs[channels[0].name]
                probes = []
                for old_probe in case["probes"]:
                    step = old_probe["step"]
                    assert old_probe["stencil"] == "central"
                    values, checks = [], []
                    for multiple in (-2, -1, 1, 2):
                        coordinate = multiple*step
                        variable.setVal(coordinate)
                        combine_delta = float(reference.getVal())-zero
                        means = predict()
                        delta = means-baseline
                        ratio = delta/baseline
                        stable = np.sum((1-weights/baseline)*delta + weights*log_remainder(ratio)) + LD('.5')*LD(coordinate)**2
                        values.append(stable)
                        error = abs(float(stable)-combine_delta)
                        assert error < 1e-6
                        checks.append(dict(coordinate=coordinate, stable_delta=float(stable), combine_delta=combine_delta, absolute_error=error))
                    derivative = (8*(values[2]-values[1])-(values[3]-values[0]))/(12*LD(step))
                    absolute = abs(ad[name]-float(derivative))
                    probes.append(dict(step=step, derivative=float(derivative), original_combine_fd=old_probe["derivative"],
                                       ad_absolute_error=absolute, ad_relative_error=absolute/max(abs(float(derivative)),1e-8), delta_checks=checks))
                    variable.setVal(theta0)
                    reference.getVal()
                report["cases"].append(dict(name=name, channel=channels[0].name, ad=ad[name], probes=probes))
                save()
                print(name, "AD", ad[name], "stable FD at original smaller step", probes[1]["derivative"],
                      "relative error", probes[1]["ad_relative_error"], flush=True)
            del reference
        report.update(status="diagnostic_complete",
                      original_step_status="passed" if all(c["probes"][1]["ad_relative_error"] < 1e-4 for c in report["cases"]) else "failed",
                      original_step_halving_max_relative_change=max(abs(c["probes"][0]["derivative"]-c["probes"][1]["derivative"])/max(abs(c["probes"][1]["derivative"]),1e-8) for c in report["cases"]),
                      original_step_max_relative_error=max(c["probes"][1]["ad_relative_error"] for c in report["cases"]),
                      original_step_max_absolute_error=max(c["probes"][1]["ad_absolute_error"] for c in report["cases"]),
                      max_combine_delta_absolute_error=max(d["absolute_error"] for c in report["cases"] for p in c["probes"] for d in p["delta_checks"]))
        save()
        print(report["status"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
