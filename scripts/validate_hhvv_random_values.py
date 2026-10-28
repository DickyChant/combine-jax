"""Compare full and per-channel HHVV NLL values at bounded random points."""

import json
from pathlib import Path
import resource
import time
import traceback

import jax
import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def main():
    started = time.perf_counter()
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    source = manifest["models"]["hhvv"]["source"]
    likelihood = load_export("build/gpu_exports/hhvv.pkl", source)
    evaluator = likelihood.local_evaluator(constraint_batch_size=32)
    initial = likelihood.initial
    scales = np.array([p.error if p.error > 0 else 1. for p in likelihood.parameters])
    low = np.array([p.minimum for p in likelihood.parameters])
    high = np.array([p.maximum for p in likelihood.parameters])
    margin = np.minimum(.01*scales, .01*(high-low))
    report = dict(status="running", source=source, combine_version=manifest["combine_version"],
                  root_version=manifest["root_version"], jax_version=jax.__version__, float64=jax.config.jax_enable_x64,
                  runtime_flags=REFERENCE_FLAGS, seed=20007319, parameter_names=list(likelihood.parameter_names),
                  initial=initial.tolist(), parameter_scales=scales.tolist(), cases=[], rejected=[], timings={},
                  scope="Nominal plus 25 valid bounded random points, per-channel and full NLL; no gradient/Hessian/fit acceptance",
                  sampling="Uniform +/-2 saved errors, fallback scale one for zero errors; clip inside bounds with 1% margin, fixed bounds held; invalid-domain draws logged and resampled up to 250 attempts",
                  constraint_batch_size=32)
    output = Path("reports/hhvv_random_values.json")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > 11500:
            raise MemoryError("11,500 MiB RSS guard exceeded")

    try:
        assert report["float64"]
        start = time.perf_counter()
        corrections = {}
        for i, term in enumerate(evaluator.terms):
            term.value = term.value.lower(initial[term.indices]).compile()
            if term.profiles:
                function = jax.jit(evaluator._function(None, term.constraints, term.parameter_names, term.profiles))
                corrections[term.name] = function.lower(initial[term.indices]).compile()
            guard()
            if (i+1) % 50 == 0:
                print("Compiled", i+1, "/", len(evaluator.terms), "elapsed", time.perf_counter()-started, flush=True)
                save()
        report["timings"]["local_value_preparation_seconds"] = time.perf_counter()-start
        root = load_caching_nll()
        with open_workspace("inputs/workspace_MX2000.root") as workspace:
            assert configure_hhvv(workspace) == source["configuration"]
            model, data = workspace.obj("ModelConfig"), workspace.data("data_obs")
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(), data, model.GetNuisanceParameters())
            reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint(); reference.clearZeroPoint()
            float(reference.getVal())
            references = []
            start = time.perf_counter()
            split = data.split(model.GetPdf(), True)
            subsets = {str(subset.GetName()): subset for subset in split}
            for channel in likelihood.channels:
                columns = root.RooArgSet(*(workspace.var(n) for n in channel.observables))
                reduced = subsets[channel.name].reduce(root.RooFit.SelectVars(columns))
                nll = root.cacheutils.CachingAddNLL("random_ref_"+channel.name, "", workspace.pdf(channel.pdf), reduced, True)
                nll.clearConstantZeroPoint(); nll.clearZeroPoint()
                references.append((channel, nll, reduced))
                guard()
            del nll, reduced
            report["timings"]["channel_reference_setup_seconds"] = time.perf_counter()-start
            term_indices = {term.name: i for i, term in enumerate(evaluator.terms)}
            rng = np.random.default_rng(report["seed"])
            start = time.perf_counter()
            for attempt in range(251):
                theta = initial.copy() if attempt == 0 else np.clip(
                    initial+rng.uniform(-2, 2, len(initial))*scales, low+margin, high-margin)
                for variable, coordinate in zip(variables, theta, strict=True):
                    variable.setVal(float(coordinate))
                reference_error = None
                try:
                    expected = float(reference.getVal())
                except Exception:
                    expected = None
                    reference_error = traceback.format_exc()
                values = np.array([float(t.value(theta[t.indices])) for t in evaluator.terms])
                if expected is None or not np.isfinite(expected) or not np.isfinite(values).all():
                    if attempt == 0:
                        raise ValueError("Nominal point is invalid")
                    report["rejected"].append(dict(attempt=attempt, theta=theta.tolist(),
                        combine_nll=expected if expected is not None and np.isfinite(expected) else None,
                        reference_exception=reference_error,
                        nonfinite_jax_terms=[t.name for t,v in zip(evaluator.terms, values, strict=True) if not np.isfinite(v)],
                        reason="Outside the currently supported finite-PDF/profile domain; invalid-PDF penalty equivalence is not claimed"))
                    print("Rejected random draw", attempt, flush=True)
                    save()
                    continue
                actual = likelihood.offset+sum(values.tolist())
                channels = []
                for channel, ref_channel, held_data in references:
                    if channel.masked:
                        value = ref_value = 0.
                    else:
                        index = term_indices[channel.name]
                        term = evaluator.terms[index]
                        correction = float(corrections[channel.name](theta[term.indices])) if channel.name in corrections else 0.
                        value = float(values[index])-correction
                        ref_value = float(ref_channel.getVal())
                    if not np.isfinite(value) or not np.isfinite(ref_value):
                        raise ValueError("Nonfinite per-channel value at accepted full-model point: "+channel.name)
                    channels.append(dict(name=channel.name, jax=value, combine=ref_value, absolute_error=abs(value-ref_value)))
                case = dict(attempt=attempt, theta=theta.tolist(), jax_nll=actual, combine_nll=expected,
                            nll_absolute_error=abs(actual-expected), channels=channels,
                            max_channel_absolute_error=max(c["absolute_error"] for c in channels))
                report["cases"].append(case)
                print("Point", len(report["cases"])-1, "draw", attempt, "NLL error", case["nll_absolute_error"],
                      "channel error", case["max_channel_absolute_error"], flush=True)
                save()
                if case["nll_absolute_error"] >= 1e-6 or case["max_channel_absolute_error"] >= 1e-6:
                    raise AssertionError("HHVV random NLL agreement failed; inspect saved point and channels")
                if len(report["cases"]) == 26:
                    break
            report["timings"]["point_comparison_seconds"] = time.perf_counter()-start
            references.clear()
            del reference
        report.update(status="passed" if len(report["cases"]) == 26 else "insufficient_valid_points",
                      valid_random_points=len(report["cases"])-1,
                      max_nll_absolute_error=max(c["nll_absolute_error"] for c in report["cases"]),
                      max_channel_absolute_error=max(c["max_channel_absolute_error"] for c in report["cases"]))
        save()
        print(report["status"], report["valid_random_points"], report["max_nll_absolute_error"],
              report["max_channel_absolute_error"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
    except Exception:
        report.update(status="failed", exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
