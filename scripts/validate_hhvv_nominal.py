"""First full HHVV value gate; intentionally separate from derivative/fit gates."""

from importlib.metadata import version
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time
import traceback

import jax
import numpy as np

from combine_jax.likelihood import from_workspace
from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_setup import configure_hhvv
from hhvv_export_cache import load_export, save_export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cached-export", action="store_true", help="Reuse this benchmark's own local export after reviewing IR compatibility")
    arguments = parser.parse_args()
    path = Path("reports/hhvv_nominal.json")
    report = dict(workspace="inputs/workspace_MX2000.root", status="running",
                  combine_version=version("higgs-combine-tool"), jax_version=jax.__version__,
                  runtime_flags=REFERENCE_FLAGS, channels=[], timings={},
                  offset="clearConstantZeroPoint and clearZeroPoint",
                  scope="Nominal values only; no derivative, random-point or fit acceptance")
    start = time.perf_counter()

    def save():
        report["elapsed_seconds"] = time.perf_counter()-start
        report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    try:
        root = load_caching_nll()
        report["root_version"] = str(root.gROOT.GetVersion())
        with open_workspace(report["workspace"]) as workspace:
            report["configuration"] = configure_hhvv(workspace)
            report["timings"]["load_seconds"] = time.perf_counter()-start
            with Path(report["workspace"]).open("rb") as handle:
                source = dict(sha256=hashlib.file_digest(handle, "sha256").hexdigest(),
                              configuration=report["configuration"])
            cache_path = Path("build/hhvv_export.pkl")
            print("Loaded HHVV at MX=2000; exporting" if not arguments.cached_export else "Loading local export cache", flush=True)
            checkpoint = time.perf_counter()
            likelihood = load_export(cache_path, source) if arguments.cached_export else from_workspace(workspace)
            report["cached_export"] = arguments.cached_export
            report["timings"]["export_cache_load_seconds" if arguments.cached_export else "export_and_eager_check_seconds"] = time.perf_counter()-checkpoint
            if not arguments.cached_export:
                save_export(likelihood, cache_path, source)
            report.update(nodes=len(likelihood.nodes), parameters=list(likelihood.parameter_names),
                          initial=likelihood.initial.tolist(), channel_count=len(likelihood.channels),
                          profiled_parameters=[n for p in likelihood.profiles for n in p.parameters],
                          discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
                          rows=sum(len(c.data.weights) for c in likelihood.channels))
            print("Exported", len(likelihood.nodes), "nodes,", len(likelihood.parameters), "parameters", flush=True)
            save()
            model = workspace.obj("ModelConfig")
            data = workspace.data("data_obs")
            checkpoint = time.perf_counter()
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(), data, model.GetNuisanceParameters())
            if model.GetPdf().ClassName() == "RooSimultaneousOpt":
                reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint()
            reference.clearZeroPoint()
            report["combine_nll"] = float(reference.getVal())
            report["timings"]["combine_setup_and_value_seconds"] = time.perf_counter()-checkpoint
            print("Combine nominal", report["combine_nll"], flush=True)
            checkpoint = time.perf_counter()
            overrides = likelihood.profile_values(likelihood.initial)
            report["jax_constraints"] = float(likelihood.constraint_value(likelihood.initial, overrides))
            # ROOT performs the split. Select columns on each small subset, not
            # by reducing the entire combined dataset once for every channel.
            split = data.split(model.GetPdf(), True)
            subsets = {str(subset.GetName()): subset for subset in split}
            print("ROOT data split complete", flush=True)
            for i, channel in enumerate(likelihood.channels):
                subset = subsets.get(channel.name)
                if not subset:
                    raise ValueError(f"No ROOT data subset for {channel.name}")
                columns = root.RooArgSet(*(workspace.var(n) for n in channel.observables))
                reduced = subset.reduce(root.RooFit.SelectVars(columns))
                nll = root.cacheutils.CachingAddNLL("ref_"+channel.name, "", workspace.pdf(channel.pdf), reduced, True)
                nll.clearConstantZeroPoint()
                nll.clearZeroPoint()
                expected = 0. if channel.masked else float(nll.getVal())
                actual = float(likelihood.channel_value(channel, likelihood.initial, overrides))
                if not np.isfinite(actual) or not np.isfinite(expected):
                    raise ValueError(f"Nonfinite nominal NLL in {channel.name}: JAX {actual}, Combine {expected}")
                report["channels"].append(dict(name=channel.name, jax=actual, combine=expected,
                                                absolute_error=abs(actual-expected),
                                                jax_sum_weights=float(channel.data.weights.sum()),
                                                combine_sum_weights=float(reduced.sumEntries())))
                del nll, reduced
                if (i+1) % 25 == 0:
                    print("Channels", i+1, "max error", max(c["absolute_error"] for c in report["channels"]), flush=True)
                    save()
            report["timings"]["channel_comparison_seconds"] = time.perf_counter()-checkpoint
            report["jax_nll"] = sum(c["jax"] for c in report["channels"])+report["jax_constraints"]+likelihood.offset
            report["combine_constraints_by_subtraction"] = report["combine_nll"]-sum(c["combine"] for c in report["channels"])
            report["nll_absolute_error"] = abs(report["jax_nll"]-report["combine_nll"])
            report["max_channel_absolute_error"] = max(c["absolute_error"] for c in report["channels"])
            report["status"] = "passed" if report["nll_absolute_error"] < 1e-6 and report["max_channel_absolute_error"] < 1e-6 else "failed"
            save()
            print({k: report[k] for k in ("status", "nll_absolute_error", "max_channel_absolute_error", "elapsed_seconds", "peak_rss_mib")}, flush=True)
            if report["status"] != "passed":
                raise AssertionError("HHVV nominal NLL gate failed; see report")
    except Exception:
        report["status"] = "failed"
        report["exception"] = traceback.format_exc()
        save()
        raise


if __name__ == "__main__":
    main()
