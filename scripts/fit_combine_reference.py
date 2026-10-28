"""Fit the certified observed-data model with iminuit and the v11 CachingSimNLL.

This is an independent local reference for the main fit gate. It does not run JAX
derivatives or alter the ongoing same-node GPU/CPU benchmark.
"""

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

from iminuit import Minuit
import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
from hhvv_export_cache import load_export
from hhvv_setup import configure_hhvv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("counting", "hhvv"), required=True)
    parser.add_argument("--max-calls", type=int, default=3000000)
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    parser.add_argument("--rss-limit-mib", type=float, default=11500.)
    parser.add_argument("--initial-check", type=Path,
                        help="Use a separately verified restart point; its NLL is checked again before fitting")
    parser.add_argument("--report-tag", choices=("", "restart", "restart_strategy1", "restart_jax",
                                               "restart_centered", "restart_centered_iterated",
                                               "restart_best_native", "restart_best_hesse1"), default="")
    parser.add_argument("--center-objective", action="store_true",
                        help="Subtract the checked starting NLL only inside the optimizer; "
                             "restore the original zero-point and canonical conventions in reports")
    parser.add_argument("--strategy", type=int, choices=(0,1,2), default=1)
    parser.add_argument("--hesse-strategy", type=int, choices=(0,1,2),
                        help="Override strategy only for explicit HESSE; omitted preserves "
                             "Minuit's strategy after MIGRAD")
    parser.add_argument("--migrad-iterate", type=int, default=5)
    parser.add_argument("--no-simplex", action="store_true")
    arguments = parser.parse_args()
    if (arguments.max_calls < 1 or arguments.wall_seconds <= 0 or arguments.migrad_iterate < 1
            or not 0 < arguments.rss_limit_mib <= 11500):
        parser.error("Limits must be positive")
    if bool(arguments.initial_check) != bool(arguments.report_tag):
        parser.error("A verified restart requires the separate restart report tag")
    if arguments.center_objective and not arguments.initial_check:
        parser.error("Objective centering requires a separately checked starting point")
    suffix = "_"+arguments.report_tag if arguments.report_tag else ""
    output = Path("reports")/(arguments.model+"_combine_fit"+suffix+".json")
    artifact = Path("build/fit_results")/(arguments.model+"_combine_fit"+suffix+".npz")
    if output.exists() or artifact.exists():
        raise FileExistsError("Preserve existing fit reports and arrays")
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    started = time.perf_counter()
    cpu_started = time.process_time()
    manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
    source = manifest["models"][arguments.model]["source"]
    likelihood = load_export(Path("build/gpu_exports")/(arguments.model+".pkl"), source)
    workspace_path = Path("inputs/workspace_MX2000.root" if arguments.model == "hhvv" else "build/gpu_exports/counting.root")
    with workspace_path.open("rb") as handle:
        assert hashlib.file_digest(handle,"sha256").hexdigest() == source["sha256"]
    root = load_caching_nll()
    report = dict(status="running", stage="setup", model=arguments.model, source=source,
                  host=platform.node(), python=platform.python_version(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                  iminuit_version=version("iminuit"), runtime_flags=REFERENCE_FLAGS,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  objective="Combine v11 CachingSimNLL, observed data, analytic Barlow-Beeston enabled; local numerical-gradient iminuit reference",
                  offset="Canonical NLL checked; channel zero points set at initial state for minimization; final canonical NLL restored and checked",
                  scope="Continuous reference fit at exported discrete choices/masks only; no discrete-envelope, JAX fit, uncertainty, profile or speedup acceptance",
                  discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
                  optimizer=dict(errordef=.5,tolerance=1e-4,strategy=arguments.strategy,max_calls=arguments.max_calls,
                                 migrad_iterate=arguments.migrad_iterate, use_simplex=not arguments.no_simplex,
                                 hesse_strategy_requested=arguments.hesse_strategy,
                                 hesse_strategy_used=None,
                                 objective_centered=arguments.center_objective,
                                 print_level=1 if arguments.center_objective else 0,
                                 wall_limit_seconds=arguments.wall_seconds,rss_limit_mib=arguments.rss_limit_mib),
                  parameter_names=list(likelihood.parameter_names), initial=likelihood.initial.tolist(),
                  profiled_parameters=[n for p in likelihood.profiles for n in p.parameters],
                  fixed_bounds=[p.name for p in likelihood.parameters if p.minimum == p.maximum], timings={})
    fit_initial = likelihood.initial.copy()
    checked = None
    if arguments.initial_check:
        raw = arguments.initial_check.read_bytes()
        checked = json.loads(raw)
        assert checked["status"] == "candidate_checked" and checked["source"] == source
        assert hashlib.sha256(Path(checked["parent"]).read_bytes()).hexdigest() == checked["parent_sha256"]
        for key in ("combine_version","root_version","runtime_flags","discrete_choices","channel_masks"):
            assert json.loads(json.dumps(report[key])) == checked[key], key
        np.testing.assert_array_equal(checked["parameter_names"], likelihood.parameter_names)
        fit_initial = np.asarray(checked["point"], dtype=float)
        assert fit_initial.shape == likelihood.initial.shape and np.isfinite(fit_initial).all()
        assert np.all(fit_initial >= [p.minimum for p in likelihood.parameters])
        assert np.all(fit_initial <= [p.maximum for p in likelihood.parameters])
        for index, parameter in enumerate(likelihood.parameters):
            if parameter.minimum == parameter.maximum:
                assert fit_initial[index] == likelihood.initial[index]
        report.update(initial_check=str(arguments.initial_check),initial_check_sha256=hashlib.sha256(raw).hexdigest(),
                      fit_initial=fit_initial.tolist())
    artifact.parent.mkdir(parents=True,exist_ok=True)
    objective_shift = 0.

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      cpu_seconds=time.process_time()-cpu_started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp = output.with_suffix(".json.tmp")
        temp.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
        temp.replace(output)

    def summarize(minuit):
        state = minuit.fmin
        fields = ("is_valid","has_accurate_covar","has_posdef_covar","has_made_posdef_covar",
                  "has_parameters_at_limit","is_above_max_edm","hesse_failed","has_reached_call_limit")
        result = {k:bool(getattr(state,k)) for k in fields}
        result.update(edm=float(state.edm), nll_zero_point=float(minuit.fval)+objective_shift,
                      optimizer_fval=float(minuit.fval), nfcn=minuit.nfcn,
                      values=list(minuit.values), errors=list(minuit.errors))
        return result

    try:
        assert root.gInterpreter.Declare(r'''
          #include "RooRealVar.h"
          #include "RooAbsReal.h"
          #include <vector>
          #include <cmath>
          #include <limits>
          namespace combine_jax_fit_reference {
          struct Objective {
            std::vector<RooRealVar*> variables;
            RooAbsReal* nll;
            unsigned long long calls=0;
            double best=std::numeric_limits<double>::infinity();
            std::vector<double> bestPoint;
            Objective(std::vector<RooRealVar*> v, RooAbsReal* n):variables(v),nll(n) {}
            double evaluate(std::vector<double> const& point) {
              if (point.size()!=variables.size()) throw std::invalid_argument("Parameter count mismatch");
              for (size_t i=0;i<point.size();++i)
                if (variables[i]->getVal()!=point[i]) variables[i]->setVal(point[i]);
              double value=nll->getVal(); ++calls;
              if (std::isfinite(value) && value<best) { best=value; bestPoint=point; }
              return value;
            }
          };
          }
        ''')
        with open_workspace(workspace_path) as workspace:
            if arguments.model == "hhvv":
                assert configure_hhvv(workspace) == source["configuration"]
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            for variable,coordinate in zip(variables,likelihood.initial,strict=True):
                variable.setVal(float(coordinate))
            model = workspace.obj("ModelConfig")
            reference = root.cacheutils.CachingSimNLL(model.GetPdf(),workspace.data("data_obs"),model.GetNuisanceParameters())
            if model.GetPdf().ClassName() == "RooSimultaneousOpt":
                reference.setChannelMasks(model.GetPdf().channelMasks())
            reference.setAnalyticBarlowBeeston(True)
            reference.clearConstantZeroPoint(); reference.clearZeroPoint()
            nominal = float(reference.getVal())
            report["canonical_initial_nll"] = nominal
            assert abs(nominal-manifest["models"][arguments.model]["reference"]["cases"][0]["combine_nll"]) < 1e-6
            reference.setZeroPoint()
            report["initial_zero_point_nll"] = float(reference.getVal())
            assert abs(report["initial_zero_point_nll"]) < 1e-6
            native = root.combine_jax_fit_reference.Objective(variables,reference)
            next_update = time.perf_counter()+20

            def objective(*point):
                nonlocal next_update
                value = float(native.evaluate(point))
                now = time.perf_counter()
                if now > next_update:
                    report.update(function_calls=int(native.calls), best_seen_zero_point_nll=float(native.best),
                                  best_seen_values=list(native.bestPoint))
                    save()
                    print(report["stage"],report["function_calls"],report["best_seen_zero_point_nll"],
                          report["elapsed_seconds"],report["peak_rss_mib"],flush=True)
                    if report["peak_rss_mib"] > arguments.rss_limit_mib:
                        raise MemoryError(f"{arguments.rss_limit_mib:,.0f} MiB RSS guard exceeded")
                    if now-started > arguments.wall_seconds:
                        raise TimeoutError("Reference fit reached its wall-time limit")
                    next_update = now+20
                return value-objective_shift

            if checked is not None:
                actual_start = objective(*fit_initial)
                report["restart_zero_point_absolute_error"] = abs(actual_start-checked["zero_point_candidate_nll"])
                report["restart_canonical_absolute_error"] = abs(actual_start+nominal-checked["canonical_candidate_nll"])
                assert max(report["restart_zero_point_absolute_error"],report["restart_canonical_absolute_error"]) < 1e-6
                if arguments.center_objective:
                    objective_shift = actual_start
                    report["centered_initial_nll"] = objective(*fit_initial)
                    assert abs(report["centered_initial_nll"]) < 1e-6
            report["optimizer"]["constant_objective_shift"] = objective_shift
            report["optimizer_offset_scope"] = (
                "Only the Minuit callable subtracts this fixed scalar. Native best-seen, "
                "final.nll_zero_point and canonical NLL fields retain the original "
                "nominal-point offset convention. No precision or tolerance change.")
            minuit = Minuit(objective,*fit_initial,name=likelihood.parameter_names)
            minuit.errordef=.5; minuit.tol=1e-4; minuit.strategy=arguments.strategy
            minuit.print_level=report["optimizer"]["print_level"]
            for parameter in likelihood.parameters:
                if parameter.minimum == parameter.maximum:
                    minuit.fixed[parameter.name] = True
                    continue
                minuit.limits[parameter.name] = (parameter.minimum,parameter.maximum)
                step = parameter.error if parameter.error > 0 else .1*max(abs(parameter.value),1.)
                minuit.errors[parameter.name] = min(step,.1*(parameter.maximum-parameter.minimum))
            report["timings"]["setup_seconds"] = time.perf_counter()-started
            report["stage"]="migrad"
            save()
            begin = time.perf_counter()
            minuit.migrad(ncall=arguments.max_calls,iterate=arguments.migrad_iterate,use_simplex=not arguments.no_simplex)
            report["timings"]["migrad_seconds"] = time.perf_counter()-begin
            report["migrad"] = summarize(minuit)
            save()
            if minuit.valid:
                if arguments.hesse_strategy is not None:
                    minuit.strategy = arguments.hesse_strategy
                report["optimizer"]["hesse_strategy_used"] = int(minuit.strategy.strategy)
                report["hesse_settings"] = {key: getattr(minuit.strategy, key) for key in (
                    "strategy", "hessian_ncycles", "hessian_step_tolerance",
                    "hessian_g2_tolerance", "hessian_gradient_ncycles")}
                report["stage"]="hesse"
                save()
                begin = time.perf_counter()
                minuit.hesse(ncall=arguments.max_calls)
                report["timings"]["hesse_seconds"] = time.perf_counter()-begin
            report["final"] = summarize(minuit)
            values = np.array(minuit.values)
            actual_zero = float(native.evaluate(values))
            report["optimizer_offset_restoration_error"] = abs(actual_zero-(minuit.fval+objective_shift))
            assert report["optimizer_offset_restoration_error"] < 1e-6
            reference.clearZeroPoint()
            canonical = float(reference.getVal())
            report.update(canonical_fitted_nll=canonical,
                          canonical_restoration_error=abs(canonical-(minuit.fval+objective_shift+nominal)))
            assert report["canonical_restoration_error"] < 1e-6
            arrays = dict(parameter_names=np.array(likelihood.parameter_names), values=values, errors=np.array(minuit.errors))
            if minuit.covariance is not None:
                arrays["covariance"] = np.array(minuit.covariance)
                assert np.isfinite(arrays["covariance"]).all()
            np.savez_compressed(artifact,**arrays)
            report.update(arrays=str(artifact),arrays_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                          status="reference_ready" if minuit.valid and minuit.accurate else "fit_incomplete",stage="finished")
            save()
            print(report["status"],report["final"]["nfcn"],report["final"]["edm"],report["timings"],flush=True)
            del native, reference, minuit
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
