"""Reoptimize a frozen nuisance grid with either JAX gradients or native Combine."""

import argparse
from contextlib import ExitStack
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

import numpy as np

from profile_plan import MANIFEST, digest, load_plan


def check_overlap():
    if Path.cwd() == Path("/tmp/sqian/combine-jax") and platform.node() == "lxplus905.cern.ch":
        for policy in ("spooled", "scoped"):
            for phase in ("smoke", "hhvv"):
                path = Path(f"reports/gpu_{policy}_{phase}_driver.json")
                if path.exists() and json.loads(path.read_bytes())["status"] == "running":
                    raise RuntimeError(f"Finish the active benchmark before profiles: {path}")


def finite_number(value):
    return float(value) if np.isfinite(value) else None


def store_profile_point(artifact, report, record, rows, nlls, plan, lock, save):
    """Preserve the preceding complete artifact if interrupted during writing."""
    temporary = artifact.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, parameter_names=np.asarray(plan["parameter_names"]),
                        initial=np.asarray(plan["initial"]), values=np.asarray(rows),
                        canonical_nll=np.asarray(nlls),
                        fixed_values=np.asarray([p["value"] for p in plan["points"][:len(rows)]]))
    checksum = digest(temporary)
    with lock:
        temporary.replace(artifact)
        report["points"].append(record)
        report.update(arrays=str(artifact), arrays_sha256=checksum)
        save()


def native_objective(stack, likelihood, plan, manifest):
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    from hhvv_setup import configure_hhvv

    root = load_caching_nll()
    assert version("higgs-combine-tool") == plan["export_stack"]["combine_version"] == "11.0.0"
    assert str(root.gROOT.GetVersion()) == plan["export_stack"]["root_version"] == "6.40.00"
    if plan["model"] not in ("counting", "hhvv"):
        raise NotImplementedError(f"No native profile workspace path configured for {plan['model']}")
    workspace_path = Path("inputs/workspace_MX2000.root" if plan["model"] == "hhvv"
                          else "build/gpu_exports/counting.root")
    assert digest(workspace_path) == plan["source"]["sha256"]
    workspace = stack.enter_context(open_workspace(workspace_path))
    if plan["model"] == "hhvv":
        assert configure_hhvv(workspace) == plan["source"]["configuration"]
    for choice in plan["discrete_choices"].values():
        category = workspace.cat(choice["category"])
        category.setIndex(choice["index"])
        assert category.getIndex() == choice["index"]
    for config in plan["channel_masks"].values():
        mask = workspace.obj(config["name"])
        assert mask and str(mask.ClassName()) == config["class_name"]
        if mask.InheritsFrom("RooRealVar"):
            mask.setVal(config["value"])
        assert float(mask.getVal()) == config["value"]
    variables = [workspace.var(name) for name in plan["parameter_names"]]
    for variable, value in zip(variables, likelihood.initial, strict=True):
        variable.setVal(float(value))
    model = workspace.obj("ModelConfig")
    reference = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"),
                                            model.GetNuisanceParameters())
    if model.GetPdf().ClassName() == "RooSimultaneousOpt":
        reference.setChannelMasks(model.GetPdf().channelMasks())
    reference.setAnalyticBarlowBeeston(True)
    reference.clearConstantZeroPoint()
    reference.clearZeroPoint()
    nominal = float(reference.getVal())
    assert abs(nominal-manifest["models"][plan["model"]]["reference"]["cases"][0]["combine_nll"]) < 1e-6
    assert root.gInterpreter.Declare(r'''
      #include "RooRealVar.h"
      #include "RooAbsReal.h"
      #include <vector>
      namespace combine_jax_profile_reference {
      struct Objective {
        std::vector<RooRealVar*> variables;
        RooAbsReal* nll;
        unsigned long long calls=0;
        Objective(std::vector<RooRealVar*> v, RooAbsReal* n):variables(v),nll(n) {}
        double evaluate(std::vector<double> const& point) {
          if (point.size()!=variables.size()) throw std::invalid_argument("Parameter count mismatch");
          for (size_t i=0;i<point.size();++i)
            if (variables[i]->getVal()!=point[i]) variables[i]->setVal(point[i]);
          ++calls; return nll->getVal();
        }
      };
      }
    ''')
    objective = root.combine_jax_profile_reference.Objective(variables, reference)
    root.RooAbsReal.clearEvalErrorLog()
    baseline = float(objective.evaluate(plan["initial"]))
    assert int(root.RooAbsReal.numEvalErrors()) == 0
    assert abs(baseline-plan["canonical_baseline_nll"]) < 1e-6
    reference.setZeroPoint()
    assert abs(float(reference.getVal())) < 1e-6
    return root, reference, objective, baseline, dict(
        combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
        runtime_flags=REFERENCE_FLAGS, nominal_nll=nominal, baseline_nll=baseline)


def fit_native(root, reference, objective, baseline, likelihood, plan, fixed_value, args):
    from iminuit import Minuit

    initial = np.asarray(plan["initial"]).copy()
    initial[plan["parameter_index"]] = fixed_value
    root.RooAbsReal.clearEvalErrorLog()
    minuit = Minuit(lambda *v: float(objective.evaluate(v)), *initial, name=likelihood.parameter_names)
    minuit.errordef = .5
    minuit.tol = args.tolerance
    minuit.strategy = args.strategy
    for p in likelihood.parameters:
        if p.minimum == p.maximum:
            minuit.fixed[p.name] = True
            continue
        minuit.limits[p.name] = (p.minimum, p.maximum)
        step = p.error if p.error > 0 else .1*max(abs(p.value), 1.)
        minuit.errors[p.name] = min(step, .1*(p.maximum-p.minimum))
    minuit.fixed[plan["parameter"]] = True
    begin = time.perf_counter()
    minuit.migrad(ncall=args.max_calls, iterate=args.migrad_iterate,
                  use_simplex=not args.no_simplex)
    minimization_seconds = time.perf_counter()-begin
    values = np.asarray(minuit.values)
    exploration_errors = int(root.RooAbsReal.numEvalErrors())
    root.RooAbsReal.clearEvalErrorLog()
    zero = float(objective.evaluate(values))
    reference.clearZeroPoint()
    canonical = float(reference.getVal())
    final_errors = int(root.RooAbsReal.numEvalErrors())
    restoration = abs(canonical-(zero+baseline))
    record = dict(valid_minimum=bool(minuit.valid), edm=finite_number(minuit.fmin.edm),
                  function_calls=minuit.nfcn, gradient_calls=0,
                  native_exploration_evaluation_errors=exploration_errors,
                  native_final_evaluation_errors=final_errors, offset_restoration_error=finite_number(restoration),
                  timings=dict(minimization_seconds=minimization_seconds))
    record["valid_minimum"] &= final_errors == 0 and restoration < 1e-6
    # Restore exactly the same baseline zero point for the next independent fit.
    restored = float(objective.evaluate(plan["initial"]))
    assert abs(restored-baseline) < 1e-6
    reference.setZeroPoint()
    assert abs(float(reference.getVal())) < 1e-6
    return values, canonical, record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--engine", choices=("combine", "cpu", "gpu"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--strategy", type=int, choices=(0, 1, 2), default=0)
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--max-calls", type=int, default=3000000)
    parser.add_argument("--migrad-iterate", type=int, default=5,
                        help="MIGRAD passes; nondefault retry controls require Combine or bounded CPU mode")
    parser.add_argument("--no-simplex", action="store_true",
                        help="Disable MIGRAD's simplex fallback in Combine or bounded CPU mode")
    parser.add_argument("--wall-seconds", type=float, default=21600.)
    parser.add_argument("--rss-limit-mib", type=float, default=11000.)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--progress-seconds", type=float, default=20.)
    parser.add_argument("--bounded-minimum", type=Path,
                        help="Local CPU only: reuse bounded loading after verifying this completed minimum's gradient references")
    args = parser.parse_args()
    if (args.output.parent != Path("reports") or args.output.suffix != ".json"
            or args.max_calls < 1 or args.migrad_iterate < 1
            or not 0 < args.tolerance <= 1e-4 or not 1 <= args.cores <= 8
            or not 0 < args.rss_limit_mib <= 11000 or args.wall_seconds <= 0
            or args.progress_seconds <= 0
            or not np.isfinite([args.tolerance, args.rss_limit_mib, args.wall_seconds,
                                args.progress_seconds]).all()):
        parser.error("Use reports/*.json, positive limits, tolerance <=1e-4 and at most eight cores")
    if args.bounded_minimum and (args.engine != "cpu" or Path.cwd() == Path("/tmp/sqian/combine-jax")):
        parser.error("Bounded profile loading requires local CPU execution")
    if (args.engine != "combine" and not args.bounded_minimum
            and (args.migrad_iterate != 5 or args.no_simplex)):
        parser.error("Nondefault retry controls require Combine or bounded CPU mode")
    artifact = Path("build/profile_results")/(args.output.stem+".npz")
    if args.output.exists() or artifact.exists():
        raise FileExistsError("Preserve previous profile outputs")
    if args.bounded_minimum and any(p.exists() for p in (
            Path("build/profile_results")/(args.output.stem+"_executables"),
            Path("build/profile_results")/(args.output.stem+"_gradient_validation.npz"))):
        raise FileExistsError("Preserve previous bounded profile preparation")
    check_overlap()
    started, cpu_started = time.perf_counter(), time.process_time()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:args.cores])
    os.environ["JAX_PLATFORMS"] = "cuda" if args.engine == "gpu" else "cpu"
    os.environ["JAX_ENABLE_X64"] = "true"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.bounded_minimum:
        os.environ["JAX_USE_SIMPLIFIED_JAXPR_CONSTANTS"] = "false"
    import jax
    import jaxlib
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_enable_compilation_cache", False)
    devices = jax.devices()
    assert all(d.platform == ("gpu" if args.engine == "gpu" else "cpu") for d in devices)
    if args.engine == "gpu":
        assert Path.cwd() == Path("/tmp/sqian/combine-jax") and platform.node() == "lxplus905.cern.ch"
    from hhvv_export_cache import load_export
    provisional = json.loads(args.plan.read_bytes())
    manifest = json.loads(MANIFEST.read_bytes())
    model = manifest["models"][provisional["model"]]
    likelihood = load_export(Path("build/gpu_exports")/model["cache"], model["source"])
    plan = load_plan(args.plan, likelihood, model["source"], host=platform.node())
    report = dict(status="running", stage="prepare", engine=args.engine, model=plan["model"],
                  host=platform.node(), plan=str(args.plan), plan_sha256=digest(args.plan),
                  runner_sha256=digest(__file__), plan_helper_sha256=digest("scripts/profile_plan.py"),
                  core_source_sha256=hashlib.sha256(b"".join(
                      str(p).encode()+b"\0"+p.read_bytes()
                      for p in sorted(Path("src/combine_jax").rglob("*.py")))).hexdigest(),
                  source=plan["source"], float64=True, devices=[str(d) for d in devices],
                  jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
                  iminuit_version=version("iminuit"), cpu_affinity=sorted(os.sched_getaffinity(0)),
                  lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"),
                  optimizer=dict(errordef=.5, strategy=args.strategy, tolerance=args.tolerance,
                                 max_calls=args.max_calls, migrad_iterate=args.migrad_iterate,
                                 use_simplex=not args.no_simplex),
                  scope=plan["scope"], point_definition=plan["point_definition"],
                  parameter_names=plan["parameter_names"], parameter=plan["parameter"], poi=plan["poi"],
                  discrete_choices=plan["discrete_choices"], channel_masks=plan["channel_masks"],
                  baseline_nll=plan["canonical_baseline_nll"], initial=plan["initial"],
                  wall_limit_seconds=args.wall_seconds, rss_limit_mib=args.rss_limit_mib,
                  progress_seconds=args.progress_seconds, preflight_seconds=time.perf_counter()-started,
                  timing_scope="Elapsed from after CLI/overlap checks, including device initialization, export/plan validation, native or JAX setup, all profile fits, saved artifacts and ExitStack cleanup. Python startup and module imports before main are excluded.",
                  points=[])
    bounded_evaluator = None
    if args.bounded_minimum:
        report.update(memory_policy="resident_value_spooled_gradient",
            bounded_minimum=str(args.bounded_minimum),
            bounded_minimum_sha256=digest(args.bounded_minimum),
            xla_flags=os.environ.get("XLA_FLAGS"), simplified_jaxpr_constants=False,
            mapping_guard=int(Path("/proc/sys/vm/max_map_count").read_text())-4096,
            objective_policy="Each conditional Minuit objective is centered at its initial canonical NLL; reported NLLs are canonical and offset restoration is checked.")
    lock, stop = threading.RLock(), threading.Event()
    monitor = None
    if args.engine == "gpu":
        from gpu_benchmark import MemoryMonitor
        import subprocess
        report["gpu_metadata"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,uuid", "--format=csv,noheader"],
            text=True).strip()
        monitor = MemoryMonitor()
    rows, nlls = [], []
    artifact.parent.mkdir(parents=True, exist_ok=True)

    def save():
        with lock:
            report.update(elapsed_seconds=time.perf_counter()-started,
                          cpu_seconds=time.process_time()-cpu_started,
                          peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
            if monitor is not None:
                monitor.stage = report["stage"]
                report["gpu_memory"] = monitor.snapshot()
            if args.bounded_minimum:
                report["mapping_count"] = len(Path("/proc/self/maps").read_text().splitlines())
                if bounded_evaluator is not None:
                    report["bounded_evaluation_counters"] = bounded_evaluator.snapshot()
            temporary = args.output.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            temporary.replace(args.output)

    def watch():
        last = started
        while not stop.wait(min(.25, args.progress_seconds)):
            with lock:
                if report["status"] != "running":
                    return
                if (resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024 > args.rss_limit_mib
                        or time.perf_counter()-started > args.wall_seconds
                        or (args.bounded_minimum and len(Path("/proc/self/maps").read_text().splitlines())
                            >= report["mapping_guard"])):
                    report.update(status="resource_limit", exception="Profile RSS, mapping or wall guard exceeded")
                    save()
                    os._exit(86)
                if time.perf_counter()-last > args.progress_seconds:
                    save()
                    print(report["stage"], report["elapsed_seconds"], report["peak_rss_mib"], flush=True)
                    last = time.perf_counter()

    def publish(**fields):
        with lock:
            report.update(fields)
            save()

    save()
    watcher = threading.Thread(target=watch, daemon=True)
    watcher.start()
    try:
        with ExitStack() as stack:
            if args.engine == "combine":
                root, reference, objective, baseline, metadata = native_objective(stack, likelihood, plan, manifest)
                publish(native_reference=metadata, root_imported=True)
            elif args.bounded_minimum:
                from bounded_cpu_profile import prepare, fit_point
                bounded_evaluator, evidence = prepare(likelihood, plan, args.bounded_minimum, args.output, stack,
                    lambda event: publish(stage=f"{event['operation']} {event['index']+1}/{event['terms']} {event['name']}"))
                publish(bounded_gradient_preparation=evidence)
                baseline = bounded_evaluator.value(plan["initial"])
                assert "ROOT" not in sys.modules
                publish(root_imported=False)
            else:
                from combine_jax.fitting import fit
                evaluator = likelihood.local_evaluator(constraint_batch_size=32, hessian_chunk_size=4)
                preparation = evaluator.prepare("value_and_grad", plan["initial"], release_cache=True)
                publish(gradient_preparation=preparation)
                baseline, gradient = evaluator.value_and_grad(plan["initial"])
                assert np.isfinite(gradient).all()
                assert "ROOT" not in sys.modules
                publish(root_imported=False)
            baseline_error = abs(float(baseline)-plan["canonical_baseline_nll"])
            publish(baseline_nll_absolute_error=baseline_error)
            assert baseline_error < 1e-6
            for index, point in enumerate(plan["points"]):
                publish(stage=f"profile point {index+1}/{len(plan['points'])}")
                begin = time.perf_counter()
                if args.engine == "combine":
                    values, value, record = fit_native(
                        root, reference, objective, baseline, likelihood, plan, point["value"], args)
                elif args.bounded_minimum:
                    evidence = report["bounded_gradient_preparation"]
                    assert digest(args.bounded_minimum) == report["bounded_minimum_sha256"]
                    assert all(digest(p) == checksum for p, checksum in evidence["frozen_sources"].items())
                    values, value, record = fit_point(likelihood, bounded_evaluator, plan, point["value"], args,
                        on_evaluation=lambda kind, theta, nll: publish(
                            stage=f"profile point {index+1}/{len(plan['points'])} / load full gradient")
                        if kind == "gradient_start" else None)
                    record = {key: finite_number(value) if isinstance(value, (float, np.floating)) else value
                              for key, value in record.items()}
                else:
                    fitted = fit(likelihood, initial=plan["initial"], fixed={plan["parameter"]: point["value"]},
                                 evaluator=evaluator, strategy=args.strategy, tolerance=args.tolerance,
                                 max_calls=args.max_calls, covariance=False, memory_policy="release")
                    values, value = fitted.values, fitted.nll
                    record = dict(valid_minimum=fitted.valid_minimum, edm=finite_number(fitted.edm),
                                  function_calls=fitted.function_calls, gradient_calls=fitted.gradient_calls,
                                  timings=fitted.timings)
                record.update(index=index, **point, elapsed_seconds=time.perf_counter()-begin,
                              canonical_nll=finite_number(value),
                              delta_nll=finite_number(value-plan["canonical_baseline_nll"]),
                              poi_value=finite_number(values[plan["poi_index"]]),
                              poi_shift=finite_number(values[plan["poi_index"]]-plan["initial"][plan["poi_index"]]))
                record["valid_minimum"] &= bool(np.isfinite(values).all() and np.isfinite(value))
                assert values[plan["parameter_index"]] == point["value"]
                rows.append(values.copy())
                nlls.append(value)
                store_profile_point(artifact, report, record, rows, nlls, plan, lock, save)
                print(index, record["valid_minimum"], record["delta_nll"], record["poi_shift"], flush=True)
            if args.engine == "combine":
                # Destroy native objects while their workspace is still alive.
                del objective, reference
        if args.bounded_minimum:
            assert bounded_evaluator.closed
        publish(status="profile_computed" if all(p["valid_minimum"] for p in report["points"])
                else "profile_incomplete", stage="finished")
    except BaseException:
        publish(status="failed", exception=traceback.format_exc())
        raise
    finally:
        stop.set()
        watcher.join()
        if monitor is not None:
            monitor.close()
    raise SystemExit(0 if report["status"] == "profile_computed" else 1)


if __name__ == "__main__":
    main()
