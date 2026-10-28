"""Measure ROOT codegen/Clad feasibility without claiming Combine NLL equivalence."""

import argparse
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback


def run_case(case, backend):
    from combine_jax.root_io import load_caching_nll, open_workspace
    from hhvv_setup import configure_hhvv
    import numpy as np

    started = time.perf_counter()
    root = load_caching_nll()
    assert root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/RooSimultaneousOpt.h"
      #include <typeinfo>
      namespace combine_jax_codegen_probe {
      std::string dynamicType(RooAbsReal const& value) { return typeid(value).name(); }
      RooAbsReal* nativeNLL(RooSimultaneousOpt& pdf, RooAbsData& data, RooLinkedList const& commands) {
        return pdf.createNLLPassthrough(data, commands).release();
      }
      }
    ''')
    report = dict(case=case, backend=backend, root_version=str(root.gROOT.GetVersion()),
                  host=platform.node(), scope="Native RooFit codegen feasibility; Combine CachingSimNLL offsets, masks and analytic bin profiling are not certified for this path",
                  status="running", stage="setup")
    output = Path(f"reports/clad_{case}_{backend}.json")

    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")

    def evaluate(pdf, data, extra_options=()):
        option = root.RooFit.EvalBackend.Codegen() if backend == "codegen" else root.RooFit.EvalBackend.CodegenNoGrad()
        expected = root.RooFit.EvalBackend.Value.Codegen if backend == "codegen" else root.RooFit.EvalBackend.Value.CodegenNoGrad
        report.update(effective_option_backend=str(option.name()), fallback_detected=bool(option.value() != expected))
        if case == "hhvv" and report["fallback_detected"]:
            report.update(status="backend_unavailable", stage="backend_selection",
                          reason="Requested codegen backend was replaced before NLL creation; no codegen timing claimed")
            return
        report["stage"] = "createNLL"
        save()
        start = time.perf_counter()
        if case == "hhvv":
            report["nll_route"] = "RooSimultaneousOpt.createNLLPassthrough, explicitly bypassing the Combine override"
            commands = root.RooLinkedList()
            held_options = [option, *extra_options]
            for item in held_options:
                commands.Add(item)
            nll = root.combine_jax_codegen_probe.nativeNLL(pdf, data, commands)
            root.SetOwnership(nll, True)
        else:
            nll = pdf.createNLL(data, option, *extra_options)
        report.update(create_nll_seconds=time.perf_counter()-start, nll_class=str(nll.ClassName()),
                      nll_dynamic_type=str(root.combine_jax_codegen_probe.dynamicType(nll)),
                      has_gradient=bool(nll.hasGradient()), has_hessian=bool(nll.hasHessian()), stage="evaluate")
        if backend == "codegen" and not nll.hasGradient():
            report.update(status="backend_unavailable", reason="Requested codegen did not provide a gradient; fallback output is not a Clad result")
            save()
            return
        save()
        start = time.perf_counter()
        value = float(nll.getVal())
        report.update(nll=value, value_seconds=time.perf_counter()-start)
        if case == "gaussian":
            expected = 4*.5*np.log(2*np.pi)+.5*sum((x-.2)**2 for x in (-1., .4, 1.2, 2.))
            report["analytic_nll_absolute_error"] = abs(value-expected)
            assert abs(value-expected) < 1e-10
            if nll.hasGradient():
                report["stage"] = "gradient"
                save()
                gradient = np.zeros(1, dtype=np.float64)
                start = time.perf_counter()
                nll.gradient(gradient)
                report.update(gradient=gradient.tolist(), gradient_seconds=time.perf_counter()-start,
                              analytic_gradient_absolute_error=abs(float(gradient[0])+1.8))
                assert report["analytic_gradient_absolute_error"] < 1e-10
        report.update(status="backend_fallback" if report["fallback_detected"] else "evaluated", stage="complete")

    try:
        report["root_features"] = str(root.gROOT.GetConfigFeatures())
        if case == "gaussian":
            x = root.RooRealVar("clad_x", "", 0., -10., 10.)
            mean = root.RooRealVar("clad_mean", "", .2, -5., 5.)
            sigma = root.RooConstVar("clad_sigma", "", 1.)
            pdf = root.RooGaussian("clad_gaussian", "", x, mean, sigma)
            variables = root.RooArgSet(x)
            data = root.RooDataSet("clad_data", "", variables)
            for value in (-1., .4, 1.2, 2.):
                x.setVal(value); data.add(variables)
            evaluate(pdf, data)
        else:
            manifest = json.loads(Path("build/gpu_exports/manifest.json").read_text())
            report.update(source=manifest["models"]["hhvv"]["source"], combine_version=manifest["combine_version"])
            with open_workspace("inputs/workspace_MX2000.root") as workspace:
                configure_hhvv(workspace)
                model = workspace.obj("ModelConfig")
                report["pdf_class"] = str(model.GetPdf().ClassName())
                evaluate(model.GetPdf(), workspace.data("data_obs"), (
                    root.RooFit.Extended(True), root.RooFit.Constrain(model.GetNuisanceParameters()),
                    root.RooFit.GlobalObservables(model.GetGlobalObservables())))
    except Exception:
        report.update(status="unavailable", exception=traceback.format_exc())
        print(report["exception"], flush=True)
    finally:
        save()
    print(output, report["status"], report["stage"], report["elapsed_seconds"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("gaussian", "hhvv"))
    parser.add_argument("--backend", choices=("codegen", "codegen_no_grad"))
    arguments = parser.parse_args()
    if arguments.case:
        assert arguments.backend
        run_case(arguments.case, arguments.backend)
        return
    report = dict(status="running", cases=[], per_case_timeout_seconds=180, per_case_rss_limit_mib=11500)
    for case in ("gaussian", "hhvv"):
        for backend in ("codegen", "codegen_no_grad"):
            log = Path(f"reports/clad_{case}_{backend}.log")
            start = time.perf_counter()
            reason = None
            peak_rss = 0.
            with log.open("w") as stream:
                process = subprocess.Popen([sys.executable, __file__, "--case", case, "--backend", backend],
                                           stdout=stream, stderr=subprocess.STDOUT)
                while process.poll() is None:
                    try:
                        status = Path(f"/proc/{process.pid}/status").read_text()
                        rss = next((int(line.split()[1])/1024 for line in status.splitlines() if line.startswith("VmRSS:")), 0.)
                        peak_rss = max(peak_rss, rss)
                    except FileNotFoundError:
                        pass
                    if time.perf_counter()-start > 180 or peak_rss > 11500:
                        reason = "timeout" if time.perf_counter()-start > 180 else "rss_limit"
                        process.terminate()
                        try:
                            process.wait(timeout=2)
                        except subprocess.TimeoutExpired:
                            process.kill(); process.wait()
                        break
                    time.sleep(.05)
            entry = dict(case=case, backend=backend, exit_code=process.returncode, termination_reason=reason,
                         wall_seconds=time.perf_counter()-start, sampled_peak_rss_mib=peak_rss, log=str(log))
            result = Path(f"reports/clad_{case}_{backend}.json")
            if result.exists():
                entry["result"] = json.loads(result.read_text())
                if reason:
                    entry["result"].update(status=reason, last_completed_snapshot_seconds=entry["result"]["elapsed_seconds"],
                                           elapsed_seconds=entry["wall_seconds"], sampled_peak_rss_mib=peak_rss)
                    result.write_text(json.dumps(entry["result"], indent=2, allow_nan=False)+"\n")
            report["cases"].append(entry)
            Path("reports/hhvv_codegen_probe.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            print(case, backend, entry["exit_code"], entry.get("result", {}).get("status"), entry["wall_seconds"], flush=True)
    report["status"] = "probe_complete"
    Path("reports/hhvv_codegen_probe.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
