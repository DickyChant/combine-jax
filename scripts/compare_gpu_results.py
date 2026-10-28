"""Compare full arrays and timings from sequential same-node CPU/GPU jobs."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from gpu_gradient_evidence import random_validation


def check_execution_policy(cpu, gpu):
    """Reject mixed lifetimes and unrecorded backend-specific compiler changes."""
    for key in ("spool_hessian_executables", "spool_value_gradient", "hessian_chunk_size",
                "spooled_executable_sha256", "scoped_loads", "scoped_executable_sha256",
                "cpu_hessian_no_fusion"):
        assert cpu.get(key) == gpu.get(key), key
    if cpu.get("scoped_loads"):
        assert cpu.get("scoped_executable_sha256"), "scoped_executable_sha256"
        assert cpu.get("spool_hessian_executables") and cpu.get("spool_value_gradient")
        for backend, report in (("cpu", cpu), ("gpu", gpu)):
            expected = ({"xla_disable_hlo_passes": "fusion",
                         "xla_cpu_parallel_codegen_split_count": 1}
                        if backend == "cpu" and report.get("cpu_hessian_no_fusion") else {})
            assert report.get("hessian_compiler_options") == expected, "hessian_compiler_options"
            for operation, timing in report["timings"].items():
                for term in timing["term_timings"]:
                    if term.get("skipped_empty_hessian"):
                        assert operation == "hessian" and term["parameters"] == 0
                        assert term.get("compiler_options") == {}, "term compiler_options"
                        continue
                    assert term.get("compiler_options") == (
                        expected if operation == "hessian" else {}), "term compiler_options"
    else:
        assert not cpu.get("cpu_hessian_no_fusion"), "cpu_hessian_no_fusion"


def compare_combine_directions(cpu, gpu, source):
    path = Path("reports/gpu_combine_directions.json")
    reference = json.loads(path.read_text())
    assert reference["status"] == "reference_ready"
    assert reference["source"] == source
    np.testing.assert_array_equal(cpu["parameter_names"], reference["parameter_names"])
    np.testing.assert_array_equal(cpu["point"], reference["initial"])
    cases = []
    for case in reference["cases"]:
        direction = np.asarray(case["direction"])
        derivative = case["probes"][-1]["derivative"]
        checks = {}
        for backend, arrays in (("cpu", cpu), ("gpu", gpu)):
            ad = float(np.dot(arrays["gradient"], direction))
            error = abs(ad-derivative)
            checks[backend] = dict(derivative=ad, absolute_error=error,
                                   relative_error=error/max(abs(derivative), 1e-8))
        cases.append(dict(label=case["label"], combine_derivative=derivative,
                          combine_step=case["probes"][-1]["step"],
                          step_halving_difference=case["step_halving_difference"], **checks))
    maxima = {backend: max(c[backend]["relative_error"] for c in cases)
              for backend in ("cpu", "gpu")}
    return dict(status="passed" if max(maxima.values()) < 1e-4 else "failed",
                reference=str(path), reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope=reference["scope"], parameters_in_dense_probes=reference["parameters_in_dense_probes"],
                excluded_near_bounds=reference["excluded_near_bounds"],
                max_relative_errors=maxima, cases=cases)


def verified_refinement(benchmark_path, benchmark):
    path = benchmark_path.with_name(benchmark_path.stem+"_resolved_gradient.json")
    result = json.loads(path.read_text())
    assert result["benchmark_sha256"] == hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    assert result["arrays_sha256"] == benchmark["arrays_sha256"] and result["source"] == benchmark["source"]
    assert result["reference_sha256"] == hashlib.sha256(Path(result["reference"]).read_bytes()).hexdigest()
    directions = result["directions"]
    assert directions["reference_sha256"] == hashlib.sha256(Path(directions["reference"]).read_bytes()).hexdigest()
    return dict(status=result["status"], validation=str(path),
                validation_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                reference=result["reference"], reference_sha256=result["reference_sha256"],
                method_counts=result["method_counts"], max_relative_error=result["max_relative_error"],
                max_absolute_error=result["max_absolute_error"], excluded=result["excluded"],
                directions_status=directions["status"], directions_max_relative_error=directions["max_relative_error"],
                directions_reference=directions["reference"], directions_reference_sha256=directions["reference_sha256"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-partial", action="store_true",
                        help="Permit smoke-only reports before the full HHVV jobs finish")
    parser.add_argument("--report-tag", default="", choices=("", "staged", "cachetrim", "spooled", "scoped"))
    parser.add_argument("--refined-references", action="store_true",
                        help="Require frozen per-parameter and refined directional validation for both HHVV backends; retain original directional errors")
    parser.add_argument("--refined-random-references", action="store_true",
                        help="Also require all 25 random-point Combine gradient checks with the explicit follow-up references")
    arguments = parser.parse_args()
    if arguments.refined_random_references and not arguments.refined_references:
        parser.error("Random refinements also require --refined-references for nominal validation")
    suffix = "_"+arguments.report_tag if arguments.report_tag else ""
    reports = {}
    for model in ("counting", "automcstats", "hhvv"):
        paths = [Path("reports")/f"gpu_{model}_{backend}{suffix}.json" for backend in ("cpu", "gpu")]
        if not all(p.exists() for p in paths):
            if not arguments.allow_partial:
                reports[model] = dict(status="missing", missing=[str(p) for p in paths if not p.exists()])
            continue
        cpu, gpu = [json.loads(p.read_text()) for p in paths]
        if cpu["status"] != "passed" or gpu["status"] != "passed":
            reports[model] = dict(status="incomplete", cpu_status=cpu["status"], gpu_status=gpu["status"])
            continue
        assert cpu["source"] == gpu["source"] and cpu["host"] == gpu["host"]
        assert cpu["benchmark_sha256"] == gpu["benchmark_sha256"]
        for key in ("source_sha256", "constraint_batch_size", "release_stages"):
            assert cpu.get(key) == gpu.get(key), key
        assert cpu.get("trim_compilation_cache",False) == gpu.get("trim_compilation_cache",False)
        check_execution_policy(cpu, gpu)
        if arguments.report_tag == "scoped":
            assert cpu.get("scoped_loads"), "scoped_loads"
        assert cpu.get("xla_flags") == gpu.get("xla_flags")
        assert cpu.get("effective_xla_options") == gpu.get("effective_xla_options")
        assert cpu.get("reference_manifest_sha256") == gpu.get("reference_manifest_sha256")
        assert cpu["jax_version"] == gpu["jax_version"] and cpu["float64"] and gpu["float64"]
        assert cpu["jaxlib_version"] == gpu["jaxlib_version"] and cpu["lcg_view"] == gpu["lcg_view"]
        for report in (cpu, gpu):
            assert hashlib.sha256(Path(report["arrays"]).read_bytes()).hexdigest() == report["arrays_sha256"]
        directional_checks = None
        with np.load(cpu["arrays"], allow_pickle=False) as c, np.load(gpu["arrays"], allow_pickle=False) as g:
            np.testing.assert_array_equal(c["parameter_names"],g["parameter_names"])
            np.testing.assert_array_equal(c["point"],g["point"])
            gradients = np.abs(c["gradient"]-g["gradient"])
            hessians = np.abs(c["hessian"]-g["hessian"])
            scales = c["parameter_scales"]
            relative = gradients/np.maximum(np.abs(c["gradient"]),1e-8)
            worst = np.argsort(relative)[-10:][::-1]
            errors = dict(nll_absolute_error=float(abs(c["value"]-g["value"])),
                          value_gradient_nll_absolute_error=float(abs(c["vg_value"]-g["vg_value"])),
                          max_term_absolute_error=float(np.abs(c["term_values"]-g["term_values"]).max(initial=0.)),
                          gradient_max_absolute_error=float(gradients.max(initial=0.)),
                          gradient_max_relative_error=float(relative.max(initial=0.)),
                          hessian_max_absolute_error=float(hessians.max(initial=0.)),
                          hessian_max_relative_error=float((hessians/np.maximum(np.abs(c["hessian"]),1e-8)).max(initial=0.)),
                          hessian_max_scaled_error=float((hessians*np.outer(scales,scales)).max(initial=0.)),
                          worst_gradients=[dict(name=str(c["parameter_names"][i]),cpu=float(c["gradient"][i]),gpu=float(g["gradient"][i]),relative_error=float(relative[i])) for i in worst])
            if model == "hhvv":
                directional_checks = compare_combine_directions(c, g, cpu["source"])
            assert ("validation_gradients" in c.files) == ("validation_gradients" in g.files)
            if "validation_gradients" in c.files:
                np.testing.assert_array_equal(c["validation_points"],g["validation_points"])
                np.testing.assert_array_equal(c["validation_labels"],g["validation_labels"])
                for key in ("validation_values","validation_vg_values","validation_gradients"):
                    assert np.isfinite(c[key]).all() and np.isfinite(g[key]).all()
                errors.update(validation_points=len(c["validation_points"]),
                              validation_nll_max_absolute_error=float(np.max(abs(c["validation_values"]-g["validation_values"]))),
                              validation_vg_nll_max_absolute_error=float(np.max(abs(c["validation_vg_values"]-g["validation_vg_values"]))),
                              validation_gradient_max_relative_error=float(np.max(abs(c["validation_gradients"]-g["validation_gradients"])/np.maximum(abs(c["validation_gradients"]),1e-8))))
        passed = errors["nll_absolute_error"] < 1e-6 and errors["value_gradient_nll_absolute_error"] < 1e-6 and errors["gradient_max_relative_error"] < 1e-4 and errors["hessian_max_scaled_error"] < 1e-6
        passed &= errors.get("validation_nll_max_absolute_error",0.) < 1e-6 and errors.get("validation_vg_nll_max_absolute_error",0.) < 1e-6 and errors.get("validation_gradient_max_relative_error",0.) < 1e-4
        refinements = None
        if directional_checks is not None and arguments.refined_references:
            refinements = {backend: verified_refinement(path, benchmark)
                           for backend,path,benchmark in zip(("cpu","gpu"),paths,(cpu,gpu),strict=True)}
            assert refinements["cpu"]["reference_sha256"] == refinements["gpu"]["reference_sha256"]
            assert refinements["cpu"]["directions_reference_sha256"] == refinements["gpu"]["directions_reference_sha256"]
            passed &= all(r["status"] == r["directions_status"] == "passed" and r["max_relative_error"] < 1e-4
                          and r["directions_max_relative_error"] < 1e-4 for r in refinements.values())
        elif directional_checks is not None:
            passed &= directional_checks["status"] == "passed"
        random_checks = None
        if model == "hhvv" and arguments.refined_random_references:
            random_checks = {backend: random_validation(path, benchmark,
                hashlib.sha256(path.read_bytes()).hexdigest(), refined=True)
                for backend,path,benchmark in zip(("cpu","gpu"),paths,(cpu,gpu),strict=True)}
            for key in ("reference_sha256", "stable_reference_sha256", "refinement_documents_sha256"):
                assert random_checks["cpu"][key] == random_checks["gpu"][key]
            passed &= all(r["status"] == "passed" for r in random_checks.values())
        reports[model] = dict(status="passed" if passed else "failed", errors=errors,
                             cpu_reference_errors=cpu["reference_validation"], gpu_reference_errors=gpu["reference_validation"],
                             cpu_peak_rss_mib=cpu["peak_rss_mib"], gpu_peak_rss_mib=gpu["peak_rss_mib"],
                             gpu_peak_memory_mib=gpu["gpu_memory"]["process_peak_mib"],
                             gpu_device_peak_used_mib=gpu["gpu_memory"]["device_peak_used_mib"],
                             gpu_device_baseline_used_mib=gpu["gpu_memory"]["device_baseline_used_mib"],
                             gpu_memory_scope="Sampled NVML process and device-wide memory reported separately; device peak is not automatically attributed to this process",
                             timings={})
        if directional_checks is not None:
            reports[model]["combine_directional_validation_original" if arguments.refined_references else "combine_directional_validation"] = directional_checks
        if refinements is not None:
            reports[model]["combine_refined_gradient_validation"] = refinements
        if random_checks is not None:
            reports[model]["combine_refined_random_gradient_validation"] = random_checks
        if cpu.get("spool_hessian_executables"):
            reports[model].update(spool_hessian_executables=True,hessian_chunk_size=cpu.get("hessian_chunk_size"),
                                 spool_value_gradient=cpu.get("spool_value_gradient",False),
                                 hessian_execution=cpu["hessian_execution"])
            if cpu.get("spool_value_gradient"):
                reports[model]["value_gradient_execution"] = cpu["value_gradient_execution"]
        if cpu.get("scoped_loads"):
            reports[model].update(scoped_loads=True,
                scoped_executable_sha256=cpu["scoped_executable_sha256"],
                cpu_hessian_no_fusion=cpu["cpu_hessian_no_fusion"],
                hessian_compiler_options={backend: report["hessian_compiler_options"]
                    for backend, report in (("cpu", cpu), ("gpu", gpu))},
                gradient_validation_execution=cpu["gradient_validation_execution"])
        for operation in ("value", "value_and_grad", "hessian"):
            c, g = cpu["timings"][operation], gpu["timings"][operation]
            reports[model]["timings"][operation] = dict(cpu_lowering_seconds=c["lowering_seconds"],gpu_lowering_seconds=g["lowering_seconds"],
                cpu_compile_seconds=c["compile_seconds"],gpu_compile_seconds=g["compile_seconds"],
                cpu_cache_trim_seconds=c.get("cache_trim_seconds",0.),gpu_cache_trim_seconds=g.get("cache_trim_seconds",0.),
                cpu_total_preparation_seconds=c["total_preparation_seconds"],gpu_total_preparation_seconds=g["total_preparation_seconds"],
                cpu_first_execution_seconds=c["first_execution_seconds"],gpu_first_execution_seconds=g["first_execution_seconds"],
                cpu_median_warm_seconds=c["median_warm_seconds"],gpu_median_warm_seconds=g["median_warm_seconds"],
                cpu_over_gpu_ratio=c["median_warm_seconds"]/g["median_warm_seconds"])
            if "spooled_timed_calls" in c:
                reports[model]["timings"][operation].update(
                    cpu_spooled_timed_calls=c["spooled_timed_calls"],gpu_spooled_timed_calls=g["spooled_timed_calls"],
                    cpu_serialization_seconds=c["serialization_seconds"],gpu_serialization_seconds=g["serialization_seconds"],
                    cpu_executable_write_seconds=c["executable_write_seconds"],gpu_executable_write_seconds=g["executable_write_seconds"])
    output_suffix = suffix+("_refined" if arguments.refined_references else "")+("_random" if arguments.refined_random_references else "")
    Path(f"reports/gpu_comparison{output_suffix}.json").write_text(json.dumps(dict(models=reports,
        refined_references=arguments.refined_references,
        refined_random_references=arguments.refined_random_references,
        gradient_relative_denominator="max(abs(same-node CPU AD), 1e-8)",
        hessian_scaling="abs(GPU-CPU) * outer(nominal parameter scales, nominal parameter scales)",
        scope=("Whole-model local kernels and host accumulation; nominal and all 25 random HHVV gradients require the same documented Combine follow-up references on both backends; original failures retained" if arguments.refined_random_references else
               "Whole-model local kernels and host accumulation; nominal HHVV CPU/GPU arrays, frozen per-parameter Combine FD refinements and refined gradient projections; original directional errors retained" if arguments.refined_references
               else "Whole-model local kernels and host accumulation; nominal HHVV derivatives compared with same-node CPU AD and independent Combine gradient projections, not yet full per-parameter Combine FD acceptance")),indent=2,allow_nan=False)+"\n")
    print({name: r["status"] for name,r in reports.items()})
    assert all(r["status"] == "passed" for r in reports.values())


if __name__ == "__main__":
    main()
