"""Join measured HHVV LCG stages without presenting them as one uninterrupted run."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_gpu_results import check_execution_policy, verified_refinement
from gpu_gradient_evidence import random_validation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def derivative_errors(cpu, gpu, gpu_hessian):
    size = len(cpu["point"])
    assert cpu["hessian"].shape == gpu_hessian["hessian"].shape == (size, size)
    assert cpu["gradient"].shape == gpu["gradient"].shape == (size,)
    for key in ("parameter_names", "point", "parameter_scales"):
        np.testing.assert_array_equal(cpu[key], gpu[key])
        np.testing.assert_array_equal(cpu[key], gpu_hessian[key])
    for key in ("validation_points", "validation_labels"):
        np.testing.assert_array_equal(cpu[key], gpu[key])
    for arrays, keys in ((cpu, ("value", "vg_value", "gradient", "hessian", "validation_values",
                              "validation_vg_values", "validation_gradients")),
                         (gpu, ("value", "vg_value", "gradient", "validation_values",
                               "validation_vg_values", "validation_gradients")),
                         (gpu_hessian, ("hessian",))):
        for key in keys:
            assert arrays[key].dtype == np.float64 and np.isfinite(arrays[key]).all()
    difference = abs(cpu["hessian"]-gpu_hessian["hessian"])
    scale = np.outer(cpu["parameter_scales"], cpu["parameter_scales"])
    assert cpu["validation_gradients"].shape == gpu["validation_gradients"].shape == (26, size)
    assert cpu["validation_values"].shape == gpu["validation_values"].shape == (26,)
    assert cpu["validation_vg_values"].shape == gpu["validation_vg_values"].shape == (26,)
    result = dict(
        nll_absolute_error=float(abs(cpu["value"]-gpu["value"])),
        value_gradient_nll_absolute_error=float(abs(cpu["vg_value"]-gpu["vg_value"])),
        gradient_max_absolute_error=float(np.max(abs(cpu["gradient"]-gpu["gradient"]))),
        gradient_max_relative_error=float(np.max(abs(cpu["gradient"]-gpu["gradient"])/np.maximum(abs(cpu["gradient"]), 1e-8))),
        hessian_max_absolute_error=float(difference.max()),
        hessian_max_scaled_error=float((difference*scale).max()),
        compared_hessian_entries=int(difference.size),
        validation_nll_max_absolute_error=float(np.max(abs(cpu["validation_values"]-gpu["validation_values"]))),
        validation_vg_nll_max_absolute_error=float(np.max(abs(cpu["validation_vg_values"]-gpu["validation_vg_values"]))),
        validation_gradient_max_relative_error=float(np.max(
            abs(cpu["validation_gradients"]-gpu["validation_gradients"])/np.maximum(abs(cpu["validation_gradients"]), 1e-8))))
    result["passed"] = all(result[k] < 1e-6 for k in (
        "nll_absolute_error", "value_gradient_nll_absolute_error", "hessian_max_scaled_error",
        "validation_nll_max_absolute_error", "validation_vg_nll_max_absolute_error")) and all(
            result[k] < 1e-4 for k in ("gradient_max_relative_error", "validation_gradient_max_relative_error"))
    return result


def compare(cpu_path, gpu_gradient_path, gpu_hessian_path, cpu_check_path, gpu_check_path):
    cpu, gpu, hessian = [json.loads(p.read_text()) for p in (cpu_path, gpu_gradient_path, gpu_hessian_path)]
    assert cpu["status"] == "passed" and hessian["status"] == "computed"
    assert gpu["status"] == "checkpoint" and gpu["checkpoint_stage"] == "gradient_all_points"
    assert set(cpu["completed_operations"]) == {"value", "value_and_grad", "hessian"}
    assert set(gpu["completed_operations"]) == {"value", "value_and_grad"}
    assert hessian["completed_operations"] == ["hessian"]
    check_execution_policy(cpu, gpu)
    for key in ("source", "source_sha256", "host", "workdir", "jax_version", "jaxlib_version",
                "lcg_view", "float64", "reference_manifest_sha256", "constraint_batch_size",
                "hessian_chunk_size", "cpu_affinity", "xla_flags", "gpu_metadata"):
        assert cpu[key] == gpu[key] == hessian[key], key
    for key in ("benchmark_sha256", "cpu_model", "effective_xla_options", "python"):
        assert cpu[key] == gpu[key], key
    assert cpu["benchmark_sha256"] == digest("scripts/gpu_benchmark.py")
    assert cpu["host"] == "lxplus905.cern.ch" and cpu["workdir"] == "/tmp/sqian/combine-jax"
    assert cpu["jax_version"] == cpu["jaxlib_version"] == "0.10.1"
    assert cpu["lcg_view"] == "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"
    assert cpu["float64"] and not cpu["root_imported"] and not hessian["root_imported"]
    assert cpu["parameters"] == gpu["parameters"] == hessian["parameters"] == 2656
    original = json.loads(Path(hessian["original_producer"]).read_text())
    assert digest(hessian["original_producer"]) == hessian["original_producer_sha256"]
    assert original["arrays_sha256"] == gpu["arrays_sha256"]
    for operation in ("value", "value_and_grad"):
        assert original["timings"][operation] == gpu["timings"][operation]
    checks = [json.loads(p.read_text()) for p in (cpu_check_path, gpu_check_path)]
    assert all(c["status"] == "passed" for c in checks)
    assert checks[0]["benchmark_report_sha256"] == digest(cpu_path)
    assert checks[1]["benchmark_sha256"] == digest(gpu_hessian_path)
    assert checks[0]["benchmark_arrays_sha256"] == cpu["arrays_sha256"]
    assert checks[1]["arrays_sha256"] == hessian["arrays_sha256"]
    assert checks[0]["local_arrays_sha256"] == checks[1]["local_arrays_sha256"]
    assert checks[0]["checker_sha256"] == digest("scripts/compare_local_cpu_hessian.py")
    assert checks[1]["checker_sha256"] == digest("scripts/compare_lcg_recovered_hessian.py")
    nominal, random = {}, {}
    for backend, path, report in (("cpu", cpu_path, cpu), ("gpu", gpu_gradient_path, gpu)):
        nominal[backend] = verified_refinement(path, report)
        random[backend] = random_validation(path, report, digest(path), refined=True)
    for key in ("reference_sha256", "directions_reference_sha256"):
        assert nominal["cpu"][key] == nominal["gpu"][key]
    for key in ("reference_sha256", "stable_reference_sha256", "refinement_documents_sha256"):
        assert random["cpu"][key] == random["gpu"][key]
    for report in (cpu, gpu, hessian):
        assert digest(report["arrays"]) == report["arrays_sha256"]
    with np.load(cpu["arrays"], allow_pickle=False) as c, \
            np.load(gpu["arrays"], allow_pickle=False) as g, \
            np.load(hessian["arrays"], allow_pickle=False) as h:
        errors = derivative_errors(c, g, h)
    references_pass = all(nominal[b]["status"] == nominal[b]["directions_status"] == random[b]["status"] == "passed"
                          for b in ("cpu", "gpu"))
    timings = {}
    for operation in ("value", "value_and_grad"):
        timings[operation] = {
            backend: {k: report["timings"][operation][k] for k in (
                "lowering_seconds", "compile_seconds", "total_preparation_seconds",
                "first_execution_seconds", "warm_seconds", "median_warm_seconds")}
            for backend, report in (("cpu", cpu), ("gpu", gpu))}
    timings["hessian"] = dict(
        cpu={k: cpu["timings"]["hessian"][k] for k in (
            "lowering_seconds", "compile_seconds", "total_preparation_seconds",
            "first_execution_seconds", "warm_seconds", "median_warm_seconds")},
        gpu={k: hessian[k] for k in (
            "recorded_compilation", "metadata_recovery_seconds",
            "first_execution_seconds", "warm_seconds", "median_warm_seconds")})
    for entry in timings.values():
        entry["cpu_over_gpu_median_ratio"] = entry["cpu"]["median_warm_seconds"]/entry["gpu"]["median_warm_seconds"]
    paths = [cpu_path, gpu_gradient_path, gpu_hessian_path, cpu_check_path, gpu_check_path]
    return dict(status="passed" if errors["passed"] and references_pass else "failed",
        reports_sha256={str(p): digest(p) for p in paths}, checker_sha256=digest(__file__),
        errors=errors, nominal_combine_validation=nominal, random_combine_validation=random,
        timings=timings, hessian_compiler_options={"cpu": cpu["hessian_compiler_options"], "gpu": {}},
        memory=dict(cpu_process_peak_rss_mib=cpu["peak_rss_mib"],
            original_gpu_process=dict(rss_mib=gpu["peak_rss_mib"], gpu_memory=gpu["gpu_memory"]),
            recovered_hessian_process=dict(rss_mib=hessian["peak_rss_mib"], gpu_memory=hessian["gpu_memory"])),
        runtime={key: cpu[key] for key in ("host", "workdir", "python", "lcg_view", "jax_version",
            "jaxlib_version", "gpu_metadata", "cpu_model", "float64", "cpu_affinity")},
        gradient_relative_denominator="max(abs(same-node CPU AD), 1e-8)",
        hessian_scaled_error="abs(GPU-CPU)*outer(nominal parameter scales, nominal parameter scales)",
        scope="Same-node measured stages with identical term decomposition and scoped loading "
        "lifetimes. GPU NLL/gradient come from the validated original process; GPU Hessian "
        "comes from the separately validated recovery process. CPU ran afterward. "
        "No single uninterrupted GPU total or isolated device-kernel throughput is implied. "
        "CPU Hessian uses the explicitly validated no-fusion options; GPU uses default options. "
        "Shared-node load and per-process memory scopes remain those of the source reports.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("cpu", "gpu-gradient", "gpu-hessian", "cpu-hessian-check", "gpu-hessian-check", "output"):
        parser.add_argument("--"+option, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = compare(args.cpu, args.gpu_gradient, args.gpu_hessian, args.cpu_hessian_check, args.gpu_hessian_check)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(result["status"], result["errors"], flush=True)
    raise SystemExit(0 if result["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
