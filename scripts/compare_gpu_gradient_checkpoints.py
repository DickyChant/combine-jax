"""Compare completed CPU/GPU gradient stages without accepting a missing Hessian."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_gpu_results import verified_refinement
from gpu_gradient_evidence import random_validation


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def compare(cpu_path, gpu_path, refined_random=False):
    paths = (cpu_path, gpu_path)
    loaded = [read(p) for p in paths]
    cpu, gpu = [item[0] for item in loaded]
    for key in ("model", "source", "source_sha256", "benchmark_sha256", "host", "lcg_view",
                "jax_version", "jaxlib_version", "float64", "cpu_affinity", "cpu_model",
                "gpu_metadata", "constraint_batch_size", "release_stages", "trim_compilation_cache",
                "xla_flags", "effective_xla_options", "reference_manifest_sha256",
                "parameters", "channels", "terms"):
        assert cpu[key] == gpu[key], key
    for key in ("spool_hessian_executables", "spool_value_gradient", "hessian_chunk_size", "spooled_executable_sha256"):
        assert cpu.get(key) == gpu.get(key), key
    assert cpu["backend"] == "cpu" and gpu["backend"] == "gpu" and cpu["float64"]
    assert cpu["model"] in ("counting", "automcstats", "hhvv")
    arrays, evidence = [], {}
    accepted = True
    for path, (benchmark, checksum) in zip(paths, loaded, strict=True):
        assert all(op in benchmark["completed_operations"] for op in ("value", "value_and_grad"))
        assert hashlib.sha256(Path(benchmark["arrays"]).read_bytes()).hexdigest() == benchmark["arrays_sha256"]
        with np.load(benchmark["arrays"], allow_pickle=False) as source:
            data = {key: source[key] for key in ("parameter_names", "point", "value", "vg_value", "gradient",
                    "validation_points", "validation_labels", "validation_values", "validation_vg_values", "validation_gradients")}
        count = len(data["validation_points"])
        assert data["validation_points"].shape == data["validation_gradients"].shape == (count, benchmark["parameters"])
        assert data["gradient"].shape == data["point"].shape == (benchmark["parameters"],)
        assert len(data["validation_labels"]) == count == len(benchmark["reference_validation"])
        assert data["validation_values"].shape == data["validation_vg_values"].shape == (count,)
        for key, values in data.items():
            if key not in ("parameter_names", "validation_labels"):
                assert np.isfinite(values).all(), key
        assert float(data["value"]) == benchmark["nll"]
        assert float(data["vg_value"]) == benchmark["value_gradient_nll"]
        for operation in ("value", "value_and_grad"):
            timing = benchmark["timings"][operation]
            assert timing["median_warm_seconds"] == float(np.median(timing["warm_seconds"])) > 0
            if benchmark["release_stages"]:
                assert "release_seconds" in timing
        accepted &= all(not c.get("nonfinite_operations") and c["nll_absolute_error"] < 1e-6
                        and c["value_gradient_nll_absolute_error"] < 1e-6 for c in benchmark["reference_validation"])
        if benchmark["model"] == "hhvv":
            assert count == 26
            nominal = verified_refinement(path, benchmark)
            random = random_validation(path, benchmark, checksum, refined=refined_random)
            accepted &= nominal["status"] == nominal["directions_status"] == random["status"] == "passed"
            evidence[benchmark["backend"]] = dict(nominal=nominal, random=random)
        else:
            accepted &= all(c["gradient_max_relative_error"] < 1e-4 for c in benchmark["reference_validation"])
        arrays.append(data)
    c, g = arrays
    for key in ("parameter_names", "point", "validation_points", "validation_labels"):
        np.testing.assert_array_equal(c[key], g[key])
    if evidence:
        for group, keys in (("nominal", ("reference_sha256", "directions_reference_sha256")),
                            ("random", ("reference_sha256", "stable_reference_sha256", "refinement_documents_sha256"))):
            for key in keys:
                assert evidence["cpu"][group][key] == evidence["gpu"][group][key]
    errors = {}
    for key in ("value", "vg_value", "validation_values", "validation_vg_values", "gradient", "validation_gradients"):
        absolute = np.abs(c[key]-g[key])
        errors[key+"_max_absolute_error"] = float(absolute.max(initial=0.))
        if "gradient" in key:
            relative = absolute/np.maximum(abs(c[key]), 1e-8)
            errors[key+"_max_relative_error"] = float(relative.max(initial=0.))
            accepted &= errors[key+"_max_relative_error"] < 1e-4
        else:
            accepted &= errors[key+"_max_absolute_error"] < 1e-6
    return dict(status="passed" if accepted else "failed", model=cpu["model"],
                scope="Completed NLL and value+gradient at all captured points; no Hessian acceptance or full benchmark completion",
                refined_random_references=refined_random,
                host=cpu["host"], lcg_view=cpu["lcg_view"], jax_version=cpu["jax_version"],
                jaxlib_version=cpu["jaxlib_version"], gpu_metadata=gpu["gpu_metadata"],
                cpu_model=cpu["cpu_model"], float64=True, validation_points=len(c["validation_points"]),
                errors=errors, combine_validation=evidence,
                gradient_relative_denominator="max(abs(same-node CPU AD), 1e-8)",
                provenance={r["backend"]: dict(benchmark=str(p), benchmark_sha256=h,
                    arrays_sha256=r["arrays_sha256"], parent_status=r["status"])
                    for p, (r,h) in zip(paths, loaded, strict=True)},
                timings={op: {**{r["backend"]: {k:v for k,v in r["timings"][op].items() if k != "term_timings"}
                              for r in (cpu,gpu)},
                              "gpu_over_cpu_warm_ratio": gpu["timings"][op]["median_warm_seconds"]/cpu["timings"][op]["median_warm_seconds"]}
                         for op in ("value", "value_and_grad")})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu", type=Path)
    parser.add_argument("gpu", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refined-random-references", action="store_true",
                        help="Require the explicit 14-case follow-up policy for both HHVV artifacts")
    arguments = parser.parse_args()
    report = compare(arguments.cpu, arguments.gpu, refined_random=arguments.refined_random_references)
    arguments.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(arguments.output, report["status"], report["errors"])
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
