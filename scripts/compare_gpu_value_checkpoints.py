"""Validate an NLL-only CPU/GPU comparison without accepting unfinished AD stages."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cpu",type=Path)
    parser.add_argument("gpu",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    arguments = parser.parse_args()
    paths = (arguments.cpu,arguments.gpu)
    cpu,gpu = [json.loads(p.read_text()) for p in paths]
    for key in ("source","source_sha256","benchmark_sha256","host","lcg_view","jax_version",
                "jaxlib_version","float64","cpu_affinity","constraint_batch_size","release_stages",
                "parameters","channels","terms"):
        assert cpu[key] == gpu[key], key
    assert cpu["backend"] == "cpu" and gpu["backend"] == "gpu" and cpu["float64"]
    assert cpu.get("trim_compilation_cache",False) == gpu.get("trim_compilation_cache",False)
    assert cpu.get("xla_flags") == gpu.get("xla_flags")
    assert cpu.get("effective_xla_options") == gpu.get("effective_xla_options")
    assert cpu.get("reference_manifest_sha256") == gpu.get("reference_manifest_sha256")
    for report in (cpu,gpu):
        assert "value" in report["completed_operations"]
        assert report["nominal_value_absolute_error"] < 1e-6
        assert all(c["nll_absolute_error"] < 1e-6 for c in report["reference_validation"])
        assert hashlib.sha256(Path(report["arrays"]).read_bytes()).hexdigest() == report["arrays_sha256"]
        timing = report["timings"]["value"]
        assert timing["median_warm_seconds"] == float(np.median(timing["warm_seconds"]))
    with np.load(cpu["arrays"],allow_pickle=False) as c, np.load(gpu["arrays"],allow_pickle=False) as g:
        np.testing.assert_array_equal(c["point"],g["point"])
        np.testing.assert_array_equal(c["parameter_names"],g["parameter_names"])
        assert np.isfinite(c["term_values"]).all() and np.isfinite(g["term_values"]).all()
        assert float(c["value"]) == cpu["nll"] and float(g["value"]) == gpu["nll"]
        labels = [[t["name"] for t in r["timings"]["value"]["term_timings"]] for r in (cpu,gpu)]
        assert labels[0] == labels[1]
        errors = np.abs(c["term_values"]-g["term_values"])
        difference = float(abs(c["value"]-g["value"]))
        assert difference < 1e-6 and errors.max(initial=0.) < 1e-6
        worst = int(np.argmax(errors))
        validation_errors = {}
        assert ("validation_values" in c.files) == ("validation_values" in g.files)
        if "validation_values" in c.files:
            np.testing.assert_array_equal(c["validation_points"],g["validation_points"])
            np.testing.assert_array_equal(c["validation_labels"],g["validation_labels"])
            assert np.isfinite(c["validation_values"]).all() and np.isfinite(g["validation_values"]).all()
            error = float(np.max(abs(c["validation_values"]-g["validation_values"])))
            assert error < 1e-6
            validation_errors.update(points=len(c["validation_points"]),max_nll_absolute_error=error)
        result = dict(status="passed",scope="Nominal NLL and local-term values only; no paired gradient or Hessian acceptance",
                      source=cpu["source"],host=cpu["host"],lcg_view=cpu["lcg_view"],
                      jax_version=cpu["jax_version"],jaxlib_version=cpu["jaxlib_version"],float64=True,
                      gpu_metadata=gpu["gpu_metadata"],cpu_model=cpu["cpu_model"],
                      cpu_nll=float(c["value"]),gpu_nll=float(g["value"]),nll_absolute_error=difference,
                      max_term_absolute_error=float(errors.max(initial=0.)),worst_term=labels[0][worst],
                      combine_nll_absolute_errors={k:r["nominal_value_absolute_error"] for k,r in (("cpu",cpu),("gpu",gpu))},
                      provenance={k:dict(benchmark=str(p),benchmark_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                         arrays_sha256=r["arrays_sha256"]) for k,p,r in zip(("cpu","gpu"),paths,(cpu,gpu),strict=True)},
                      timings={k:{n:v for n,v in r["timings"]["value"].items() if n != "term_timings"}
                               for k,r in (("cpu",cpu),("gpu",gpu))},
                      cpu_over_gpu_warm_ratio=cpu["timings"]["value"]["median_warm_seconds"]/gpu["timings"]["value"]["median_warm_seconds"],
                      gpu_over_cpu_warm_ratio=gpu["timings"]["value"]["median_warm_seconds"]/cpu["timings"]["value"]["median_warm_seconds"],
                      timing_scope="Five synchronous warm calls per backend on the same shared node; local kernels plus NumPy accumulation; no universal performance claim")
        if validation_errors:
            result.update(validation_values=validation_errors,scope="NLL at all saved validation points and nominal local-term values; no paired gradient or Hessian acceptance")
    arguments.output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(arguments.output,result["status"],"NLL error",difference,"term error",result["max_term_absolute_error"],
          "GPU/CPU warm ratio",result["gpu_over_cpu_warm_ratio"])


if __name__ == "__main__":
    main()
