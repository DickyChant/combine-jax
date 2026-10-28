"""Local CPU lifecycle check; mock NVIDIA telemetry, never GPU measurements."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import traceback
from unittest.mock import patch

import numpy as np

import gpu_benchmark


class NoGpuTelemetry:
    """Explicit test substitute on the CPU-only development machine."""

    def __init__(self):
        self.stage = "setup"
        self.error = None
        self.pid_seen = False
        self.mapping_peak = gpu_benchmark.process_mapping_count()

    def snapshot(self):
        return dict(scope="MOCK telemetry in a local CPU lifecycle check; no GPU measurement",
                    process_peak_mib=0.,process_seen=False,device_baseline_used_mib=0.,device_peak_used_mib=0.)

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-tag",choices=("cachetrim","spooled","scoped"),default="cachetrim")
    parser.add_argument("--hessian-chunk-size",type=int)
    parser.add_argument("--spool-value-gradient",action="store_true")
    parser.add_argument("--cpu-hessian-no-fusion",action="store_true")
    parser.add_argument("--output",type=Path,default=Path("reports/gpu_harness_capture_check.json"))
    options = parser.parse_args()
    if options.output.exists():
        raise FileExistsError("Preserve previous harness checks")
    original = Path.cwd()
    available = sorted(os.sched_getaffinity(0))
    os.sched_setaffinity(0,available[2:])  # Fit and FD reference use the first two cores.
    scratch = Path(tempfile.mkdtemp(prefix="gpu_harness_capture_",dir=original/"build/tmp"))
    exports = scratch/"build/gpu_exports"
    exports.mkdir(parents=True)
    (scratch/"src").symlink_to(original/"src",target_is_directory=True)
    shutil.copyfile(original/"build/gpu_exports/manifest_random.json",exports/"manifest_random.json")
    for model in ("counting","automcstats"):
        (exports/(model+".pkl")).symlink_to(original/"build/gpu_exports"/(model+".pkl"))
    report = dict(status="running",scratch=str(scratch),
                  scope="Local JAX CPU functional validation with mock NVIDIA telemetry; no isolated timing or GPU performance claim",
                  report_tag=options.report_tag,hessian_chunk_size=options.hessian_chunk_size,
                  spool_value_gradient=options.spool_value_gradient,
                  cpu_hessian_no_fusion=options.cpu_hessian_no_fusion,
                  harness_sha256=hashlib.sha256(Path(gpu_benchmark.__file__).read_bytes()).hexdigest(),
                  checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  models=[])
    original_check_output = gpu_benchmark.subprocess.check_output

    def check_output(command,**kwargs):
        if command[0] == "nvidia-smi":
            return "MOCK: local CPU lifecycle test, GPU not measured"
        return original_check_output(command,**kwargs)

    try:
        os.chdir(scratch)
        for model in ("counting","automcstats"):
            arguments = ["gpu_benchmark.py","--model",model,"--backend","cpu","--constraint-batch-size","32",
                         "--release-stages","--trim-compilation-cache","--report-tag",options.report_tag,
                         "--reference-manifest","build/gpu_exports/manifest_random.json"]
            if options.report_tag in ("spooled","scoped"):
                arguments += ["--spool-hessian-executables"]
            if options.report_tag == "scoped":
                arguments += ["--scoped-loads"]
            if options.cpu_hessian_no_fusion:
                arguments += ["--cpu-hessian-no-fusion"]
            if options.hessian_chunk_size is not None:
                arguments += ["--hessian-chunk-size",str(options.hessian_chunk_size)]
            if options.spool_value_gradient:
                arguments += ["--spool-value-gradient"]
            with patch.object(sys,"argv",arguments),patch.object(gpu_benchmark,"MemoryMonitor",NoGpuTelemetry),patch.object(gpu_benchmark.subprocess,"check_output",check_output):
                gpu_benchmark.main()
            measured = json.loads(Path(f"reports/gpu_{model}_cpu_{options.report_tag}.json").read_text())
            assert measured["status"] == "passed" and measured["trim_compilation_cache"]
            assert not measured["root_imported"]
            assert measured["effective_xla_options"]["backend_optimization_level"] == 3
            for operation, preparation in measured["preparation_timings"].items():
                assert preparation["completed_terms"] == len(measured["timings"][operation]["term_timings"])
            if options.report_tag in ("spooled","scoped"):
                assert measured["spool_hessian_executables"]
                assert measured["hessian_chunk_size"] == options.hessian_chunk_size
                assert measured["spool_value_gradient"] == options.spool_value_gradient
                for operation in (("value_and_grad","hessian") if options.spool_value_gradient else ("hessian",)):
                    timing = measured["timings"][operation]
                    chunk = options.hessian_chunk_size if operation == "hessian" else None
                    expected_calls = sum(int(np.ceil(t["parameters"]/chunk)) if chunk else 1
                                         for t in timing["term_timings"] if not t.get("skipped_empty_hessian"))
                    calls = timing["spooled_timed_calls"]
                    assert len(calls) == len(timing["warm_seconds"])+1 == 6
                    assert [c["total_seconds"] for c in calls] == [timing["first_execution_seconds"],*timing["warm_seconds"]]
                    for call in calls:
                        assert call["kernel_calls"] == expected_calls
                        assert sum(call[k] for k in ("read_and_verify_seconds","deserialize_seconds","execute_and_copy_seconds","cleanup_seconds")) <= call["total_seconds"]
                    assert timing["serialized_bytes"] > 0
                    if options.report_tag == "scoped":
                        assert measured["scoped_loads"]
                        sessions = timing["scoped_load_sessions"]
                        for term in timing["term_timings"]:
                            if term.get("skipped_empty_hessian"):
                                assert term["name"] not in sessions
                                continue
                            records = sessions[term["name"]]
                            if operation == "hessian":
                                columns = int(np.ceil(term["parameters"]/chunk)) if chunk else 1
                                assert len(records) == 8  # Six timed + two additional reference matrices.
                                assert all(r["calls"] == columns for r in records)
                                total_calls = 8*columns
                            else:
                                assert len(records) == 1 and records[0]["calls"] == 26
                                total_calls = 6+26  # Timed calls plus a fresh full reference batch.
                            assert all(r["exception_type"] is None for r in records)
                            assert timing["spooled_all_calls"][term["name"]]["count"] == total_calls
            if options.report_tag == "scoped":
                from compare_gpu_results import check_execution_policy
                # Validate CPU options against a synthetic GPU policy only; no device comparison.
                synthetic_gpu = json.loads(json.dumps(measured))
                synthetic_gpu["hessian_compiler_options"] = {}
                for timing in synthetic_gpu["timings"].values():
                    for term in timing["term_timings"]:
                        term["compiler_options"] = {}
                check_execution_policy(measured, synthetic_gpu)
            with np.load(measured["arrays"],allow_pickle=False) as arrays:
                count,size = arrays["validation_gradients"].shape
                assert count == 26 and size == len(arrays["parameter_names"])
                assert arrays["validation_points"].shape == (count,size)
                assert arrays["validation_values"].shape == arrays["validation_vg_values"].shape == (count,)
                np.testing.assert_array_equal(arrays["validation_gradients"][0],arrays["gradient"])
                np.testing.assert_array_equal(arrays["validation_values"][0],arrays["value"])
                assert np.isfinite(arrays["validation_gradients"]).all()
            report["models"].append(dict(model=model,reference_points=26,status="passed",
                max_combine_nll_absolute_error=max(c["nll_absolute_error"] for c in measured["reference_validation"]),
                max_combine_gradient_relative_error=max(c["gradient_max_relative_error"] for c in measured["reference_validation"]),
                hessian_reference_points=sum("hessian_max_scaled_error" in c for c in measured["reference_validation"]),
                max_reference_hessian_scaled_error=max(c["hessian_max_scaled_error"] for c in measured["reference_validation"] if "hessian_max_scaled_error" in c),
                benchmark=str(scratch/f"reports/gpu_{model}_cpu_{options.report_tag}.json"),
                spooled_timing_checks="passed" if options.report_tag in ("spooled","scoped") else "not_applicable",
                scoped_load_count_checks="passed" if options.report_tag == "scoped" else "not_applicable",
                arrays=str(scratch/measured["arrays"])))
        report["status"] = "passed"
    except Exception:
        report.update(status="failed",exception=traceback.format_exc())
        raise
    finally:
        os.chdir(original)
        options.output.write_text(json.dumps(report,indent=2)+"\n")
    print(report,flush=True)


if __name__ == "__main__":
    main()
