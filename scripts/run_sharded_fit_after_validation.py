"""Queue frozen-reference checks and a local HHVV fit after a gradient trial.

The gradient producer is observed only. A failed or incomplete gradient cannot
launch a fit. The fit remains a conditional minimum without AD covariance.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gradient-report", type=Path,
        default=Path("reports/gpu_local_hhvv_sharded_cpu_gradient_no_fusion.json"))
    parser.add_argument("--output", type=Path,
        default=Path("reports/gpu_local_hhvv_sharded_no_fusion_driver.json"))
    parser.add_argument("--fit-output", type=Path,
        default=Path("reports/gpu_local_hhvv_sharded_cpu_fit_no_fusion.json"))
    parser.add_argument("--initial-check", type=Path,
        default=Path("build/fit_starts/hhvv_strategy1_candidate.json"))
    parser.add_argument("--wait-seconds", type=float, default=21600.)
    args = parser.parse_args()
    if args.wait_seconds <= 0 or not args.wait_seconds < float("inf"):
        parser.error("Use a positive finite observation limit")
    if Path.cwd() == Path("/tmp/sqian/combine-jax"):
        raise RuntimeError("This continuation is local only")
    for path in (args.output, args.fit_output):
        if path.parent != Path("reports") or path.suffix != ".json":
            parser.error("Use reports/*.json outputs")
        if path.exists():
            raise FileExistsError(f"Preserve {path}")
    initial = json.loads(args.initial_check.read_bytes())
    assert initial["status"] == "candidate_checked"
    assert digest(initial["parent"]) == initial["parent_sha256"]
    paths = [Path(__file__), *Path("src/combine_jax").rglob("*.py"),
        *map(Path, ("scripts/sharded_cpu_evaluator.py", "scripts/measure_sharded_cpu_gradient.py",
            "scripts/fit_sharded_cpu.py", "scripts/fit_exported_likelihood.py",
            "scripts/hhvv_export_cache.py", "scripts/probe_cpu_gradient_compiler_options.py",
            "scripts/reconcile_gpu_gradient_references.py", "scripts/compare_gpu_random_gradients.py",
            "scripts/compare_gpu_results.py", "scripts/gpu_gradient_evidence.py",
            "scripts/gpu_random_refinements.py", "scripts/fd_reference.py",
            "build/gpu_exports/manifest_random.json", "build/gpu_exports/hhvv.pkl",
            "reports/gpu_combine_gradient_resolved.json", "reports/gpu_combine_directions_refined.json",
            "reports/gpu_combine_random_gradient_extended.json", "reports/gpu_random_stable_fd_corrected.json",
            "reports/gpu_random_gradient_failure_fd.json", "reports/gpu_random_intensity_fd_dense_local.json")),
        args.initial_check, Path(initial["parent"])]
    frozen = {str(path): digest(path) for path in paths}
    report = dict(status="waiting_for_gradient", started_utc=now(), gradient_report=str(args.gradient_report),
        initial_check=str(args.initial_check), fit_output=str(args.fit_output),
        observation_limit_seconds=args.wait_seconds, frozen_inputs=frozen, stages=[],
        scope="Read-only observation of the gradient producer, then sequential unchanged reference checks and an owned local minimum-only fit. Observation timeout does not terminate the gradient producer.")
    child = None
    started = time.monotonic()

    def save():
        report["updated_utc"] = now()
        report["elapsed_seconds"] = time.monotonic()-started
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    def verify_frozen():
        assert all(digest(path) == checksum for path, checksum in frozen.items()), "Frozen input changed"

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    save()
    try:
        while True:
            gradient = json.loads(args.gradient_report.read_bytes())
            report["producer_status"] = gradient["status"]
            report["producer_stage"] = gradient.get("stage")
            report["producer_elapsed_seconds"] = gradient.get("elapsed_seconds")
            if gradient["status"] not in ("running", "computed"):
                report.update(status="producer_failed", reason="No complete validated gradient; fit was not launched")
                save()
                return 1
            if gradient["status"] == "computed" and gradient.get("worker_cleanup"):
                assert all(w["exit_code"] == 0 and not w["signals"] for w in gradient["worker_cleanup"])
                break
            if time.monotonic()-started >= args.wait_seconds:
                report.update(status="observation_limit", reason="No fit launched; gradient producer unchanged")
                save()
                return 1
            save()
            time.sleep(20)
        verify_frozen()
        assert gradient["model"] == "hhvv" and gradient["float64"]
        assert gradient["gradient_compiler_options"] == {
            "xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}
        assert gradient["completed_operations"] == ["value_and_grad"]
        assert len(gradient["reference_validation"]) == 26
        assert all(c["nll_absolute_error"] < 1e-6 for c in gradient["reference_validation"])
        assert digest(gradient["arrays"]) == gradient["arrays_sha256"]
        assert gradient["runner_sha256"] == digest("scripts/measure_sharded_cpu_gradient.py")
        assert gradient["helper_sha256"] == digest("scripts/sharded_cpu_evaluator.py")
        report["gradient_report_sha256"] = digest(args.gradient_report)
        report["gradient_arrays_sha256"] = gradient["arrays_sha256"]
        nominal = args.gradient_report.with_name(args.gradient_report.stem+"_resolved_gradient.json")
        random = args.gradient_report.with_name(args.gradient_report.stem+"_combine_random_gradients_refined.json")
        stages = [
            ("nominal_and_directions", nominal, [sys.executable, "scripts/reconcile_gpu_gradient_references.py",
                str(args.gradient_report), "--directions-reference", "reports/gpu_combine_directions_refined.json"]),
            ("all_random_gradients", random, [sys.executable, "scripts/compare_gpu_random_gradients.py",
                str(args.gradient_report), "--reference", "reports/gpu_combine_random_gradient_extended.json",
                "--stable-reference", "reports/gpu_random_stable_fd_corrected.json",
                "--refined-full-nll-reference", "reports/gpu_random_gradient_failure_fd.json",
                "--refined-intensity-reference", "reports/gpu_random_intensity_fd_dense_local.json"]),
            ("minimum_fit", args.fit_output, [sys.executable, "scripts/fit_sharded_cpu.py", "--model", "hhvv",
                "--cpu-no-fusion", "--validated-gradient", str(args.gradient_report),
                "--initial-check", str(args.initial_check), "--output", str(args.fit_output),
                "--strategy", "0", "--max-calls", "50000", "--workers", "3",
                "--rss-limit-mib", "10500", "--wall-seconds", "21600"]),
        ]
        for name, output, command in stages:
            if output.exists() or output.with_suffix(".log").exists():
                raise FileExistsError(f"Preserve continuation evidence: {output}")
        for name, output, command in stages:
            verify_frozen()
            assert digest(args.gradient_report) == report["gradient_report_sha256"]
            record = dict(name=name, command=command, output=str(output), started_utc=now(), status="running")
            report["stages"].append(record)
            report["status"] = name
            with output.with_suffix(".log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                record["pid"] = child.pid
                save()
                code = child.wait()
                child = None
            record.update(return_code=code, finished_utc=now())
            if output.exists():
                result = json.loads(output.read_bytes())
                record.update(result_status=result["status"], output_sha256=digest(output))
            expected = "minimum_computed" if name == "minimum_fit" else "passed"
            accepted = code == 0 and record.get("result_status") == expected
            record["status"] = "complete" if accepted else "failed"
            if not accepted:
                report["status"] = "stage_failed"
                save()
                return 1
            save()
        report["status"] = "minimum_computed"
        save()
        return 0
    except BaseException as error:
        report.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                      exception=traceback.format_exc())
        save()
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait(timeout=5)
            report["owned_child_cleanup"] = dict(pid=child.pid, return_code=child.returncode)
            save()


if __name__ == "__main__":
    raise SystemExit(main())
