"""Sequential scoped-loading measurements on the pinned LCG node."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = [*sorted(Path("src/combine_jax").rglob("*.py")),
             *[Path("scripts")/name for name in (
                 "gpu_benchmark.py", "run_gpu_node.sh", "run_gpu_scoped_batch.py",
                 "compare_gpu_results.py", "gpu_gradient_evidence.py",
                 "hhvv_export_cache.py", "spooled_executable.py", "scoped_executable.py")],
             Path("build/gpu_exports/manifest_random.json")]
    return {str(p): digest(p) for p in paths}


def job_command(model, backend):
    return ["bash", "scripts/run_gpu_node.sh", "--model", model, "--backend", backend,
            "--constraint-batch-size", "32", "--release-stages", "--trim-compilation-cache",
            "--report-tag", "scoped", "--spool-hessian-executables", "--spool-value-gradient",
            "--scoped-loads", "--cpu-hessian-no-fusion", "--hessian-chunk-size", "4",
            "--reference-manifest", "build/gpu_exports/manifest_random.json"]


def check_preflight(phase, provenance):
    if Path.cwd() != Path("/tmp/sqian/combine-jax") or socket.getfqdn() != "lxplus905.cern.ch":
        raise RuntimeError("Run only in the authorized directory on lxplus905")
    for policy in ("spooled", "scoped"):
        for prior_phase in ("smoke", "hhvv"):
            path = Path(f"reports/gpu_{policy}_{prior_phase}_driver.json")
            if path.exists() and json.loads(path.read_text())["status"] == "running":
                raise RuntimeError(f"Benchmark driver still marked running: {path}")
    if phase == "hhvv":
        smoke_path = Path("reports/gpu_scoped_smoke_driver.json")
        comparison_path = Path("reports/gpu_comparison_scoped.json")
        smoke = json.loads(smoke_path.read_text())
        comparison = json.loads(comparison_path.read_text())
        assert smoke["status"] == "passed" and smoke["sources"] == provenance
        assert smoke["comparison_sha256"] == digest(comparison_path)
        assert len(smoke["jobs"]) == 4
        assert {(j["model"], j["backend"]) for j in smoke["jobs"]} == {
            (model, backend) for model in ("counting", "automcstats") for backend in ("gpu", "cpu")}
        for job in smoke["jobs"]:
            assert job["exit_code"] == 0 and job["benchmark_status"] == "passed"
            assert job["benchmark_sha256"] == digest(job["benchmark"])
            assert job["command"] == job_command(job["model"], job["backend"])
        assert all(comparison["models"][m]["status"] == "passed"
                   for m in ("counting", "automcstats"))
    models = ("counting", "automcstats") if phase == "smoke" else ("hhvv",)
    protected = [Path(f"reports/gpu_scoped_{phase}_driver.json")]
    if phase == "smoke":
        protected.append(Path("reports/gpu_comparison_scoped.json"))
        protected.append(Path("reports/gpu_scoped_smoke_comparison.log"))
    for model in models:
        for backend in ("gpu", "cpu"):
            protected += [Path(f"reports/gpu_{model}_{backend}_scoped.{ext}") for ext in ("json", "log")]
    for path in protected:
        if path.exists():
            raise FileExistsError(f"Preserve existing evidence: {path}")
    return models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "hhvv"))
    args = parser.parse_args()
    provenance = sources()
    models = check_preflight(args.phase, provenance)
    output = Path(f"reports/gpu_scoped_{args.phase}_driver.json")
    report = dict(status="running", phase=args.phase, pid=os.getpid(), started_utc=now(),
                  runner_sha256=digest(__file__), sources=provenance,
                  scope="Sequential child lifecycle and smoke comparison; full HHVV gradient and Hessian acceptance requires separate reference comparisons",
                  jobs=[])
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(output)

    child = None
    interrupted = None

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signal.Signals(signum).name
        if child is not None and child.poll() is None:
            child.terminate()

    previous = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for model in models:
            for backend in ("gpu", "cpu"):
                if interrupted:
                    break
                assert sources() == provenance, "Sources changed during measurement"
                command = job_command(model, backend)
                stem = f"reports/gpu_{model}_{backend}_scoped"
                job = dict(model=model, backend=backend, command=command, started_utc=now())
                report["jobs"].append(job)
                save()
                started = time.monotonic()
                with Path(stem+".log").open("x") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    job["pid"] = child.pid
                    save()
                    # A signal can arrive between the pre-launch check and Popen.
                    if interrupted and child.poll() is None:
                        child.terminate()
                    code = child.wait()
                child = None
                job.update(exit_code=code, finished_utc=now(), elapsed_seconds=time.monotonic()-started)
                benchmark = Path(stem+".json")
                if benchmark.exists():
                    result = json.loads(benchmark.read_text())
                    job.update(benchmark=str(benchmark), benchmark_sha256=digest(benchmark),
                               benchmark_status=result["status"],
                               completed_operations=result.get("completed_operations", []))
                save()
                print(model, backend, "exit", code, job.get("benchmark_status"), flush=True)
                assert sources() == provenance, "Sources changed during measurement"
                if not interrupted and args.phase == "smoke":
                    assert code == 0 and job.get("benchmark_status") == "passed", "Smoke child failed"
        if interrupted:
            report.update(status="interrupted", signal=interrupted)
        else:
            passed = all(j["exit_code"] == 0 and j.get("benchmark_status") == "passed"
                         for j in report["jobs"])
            if passed and args.phase == "smoke":
                command = [sys.executable, "scripts/compare_gpu_results.py",
                           "--report-tag", "scoped", "--allow-partial"]
                report["comparison_command"] = command
                with Path("reports/gpu_scoped_smoke_comparison.log").open("x") as log:
                    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                    if interrupted and child.poll() is None:
                        child.terminate()
                    report["comparison_exit_code"] = child.wait()
                child = None
                passed = report["comparison_exit_code"] == 0
                comparison = Path("reports/gpu_comparison_scoped.json")
                if comparison.exists():
                    report.update(comparison=str(comparison), comparison_sha256=digest(comparison))
                assert sources() == provenance, "Sources changed during comparison"
            report["status"] = "interrupted" if interrupted else "passed" if passed else "failed"
            if interrupted:
                report["signal"] = interrupted
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            report["cleanup_child_exit_code"] = child.wait()
        report["finished_utc"] = now()
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
