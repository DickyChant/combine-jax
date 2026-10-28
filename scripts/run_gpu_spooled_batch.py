"""Run sequential pinned-node jobs and preserve child exit status separately."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("smoke", "hhvv"))
    args = parser.parse_args()
    if Path.cwd() != Path("/tmp/sqian/combine-jax") or socket.getfqdn() != "lxplus905.cern.ch":
        raise RuntimeError("Run only in the authorized directory on lxplus905")
    if args.phase == "hhvv":
        smoke = json.loads(Path("reports/gpu_spooled_smoke_driver.json").read_text())
        comparison = json.loads(Path("reports/gpu_comparison_spooled.json").read_text())
        if smoke["status"] != "passed" or any(
                comparison["models"].get(model, {}).get("status") != "passed"
                for model in ("counting", "automcstats")):
            raise RuntimeError("Complete and compare both CPU/GPU smoke models first")
    output = Path(f"reports/gpu_spooled_{args.phase}_driver.json")
    report = dict(status="running", phase=args.phase, started_utc=now(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Sequential child lifecycle; benchmark JSON retains its own numerical status",
                  jobs=[])
    # Exclusive creation also protects the child logs and previous run evidence.
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2)+"\n")
        temporary.replace(output)

    models = ("counting", "automcstats") if args.phase == "smoke" else ("hhvv",)
    for model in models:
        for backend in ("gpu", "cpu"):
            command = ["bash", "scripts/run_gpu_node.sh", "--model", model,
                       "--backend", backend, "--constraint-batch-size", "32",
                       "--release-stages", "--trim-compilation-cache",
                       "--report-tag", "spooled", "--spool-hessian-executables",
                       "--spool-value-gradient", "--hessian-chunk-size", "4",
                       "--reference-manifest", "build/gpu_exports/manifest_random.json"]
            stem = f"reports/gpu_{model}_{backend}_spooled"
            job = dict(model=model, backend=backend, command=command, started_utc=now())
            report["jobs"].append(job)
            save()
            started = time.monotonic()
            with Path(stem+".log").open("x") as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                job["pid"] = child.pid
                save()
                code = child.wait()
            job.update(exit_code=code, finished_utc=now(), elapsed_seconds=time.monotonic()-started)
            benchmark = Path(stem+".json")
            if benchmark.exists():
                payload = benchmark.read_bytes()
                result = json.loads(payload)
                job.update(benchmark=str(benchmark), benchmark_sha256=hashlib.sha256(payload).hexdigest(),
                           benchmark_status=result["status"], completed_operations=result.get("completed_operations", []))
            save()
            print(model, backend, "exit", code, job.get("benchmark_status"), flush=True)
            if args.phase == "smoke" and (code != 0 or job.get("benchmark_status") != "passed"):
                report.update(status="failed", finished_utc=now())
                save()
                raise SystemExit(1)
    passed = all(job["exit_code"] == 0 and job.get("benchmark_status") == "passed" for job in report["jobs"])
    report.update(status="passed" if passed else "failed", finished_utc=now())
    save()
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
