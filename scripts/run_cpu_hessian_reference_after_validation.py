"""Start the full local reference only after the active compiler gate passes."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    output = Path("reports/gpu_local_hhvv_hessian_no_fusion_driver.json")
    gate_path = Path("reports/hhvv_cpu_no_fusion_hessian.json")
    benchmark = Path("reports/gpu_local_hhvv_hessian_no_fusion.json")
    log_path = benchmark.with_suffix(".log")
    if benchmark.exists() or log_path.exists():
        raise FileExistsError("Preserve prior full-Hessian run")
    files = [Path(__file__), Path("scripts/build_cpu_hessian_reference.py"),
             Path("scripts/validate_cpu_no_fusion_hessian.py"), Path("scripts/spooled_executable.py"),
             Path("scripts/hhvv_export_cache.py"), *sorted(Path("src/combine_jax").rglob("*.py"))]
    hashes = {str(p): digest(p) for p in files}
    report = dict(status="waiting_for_validation", started_utc=now(),
                  gate=str(gate_path), frozen_files_sha256=hashes,
                  scope="Local CPU reference lifecycle only; no GPU or fit acceptance",
                  wait_limit_seconds=3600,
                  command=[sys.executable, "scripts/build_cpu_hessian_reference.py",
                           "--model", "hhvv", "--rss-limit-mib", "3500", "--compile-cores", "1"])
    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    try:
        deadline = time.monotonic()+report["wait_limit_seconds"]
        while True:
            gate = json.loads(gate_path.read_text())
            if gate["status"] != "running":
                break
            if time.monotonic() >= deadline:
                raise TimeoutError("Compiler-validation gate did not finish within the queue's one-hour wait")
            time.sleep(5)
        report.update(gate_status=gate["status"], gate_sha256=digest(gate_path))
        assert gate["status"] == "passed", "Compiler-validation gate did not pass"
        assert hashes == {str(p): digest(p) for p in files}, "Queued numerical sources changed"
        report.update(status="running", child_started_utc=now())
        with log_path.open("x") as log:
            child = subprocess.Popen(report["command"], stdout=log, stderr=subprocess.STDOUT)
            report["pid"] = child.pid
            save()
            code = child.wait()
        report.update(exit_code=code, finished_utc=now())
        if benchmark.exists():
            result = json.loads(benchmark.read_text())
            report.update(benchmark=str(benchmark), benchmark_sha256=digest(benchmark),
                          benchmark_status=result["status"], completed_operations=result["completed_operations"])
        report["status"] = "completed" if code == 0 and report.get("benchmark_status") == "computed" else "failed"
        save()
        print(report["status"], report.get("benchmark_status"), flush=True)
        if report["status"] != "completed":
            raise SystemExit(1)
    except Exception:
        report.update(status="failed", finished_utc=now(), exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
