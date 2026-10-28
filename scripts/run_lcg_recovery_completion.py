"""Wait for owned preparation, measure GPU Hessian, then benchmark CPU sequentially."""

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


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def process_alive(pid, start_ticks):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except FileNotFoundError:
        return False
    return int(stat[19]) == start_ticks and stat[0] != "Z"


def sources():
    paths = [*sorted(Path("src/combine_jax").rglob("*.py")),
        *[Path("scripts")/n for n in (
            "run_lcg_recovery_completion.py", "measure_lcg_recovered_hessian.py",
            "trusted_lcg_recovery.py", "prepare_lcg_hessian_continuation.py",
            "hhvv_export_cache.py", "gpu_benchmark.py", "run_gpu_node.sh",
            "spooled_executable.py", "scoped_executable.py",
            "compare_local_cpu_hessian.py")]]
    return {str(p): digest(p) for p in paths}


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-checked-repeats", action="store_true",
        help="Use fresh GPU output names and validate every saved Hessian against CPU")
    args = parser.parse_args()
    assert Path.cwd() == Path("/tmp/sqian/combine-jax")
    assert socket.getfqdn() == "lxplus905.cern.ch"
    gpu = Path("reports/gpu_hhvv_gpu_recovered_hessian"+
        ("_reference_checked" if args.reference_checked_repeats else "")+".json")
    cpu = Path("reports/gpu_hhvv_cpu_scoped.json")
    gate = Path("reports/gpu_hhvv_recovered_hessian"+
        ("_reference_checked" if args.reference_checked_repeats else "")+"_array_gate.json")
    for path in [args.output, gpu, cpu, gate, gpu.with_suffix(".log"), cpu.with_suffix(".log")]:
        if path.exists():
            raise FileExistsError(f"Preserve prior evidence: {path}")
    initial = json.loads(args.preparation.read_text())
    assert initial["pid"] == args.pid and initial["process_start_ticks"] == args.start_ticks
    provenance = sources()
    report = dict(status="waiting", started_utc=now(), pid=os.getpid(),
        process_start_ticks=int(Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]),
        preparation=str(args.preparation), preparation_pid=args.pid,
        preparation_start_ticks=args.start_ticks, sources=provenance, jobs=[],
        scope="Sequential lifecycle only. Complete saved CPU-reference matrix gate "
        "before starting CPU measurements. Full provenance, Combine derivatives "
        "and same-node comparisons require the separate validation tools.")
    child = None
    interrupted = None

    def save():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    def stop(signum, frame):
        nonlocal interrupted
        interrupted = signal.Signals(signum).name
        if child is not None and child.poll() is None:
            child.terminate()

    previous = {s: signal.signal(s, stop) for s in (signal.SIGTERM, signal.SIGINT)}
    save()

    def run(command, output):
        nonlocal child
        assert sources() == provenance
        if interrupted:
            raise InterruptedError(interrupted)
        job = dict(command=command, report=str(output), started_utc=now())
        report["jobs"].append(job)
        report["status"] = "running"
        save()
        start = time.perf_counter()
        with output.with_suffix(".log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            job["pid"] = child.pid
            save()
            if interrupted:
                child.terminate()
            code = child.wait()
        child = None
        job.update(exit_code=code, elapsed_seconds=time.perf_counter()-start, finished_utc=now())
        if output.exists():
            result = json.loads(output.read_text())
            job.update(status=result["status"], report_sha256=digest(output))
        save()
        assert sources() == provenance
        assert not interrupted and code == 0, f"Child ended: code={code}, signal={interrupted}"
        return result

    try:
        absent_count = 0
        started_waiting = time.perf_counter()
        while True:
            if interrupted:
                raise InterruptedError(interrupted)
            current = json.loads(args.preparation.read_text())
            assert current["pid"] == args.pid and current["process_start_ticks"] == args.start_ticks
            alive = process_alive(args.pid, args.start_ticks)
            report.update(last_poll_utc=now(), preparation_status=current["status"],
                preparation_stage=current["stage"], preparation_process_alive=alive,
                prepared_terms=len(current["recovered_terms"])+len(current["new_terms"]))
            save()
            if current["status"] == "prepared" and not alive:
                break
            if current["status"] != "running" and current["status"] != "prepared":
                raise RuntimeError(f"Preparation terminal state: {current['status']}")
            absent_count = 0 if alive else absent_count+1
            if absent_count >= 3:
                raise RuntimeError("Preparation process absent in three polls without a completed catalog")
            if time.perf_counter()-started_waiting > 25200:
                raise TimeoutError("Observation limit reached; producer is not terminated by this waiter")
            time.sleep(10)
        report["preparation_sha256"] = digest(args.preparation)
        command = [sys.executable, "scripts/measure_lcg_recovered_hessian.py",
            "--preparation", str(args.preparation), "--preparation-sha256", digest(args.preparation),
            "--output", str(gpu)]
        if args.reference_checked_repeats:
            command.append("--reference-checked-repeats")
        result = run(command, gpu)
        assert result["status"] == "computed" and len(result["calls"]) == 6
        import numpy as np
        from compare_local_cpu_hessian import load_arrays, matrix_errors
        plan = json.loads(Path(current["plan"]).read_text())
        local_path = Path(plan["local_reference"])
        assert digest(local_path) == plan["local_reference_sha256"]
        local = json.loads(local_path.read_text())
        assert local["arrays_sha256"] == plan["local_reference_arrays_sha256"]
        left, right = load_arrays(local), load_arrays(result)
        errors = matrix_errors(left, right)
        check = dict(status="passed" if errors["passed"] else "failed",
            benchmark=str(gpu), benchmark_sha256=digest(gpu),
            arrays_sha256=result["arrays_sha256"], local_reference=str(local_path),
            local_reference_sha256=digest(local_path), local_arrays_sha256=local["arrays_sha256"],
            checker_sha256=digest("scripts/compare_local_cpu_hessian.py"), errors=errors,
            scope="Complete matrix against the hash-bound saved local CPU array. "
            "Independent block reassembly and complete provenance audit run separately.")
        with gate.open("x") as stream:
            json.dump(check, stream, indent=2, allow_nan=False)
            stream.write("\n")
        report.update(gpu_array_gate=str(gate), gpu_array_gate_sha256=digest(gate))
        save()
        assert check["status"] == "passed", "Recovered GPU Hessian differs from the CPU reference"
        result = run(["bash", "scripts/run_gpu_node.sh", "--model", "hhvv", "--backend", "cpu",
            "--constraint-batch-size", "32", "--release-stages", "--trim-compilation-cache",
            "--report-tag", "scoped", "--spool-hessian-executables", "--spool-value-gradient",
            "--scoped-loads", "--cpu-hessian-no-fusion", "--hessian-chunk-size", "4",
            "--reference-manifest", "build/gpu_exports/manifest_random.json"], cpu)
        assert result["status"] == "passed"
        report["status"] = "measurements_completed"
    except BaseException:
        report.update(status="interrupted" if interrupted else "failed",
            exception=traceback.format_exc(), signal=interrupted)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            report["cleanup_child_exit_code"] = child.wait()
        report["finished_utc"] = now()
        save()
        for sig, handler in previous.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
