"""Read-only supplemental GPU-load observations bound to one benchmark process."""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import traceback


def identity(pid):
    path = Path("/proc")/str(pid)
    try:
        fields = (path/"stat").read_text().split(") ", 1)[1].split()
        return dict(pid=pid, start_ticks=fields[19], state=fields[0],
                    cmdline=(path/"cmdline").read_bytes().replace(b"\0", b" ").decode())
    except FileNotFoundError:
        return None


def now():
    return datetime.now(timezone.utc).isoformat()


def query(fields, kind="gpu"):
    return subprocess.check_output(
        ["nvidia-smi", "--id=0", f"--query-{kind}={fields}",
         "--format=csv,noheader,nounits"], text=True, timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    parser.add_argument("--output", type=Path, required=True, help="Report stem under reports/")
    parser.add_argument("--interval", type=float, default=5.)
    parser.add_argument("--seconds", type=float, default=86400.)
    args = parser.parse_args()
    if args.interval < 1 or args.seconds <= 0:
        parser.error("Require interval >= 1 second and a positive observation duration")
    assert Path.cwd() == Path("/tmp/sqian/combine-jax") and socket.getfqdn() == "lxplus905.cern.ch"
    assert not args.output.is_absolute() and args.output.parent == Path("reports")
    output, log = args.output.with_suffix(".json"), args.output.with_suffix(".jsonl")
    if output.exists() or log.exists():
        raise FileExistsError("Preserve previous telemetry")
    original = identity(args.pid)
    assert original and original["state"] != "Z" and "scripts/gpu_benchmark.py" in original["cmdline"]
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    report = dict(status="running", started_utc=now(), host=socket.getfqdn(), target=original,
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="Supplemental global load and process-memory samples from watcher start only; utilization is device-wide, not attributed to individual PIDs. The benchmark's 20 Hz NVML measurement remains the primary peak-memory record.",
                  interval_seconds=args.interval, observation_limit_seconds=args.seconds,
                  device_metadata=query("uuid,name,driver_version,memory.total").strip(),
                  samples=0, utilization_min_percent=100., utilization_max_percent=0.,
                  sm_clock_min_mhz=None, sm_clock_max_mhz=0., temperature_max_celsius=0.,
                  sampled_device_peak_mib=0., sampled_target_peak_mib=0.,
                  samples_with_other_compute_pids=0, other_compute_pids=[], log=str(log))
    others = set()
    started = time.monotonic()

    def save():
        report.update(elapsed_seconds=time.monotonic()-started, other_compute_pids=sorted(others))
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    with output.open("x") as stream:
        stream.write(json.dumps(report, indent=2)+"\n")
    try:
        with log.open("x") as stream:
            while time.monotonic()-started < args.seconds:
                current = identity(args.pid)
                if current is None or current["state"] == "Z":
                    report["status"] = "target_terminated"
                    break
                if current["start_ticks"] != original["start_ticks"]:
                    report["status"] = "target_identity_changed"
                    break
                values = next(csv.reader(io.StringIO(query(
                    "utilization.gpu,clocks.current.sm,temperature.gpu,memory.used"))))
                utilization, clock, temperature, memory = map(float, values)
                assert 0 <= utilization <= 100 and clock >= 0 and temperature >= 0 and memory >= 0
                processes = []
                for row in csv.reader(io.StringIO(query("pid,used_gpu_memory", "compute-apps"))):
                    if row:
                        processes.append(dict(pid=int(row[0]), memory_mib=float(row[1])))
                other = {p["pid"] for p in processes if p["pid"] != args.pid}
                others.update(other)
                target = next((p["memory_mib"] for p in processes if p["pid"] == args.pid), 0.)
                record = dict(utc=now(), target_state=current["state"], utilization_percent=utilization,
                              sm_clock_mhz=clock, temperature_celsius=temperature, device_used_mib=memory,
                              target_memory_mib=target, compute_processes=processes)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                report["samples"] += 1
                report["last_sample"] = record
                report["utilization_min_percent"] = min(report["utilization_min_percent"], utilization)
                report["utilization_max_percent"] = max(report["utilization_max_percent"], utilization)
                report["sm_clock_min_mhz"] = (clock if report["sm_clock_min_mhz"] is None
                                               else min(report["sm_clock_min_mhz"], clock))
                report["sm_clock_max_mhz"] = max(report["sm_clock_max_mhz"], clock)
                report["temperature_max_celsius"] = max(report["temperature_max_celsius"], temperature)
                report["sampled_device_peak_mib"] = max(report["sampled_device_peak_mib"], memory)
                report["sampled_target_peak_mib"] = max(report["sampled_target_peak_mib"], target)
                report["samples_with_other_compute_pids"] += bool(other)
                save()
                time.sleep(min(args.interval, max(0., args.seconds-(time.monotonic()-started))))
            else:
                report["status"] = "observation_limit"
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        report["finished_utc"] = now()
        save()
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
