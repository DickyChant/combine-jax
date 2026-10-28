"""Read-only shared GPU telemetry across preparation, GPU calls and CPU calls."""

import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback

from watch_gpu_load import identity, now, query


def matches(current, *, ticks=None, script=None):
    if current is None or current["state"] == "Z":
        return False
    if ticks is not None and int(current["start_ticks"]) != int(ticks):
        return False
    return script is None or script in current["cmdline"].split()


def own_live(pid, *, ticks=None, script=None):
    current = identity(pid)
    if not matches(current, ticks=ticks, script=script):
        return None
    try:
        if Path(f"/proc/{pid}").stat().st_uid != os.getuid():
            return None
        if Path(f"/proc/{pid}/cwd").resolve() != Path.cwd():
            return None
    except FileNotFoundError:
        return None
    return current


def split_compute(processes, known):
    """Classify only verified live benchmark identities as this experiment."""
    own = [p for p in processes if p["pid"] in known]
    other = [p for p in processes if p["pid"] not in known]
    return own, other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--start-ticks", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=5.)
    parser.add_argument("--seconds", type=float, default=86400.)
    args = parser.parse_args()
    if not 1 <= args.interval <= 60 or not 0 < args.seconds <= 86400:
        parser.error("Require interval 1–60 seconds and duration in (0,86400]")
    assert Path.cwd() == Path("/tmp/sqian/combine-jax") and socket.getfqdn() == "lxplus905.cern.ch"
    assert not args.output.is_absolute() and args.output.parent == Path("reports")
    output, log = args.output.with_suffix(".json"), args.output.with_suffix(".jsonl")
    if output.exists() or log.exists():
        raise FileExistsError("Preserve prior supplemental observations")
    driver_script = "scripts/run_lcg_recovery_completion.py"
    original = own_live(args.pid, ticks=args.start_ticks, script=driver_script)
    assert original, "Expected the verified live owned completion controller"
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    report = dict(status="running", started_utc=now(), host=socket.getfqdn(),
        driver=str(args.driver), driver_identity=original, watcher_pid=os.getpid(),
        python=sys.version, affinity=sorted(os.sched_getaffinity(0)),
        lcg_view=os.environ.get("COMBINE_JAX_LCG_VIEW"),
        numerical_libraries_imported={name: name in sys.modules for name in ("jax", "ROOT")},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        query_helper_sha256=hashlib.sha256(Path("scripts/watch_gpu_load.py").read_bytes()).hexdigest(),
        device_metadata=query("uuid,name,driver_version,memory.total").strip(),
        interval_seconds=args.interval, observation_limit_seconds=args.seconds,
        log=str(log), samples=0, samples_with_other_compute_pids=0,
        observed_benchmark_identities={}, other_compute_pids=[],
        sampled_benchmark_gpu_peak_mib_by_pid={}, sampled_device_peak_mib=0.,
        utilization_min_percent=None, utilization_max_percent=0.,
        sm_clock_min_mhz=None, sm_clock_max_mhz=0., temperature_max_celsius=0.,
        scope="Supplemental device-wide utilization/clocks/temperature and compute-PID "
        "memory from this watcher start only, spanning the remaining preparation and "
        "queued measurement stages. Only live owned PIDs with matching project/script "
        "identity are classified as this experiment. Device utilization is not assigned "
        "to individual PIDs. Each numerical process's 20 Hz NVML record remains primary.")
    started = time.monotonic()
    others = set()

    def save():
        report.update(elapsed_seconds=time.monotonic()-started, other_compute_pids=sorted(others))
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    save()
    try:
        with log.open("x") as stream:
            while time.monotonic()-started < args.seconds:
                current = own_live(args.pid, ticks=args.start_ticks, script=driver_script)
                if current is None:
                    report["status"] = "controller_terminated_or_identity_changed"
                    break
                driver = json.loads(args.driver.read_text())
                assert driver["pid"] == args.pid and driver["process_start_ticks"] == args.start_ticks
                known = {}
                preparation = json.loads(Path(driver["preparation"]).read_text())
                pid = preparation["pid"]
                found = own_live(pid, ticks=preparation["process_start_ticks"],
                                 script="scripts/prepare_lcg_hessian_continuation.py")
                if found:
                    known[pid] = dict(found, role="hessian_preparation")
                for job in driver["jobs"]:
                    if "pid" not in job:
                        continue
                    script = ("scripts/measure_lcg_recovered_hessian.py" if
                        "scripts/measure_lcg_recovered_hessian.py" in job["command"]
                        else "scripts/gpu_benchmark.py")
                    found = own_live(job["pid"], script=script)
                    if found:
                        known[job["pid"]] = dict(found, role="gpu_hessian" if
                            script.endswith("measure_lcg_recovered_hessian.py") else "cpu_benchmark")
                for pid, found in known.items():
                    old = report["observed_benchmark_identities"].get(str(pid))
                    if old is not None:
                        assert old["start_ticks"] == found["start_ticks"], "Observed benchmark PID was reused"
                    report["observed_benchmark_identities"][str(pid)] = found
                utilization, clock, temperature, memory = map(float, next(csv.reader(io.StringIO(
                    query("utilization.gpu,clocks.current.sm,temperature.gpu,memory.used")))))
                assert 0 <= utilization <= 100 and min(clock, temperature, memory) >= 0
                processes = [dict(pid=int(row[0]), memory_mib=float(row[1])) for row in
                    csv.reader(io.StringIO(query("pid,used_gpu_memory", "compute-apps"))) if row]
                ours, external = split_compute(processes, known)
                others.update(p["pid"] for p in external)
                record = dict(utc=now(), driver_status=driver["status"],
                    preparation_status=preparation["status"], preparation_stage=preparation["stage"],
                    active_benchmarks=known, utilization_percent=utilization, sm_clock_mhz=clock,
                    temperature_celsius=temperature, device_used_mib=memory,
                    benchmark_compute_processes=ours, other_compute_processes=external)
                stream.write(json.dumps(record, allow_nan=False)+"\n")
                stream.flush()
                report["samples"] += 1
                report["samples_with_other_compute_pids"] += bool(external)
                report["last_sample"] = record
                for process in ours:
                    key = str(process["pid"])
                    peaks = report["sampled_benchmark_gpu_peak_mib_by_pid"]
                    peaks[key] = max(peaks.get(key, 0.), process["memory_mib"])
                for key, value in (("utilization", utilization), ("sm_clock", clock)):
                    suffix = "percent" if key == "utilization" else "mhz"
                    low, high = f"{key}_min_{suffix}", f"{key}_max_{suffix}"
                    report[low] = value if report[low] is None else min(report[low], value)
                    report[high] = max(report[high], value)
                report["temperature_max_celsius"] = max(report["temperature_max_celsius"], temperature)
                report["sampled_device_peak_mib"] = max(report["sampled_device_peak_mib"], memory)
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
    print(report["status"], report["samples"], flush=True)


if __name__ == "__main__":
    main()
