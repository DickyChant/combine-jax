"""Audit device/process memory and other compute PIDs during a running benchmark."""

import argparse
import ctypes as C
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time

from gpu_benchmark import MemoryMonitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pid", type=int)
    arguments = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    nvml = C.CDLL("libnvidia-ml.so.1")
    check = MemoryMonitor.check
    check(nvml.nvmlInit_v2())
    handle = C.c_void_p()
    check(nvml.nvmlDeviceGetHandleByIndex_v2(C.c_uint(0), C.byref(handle)))
    command_path = Path(f"/proc/{arguments.pid}/cmdline")
    assert b"scripts/gpu_benchmark.py" in command_path.read_bytes()
    output = Path("reports/gpu_hhvv_gpu_activity.jsonl")
    summary = dict(pid=arguments.pid, sample_period_seconds=.2, samples=0,
                   started_at=datetime.now(timezone.utc).isoformat(),
                   scope="Independent NVML audit started after the NLL stage; covers the remaining GPU run",
                   device_peak_mib=0., target_process_peak_mib=0., other_compute_pids=[])
    others = set()
    with output.open("w") as stream:
        while command_path.exists():
            memory = MemoryMonitor.Memory()
            check(nvml.nvmlDeviceGetMemoryInfo(handle, C.byref(memory)))
            count = C.c_uint(128)
            processes = (MemoryMonitor.Process*128)()
            check(nvml.nvmlDeviceGetComputeRunningProcesses_v2(handle, C.byref(count), processes))
            entries = [dict(pid=p.pid, memory_mib=p.usedGpuMemory/2**20) for p in processes[:count.value]]
            target = next((p["memory_mib"] for p in entries if p["pid"] == arguments.pid), 0.)
            others.update(p["pid"] for p in entries if p["pid"] != arguments.pid)
            record = dict(utc=datetime.now(timezone.utc).isoformat(), device_used_mib=memory.used/2**20,
                          processes=entries)
            stream.write(json.dumps(record)+"\n"); stream.flush()
            summary["samples"] += 1
            summary["device_peak_mib"] = max(summary["device_peak_mib"], record["device_used_mib"])
            summary["target_process_peak_mib"] = max(summary["target_process_peak_mib"], target)
            time.sleep(.2)
    summary.update(finished_at=datetime.now(timezone.utc).isoformat(), other_compute_pids=sorted(others))
    Path("reports/gpu_hhvv_gpu_activity.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(summary)


if __name__ == "__main__":
    main()
