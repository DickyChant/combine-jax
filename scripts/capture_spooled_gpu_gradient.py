"""Preserve HHVV gradient checkpoints while the pinned-node benchmark continues."""

from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import subprocess
import time
import traceback

import numpy as np


SSH = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
       "-o", "HostName=lxplus905.cern.ch", "-o", "HostKeyAlias=lxplus.cern.ch",
       "sqian@lxplus.cern.ch"]
REMOTE = "reports/gpu_hhvv_gpu_spooled.json"


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def fetch(relative):
    path = PurePosixPath(relative)
    if path.is_absolute() or ".." in path.parts or path.parts[0] not in ("reports", "build"):
        raise ValueError("Remote reads must stay in the authorized project")
    return subprocess.run(
        [*SSH, "cd /tmp/sqian/combine-jax && cat " + shlex.quote(str(path))],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=40).stdout


def check_report(report):
    assert report["model"] == "hhvv" and report["backend"] == "gpu"
    assert report["host"] == "lxplus905.cern.ch"
    assert report["workdir"] == "/tmp/sqian/combine-jax"
    assert report["float64"] and report["devices"] == ["cuda:0"]
    assert report["jax_version"] == report["jaxlib_version"] == "0.10.1"
    assert report["lcg_view"] == "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"
    assert report["spool_value_gradient"] and report["spool_hessian_executables"]
    assert report["hessian_chunk_size"] == 4 and report["constraint_batch_size"] == 32
    assert report["source_sha256"] == digest(b"".join(
        str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py"))))
    assert report["benchmark_sha256"] == digest(Path("scripts/gpu_benchmark.py").read_bytes())
    assert report["spooled_executable_sha256"] == digest(Path("scripts/spooled_executable.py").read_bytes())
    assert report["reference_manifest_sha256"] == digest(
        Path("build/gpu_exports/manifest_random.json").read_bytes())


def capture(raw, report, payload, stage):
    """Validate a complete artifact before creating immutable local evidence."""
    assert stage in ("timed", "all_points")
    assert digest(payload) == report["arrays_sha256"]
    timing = report["timings"]["value_and_grad"]
    assert len(timing["term_timings"]) == report["terms"] == 524
    assert len(timing["warm_seconds"]) == 5
    assert len(timing["spooled_timed_calls"]) == 6
    assert all(c["kernel_calls"] == 524 for c in timing["spooled_timed_calls"])
    assert [c["total_seconds"] for c in timing["spooled_timed_calls"]] == [
        timing["first_execution_seconds"], *timing["warm_seconds"]]
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        size = report["parameters"]
        assert size == 2656 and arrays["gradient"].shape == (size,)
        assert arrays["gradient"].dtype == np.float64
        assert arrays["vg_value"].shape == ()
        assert np.isfinite(arrays["gradient"]).all() and np.isfinite(arrays["vg_value"])
        np.testing.assert_array_equal(arrays["point"], arrays["validation_points"][0])
        if stage == "all_points":
            assert arrays["validation_gradients"].shape == (26, size)
            assert arrays["validation_vg_values"].shape == (26,)
            assert arrays["validation_gradients"].dtype == np.float64
            assert np.isfinite(arrays["validation_gradients"]).all()
            assert np.isfinite(arrays["validation_vg_values"]).all()
            np.testing.assert_array_equal(arrays["gradient"], arrays["validation_gradients"][0])
            np.testing.assert_array_equal(arrays["vg_value"], arrays["validation_vg_values"][0])
    stem = f"gpu_hhvv_gpu_spooled_gradient_{stage}_checkpoint"
    raw_path = Path("reports")/(stem+"_raw.json")
    output = Path("reports")/(stem+".json")
    artifact = Path("build/gpu_results")/(stem+".npz")
    for path in (raw_path, output, artifact):
        if path.exists():
            raise FileExistsError(f"Preserve prior checkpoint: {path}")
    captured = dict(report)
    captured.update(
        status="checkpoint", arrays=str(artifact),
        checkpoint_scope=("Completed gradient timings and finite saved arrays; independent Combine "
                          "gradient acceptance requires the separate frozen-reference comparisons"),
        checkpoint_stage=stage, checkpoint_parent_report=REMOTE,
        checkpoint_parent_report_sha256=digest(raw), checkpoint_raw=str(raw_path),
        checkpoint_captured_utc=now(),
        checkpoint_capture_runner_sha256=digest(Path(__file__).read_bytes()))
    for path, content in ((raw_path, raw), (artifact, payload),
                          (output, (json.dumps(captured, indent=2, allow_nan=False)+"\n").encode())):
        with path.open("xb") as stream:
            stream.write(content)
    return dict(report=str(output), report_sha256=digest(output.read_bytes()),
                arrays=str(artifact), arrays_sha256=digest(payload), stage=stage)


def main():
    output = Path("reports/gpu_spooled_gradient_capture_driver.json")
    if output.exists():
        raise FileExistsError("Preserve prior capture driver")
    state = dict(status="waiting", started_utc=now(), remote_report=REMOTE,
                 scope="Read-only checkpoint capture; no benchmark modification or numerical acceptance",
                 wait_limit_seconds=28800, poll_seconds=30,
                 runner_sha256=digest(Path(__file__).read_bytes()), checkpoints=[])

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+"\n")
        temporary.replace(output)

    with output.open("x") as stream:
        stream.write(json.dumps(state, indent=2, allow_nan=False)+"\n")
    deadline = time.monotonic()+state["wait_limit_seconds"]
    try:
        while time.monotonic() < deadline:
            raw = fetch(REMOTE)
            report = json.loads(raw)
            check_report(report)
            state.update(last_poll_utc=now(), remote_status=report["status"],
                         remote_stage=report.get("current_stage"),
                         remote_completed_operations=report.get("completed_operations", []))
            if "value_and_grad" in report.get("completed_operations", []):
                payload = fetch(report["arrays"])
                # The producer may replace arrays between the two reads.
                if digest(payload) != report["arrays_sha256"]:
                    state["last_retry"] = "Report/array race; no checkpoint created"
                else:
                    stages = {c["stage"] for c in state["checkpoints"]}
                    if "timed" not in stages:
                        state["checkpoints"].append(capture(raw, report, payload, "timed"))
                        print("Captured complete nominal gradient timings", flush=True)
                        save()
                    if all("value_gradient_nll_absolute_error" in c
                           for c in report["reference_validation"]):
                        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
                            ready = "validation_gradients" in arrays
                        if ready:
                            state["checkpoints"].append(capture(raw, report, payload, "all_points"))
                            state.update(status="captured", finished_utc=now())
                            save()
                            print("Captured all 26 gradient validation points", flush=True)
                            return
            if report["status"] != "running":
                raise RuntimeError("Benchmark terminated before the requested gradient checkpoints")
            save()
            time.sleep(state["poll_seconds"])
        raise TimeoutError("Capture wait limit reached")
    except Exception:
        state.update(status="failed", finished_utc=now(), exception=traceback.format_exc())
        save()
        raise


if __name__ == "__main__":
    main()
