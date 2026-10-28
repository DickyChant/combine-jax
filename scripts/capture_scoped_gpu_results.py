"""Preserve completed HHVV scoped benchmark stages without changing the producer."""

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import time
import traceback

import numpy as np

import capture_spooled_gpu_gradient as transport


STAGES = ("value", "gradient", "gradient_all_points", "hessian")
OPERATIONS = dict(value="value", gradient="value_and_grad",
                  gradient_all_points="value_and_grad", hessian="hessian")
MANIFEST = Path("build/gpu_exports/manifest_random.json")


def now():
    return datetime.now(timezone.utc).isoformat()


def remote_report(backend):
    assert backend in ("cpu", "gpu")
    return f"reports/gpu_hhvv_{backend}_scoped.json"


def check_report(report, backend):
    """Bind a checkpoint to this tested source, model and explicit compiler policy."""
    assert report["model"] == "hhvv" and report["backend"] == backend
    assert report["host"] == "lxplus905.cern.ch"
    assert report["workdir"] == "/tmp/sqian/combine-jax"
    assert report["devices"] == (["cuda:0"] if backend == "gpu" else ["cpu:0"])
    assert report["float64"] and report["jax_version"] == report["jaxlib_version"] == "0.10.1"
    assert report["lcg_view"] == "/cvmfs/sft.cern.ch/lcg/views/LCG_110a_cuda/x86_64-el9-gcc13-opt/setup.sh"
    assert report["parameters"] == 2656 and report["terms"] == 524
    assert report["report_tag"] == "scoped" and report["scoped_loads"]
    for key in ("spool_value_gradient", "spool_hessian_executables", "cpu_hessian_no_fusion",
                "trim_compilation_cache", "release_stages"):
        assert report[key], key
    assert report["hessian_chunk_size"] == 4 and report["constraint_batch_size"] == 32
    assert report["hessian_compiler_options"] == (
        {"xla_disable_hlo_passes": "fusion", "xla_cpu_parallel_codegen_split_count": 1}
        if backend == "cpu" else {})
    assert report["arrays"] == f"build/gpu_results/hhvv_{backend}_scoped.npz"
    assert report["source_sha256"] == transport.digest(b"".join(
        str(p).encode()+b"\0"+p.read_bytes()
        for p in sorted(Path("src/combine_jax").rglob("*.py"))))
    for key, path in {
        "benchmark_sha256": Path("scripts/gpu_benchmark.py"),
        "scoped_executable_sha256": Path("scripts/scoped_executable.py"),
        "spooled_executable_sha256": Path("scripts/spooled_executable.py"),
        "reference_manifest_sha256": MANIFEST,
    }.items():
        assert report[key] == transport.digest(path.read_bytes()), key
    manifest = json.loads(MANIFEST.read_bytes())
    assert report["export_stack"] == {k: manifest[k] for k in ("combine_version", "root_version")}
    model = manifest["models"]["hhvv"]
    assert report["source"] == model["source"]
    return model["reference"]


def ready(report, stage):
    operation = OPERATIONS[stage]
    if operation not in report.get("completed_operations", []):
        return False
    if stage in ("value", "gradient_all_points"):
        key = "nll_absolute_error" if stage == "value" else "value_gradient_nll_absolute_error"
        return len(report["reference_validation"]) == 26 and all(
            key in case for case in report["reference_validation"])
    return True


def audit_arrays(report, payload, stage, reference):
    """Check completeness and reproduce saved NLL errors; no derivative acceptance."""
    assert stage in STAGES and ready(report, stage)
    assert transport.digest(payload) == report["arrays_sha256"]
    operation = OPERATIONS[stage]
    timing = report["timings"][operation]
    terms = timing["term_timings"]
    assert len(terms) == report["terms"]
    assert len({t["name"] for t in terms}) == report["terms"]
    assert report["preparation_timings"][operation]["completed_terms"] == report["terms"]
    seconds = [timing["first_execution_seconds"], *timing["warm_seconds"]]
    assert len(seconds) == 6 and all(np.isfinite(s) and s > 0 for s in seconds)
    assert float(np.median(seconds[1:])) == timing["median_warm_seconds"]
    for term in terms:
        options = report["hessian_compiler_options"] if operation == "hessian" else {}
        if term.get("skipped_empty_hessian"):
            assert operation == "hessian" and term["parameters"] == 0
            options = {}
        assert term["compiler_options"] == options
    if operation != "value":
        calls = timing["spooled_timed_calls"]
        expected = sum((t["parameters"]+3)//4 if operation == "hessian" else 1
                       for t in terms if not t.get("skipped_empty_hessian"))
        assert len(calls) == 6 and [c["total_seconds"] for c in calls] == seconds
        assert all(c["kernel_calls"] == expected for c in calls)
    if stage == "gradient_all_points":
        sessions = timing["scoped_load_sessions"]
        assert set(sessions) == {t["name"] for t in terms}
        for term in terms:
            records = sessions[term["name"]]
            assert len(records) == 1 and records[0]["calls"] == 26
            assert records[0]["exception_type"] is None
            assert timing["spooled_all_calls"][term["name"]]["count"] == 32
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        size = report["parameters"]
        points = np.asarray([c["theta"] for c in reference["cases"]])
        assert points.shape == (26, size)
        np.testing.assert_array_equal(arrays["parameter_names"], reference["parameter_names"])
        np.testing.assert_array_equal(arrays["validation_points"], points)
        np.testing.assert_array_equal(arrays["point"], points[0])
        np.testing.assert_array_equal(arrays["validation_labels"], [
            c.get("label", f"case_{i}") for i, c in enumerate(reference["cases"])])
        shapes = dict(value=(), term_values=(report["terms"],), validation_values=(26,))
        if stage != "value":
            shapes.update(vg_value=(), gradient=(size,))
        if stage in ("gradient_all_points", "hessian"):
            shapes.update(validation_vg_values=(26,), validation_gradients=(26, size))
        if stage == "hessian":
            shapes["hessian"] = (size, size)
        for key, shape in shapes.items():
            assert arrays[key].shape == shape and arrays[key].dtype == np.float64, key
            assert np.isfinite(arrays[key]).all(), key
        expected = np.asarray([c["combine_nll"] for c in reference["cases"]])
        errors = abs(arrays["validation_values"]-expected)
        np.testing.assert_array_equal(errors, [c["nll_absolute_error"]
                                             for c in report["reference_validation"]])
        np.testing.assert_array_equal(arrays["value"], arrays["validation_values"][0])
        audit = dict(nll_max_absolute_error=float(errors.max()), nll_pass=bool((errors < 1e-6).all()))
        if stage in ("gradient_all_points", "hessian"):
            errors = abs(arrays["validation_vg_values"]-expected)
            np.testing.assert_array_equal(errors, [c["value_gradient_nll_absolute_error"]
                                                 for c in report["reference_validation"]])
            audit.update(value_gradient_nll_max_absolute_error=float(errors.max()),
                         value_gradient_nll_pass=bool((errors < 1e-6).all()),
                         fresh_nominal_gradient_max_absolute_difference=float(abs(
                             arrays["gradient"]-arrays["validation_gradients"][0]).max()))
        if stage == "hessian":
            audit["hessian_max_asymmetry"] = float(abs(arrays["hessian"]-arrays["hessian"].T).max())
    return audit


def capture(raw, payload, backend, stage):
    report = json.loads(raw)
    reference = check_report(report, backend)
    audit = audit_arrays(report, payload, stage, reference)
    stem = f"gpu_hhvv_{backend}_scoped_{stage}_checkpoint"
    raw_path, output = [Path("reports")/(stem+suffix) for suffix in ("_raw.json", ".json")]
    artifact = Path("build/gpu_results")/(stem+".npz")
    for path in (raw_path, output, artifact):
        if path.exists():
            raise FileExistsError(f"Preserve prior checkpoint: {path}")
    captured = dict(report)
    captured.update(status="checkpoint", arrays=str(artifact), checkpoint_stage=stage,
        checkpoint_parent_status=report["status"], checkpoint_parent_report=remote_report(backend),
        checkpoint_parent_report_sha256=transport.digest(raw), checkpoint_raw=str(raw_path),
        checkpoint_captured_utc=now(), checkpoint_integrity_audit=audit,
        checkpoint_scope="Completed stage timings and finite complete arrays. NLL errors independently reproduced from the frozen manifest. Gradient/Hessian acceptance requires separate frozen-reference comparisons. Later-stage failures remain possible.",
        checkpoint_capture_sources={str(Path(p).relative_to(Path.cwd())): transport.digest(Path(p).read_bytes())
                                    for p in (__file__, transport.__file__)})
    for path, content in ((raw_path, raw), (artifact, payload),
                          (output, (json.dumps(captured, indent=2, allow_nan=False)+"\n").encode())):
        with path.open("xb") as stream:
            stream.write(content)
    return dict(stage=stage, report=str(output), report_sha256=transport.digest(output.read_bytes()),
                arrays=str(artifact), arrays_sha256=transport.digest(payload), audit=audit)


def poll_once(backend, stages, state):
    raw = transport.fetch(remote_report(backend))
    report = json.loads(raw)
    check_report(report, backend)
    state.update(last_poll_utc=now(), remote_status=report["status"],
                 remote_stage=report.get("current_stage"),
                 remote_completed_operations=report.get("completed_operations", []))
    captured = {c["stage"] for c in state["checkpoints"]}
    available = [s for s in stages if s not in captured and ready(report, s)]
    if available:
        payload = transport.fetch(report["arrays"])
        # The live producer saves arrays separately from its atomic JSON update.
        if transport.digest(payload) != report["arrays_sha256"]:
            state["last_retry"] = "Report/array race; no checkpoint created"
            return False
        for stage in available:
            result = capture(raw, payload, backend, stage)
            state["checkpoints"].append(result)
            print(json.dumps(result), flush=True)
    return all(s in {c["stage"] for c in state["checkpoints"]} for s in stages)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--wait-seconds", type=float, default=86400.)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.is_absolute() or args.output.parent != Path("reports") or args.wait_seconds <= 0:
        parser.error("Use a fresh report path under reports/ and a positive observation limit")
    state = dict(status="waiting", started_utc=now(), backend=args.backend, stages=args.stages,
                 remote_report=remote_report(args.backend), wait_limit_seconds=args.wait_seconds,
                 scope="Read-only capture. Observation limits never terminate or determine benchmark status.",
                 runner_sha256=transport.digest(Path(__file__).read_bytes()), checkpoints=[])
    with args.output.open("x") as stream:
        stream.write(json.dumps(state, indent=2)+"\n")

    def save():
        temporary = args.output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(state, indent=2, allow_nan=False)+"\n")
        temporary.replace(args.output)

    deadline = time.monotonic()+args.wait_seconds
    try:
        while time.monotonic() < deadline:
            done = poll_once(args.backend, args.stages, state)
            save()
            if done:
                state["status"] = "captured"
                break
            if state["remote_status"] != "running":
                state["status"] = "producer_terminated_before_requested_stages"
                break
            time.sleep(min(30., max(0., deadline-time.monotonic())))
        else:
            state["status"] = "observation_limit"
    except BaseException:
        state.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        state["finished_utc"] = now()
        save()
    raise SystemExit(0 if state["status"] == "captured" else 1)


if __name__ == "__main__":
    main()
