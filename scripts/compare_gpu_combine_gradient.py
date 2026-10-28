"""Compare an available full HHVV gradient with independent Combine references."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def error(ad, reference):
    absolute = abs(ad-reference)
    return dict(ad=float(ad), reference=float(reference), absolute_error=float(absolute),
                relative_error=float(absolute/max(abs(reference), 1e-8)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    arguments = parser.parse_args()
    benchmark = json.loads(arguments.benchmark.read_text())
    artifact = Path(benchmark["arrays"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == benchmark["arrays_sha256"]
    reference_path = Path("reports/gpu_combine_gradient.json")
    reference = json.loads(reference_path.read_text())
    assert reference["source"] == benchmark["source"]
    assert reference["status"] == "reference_ready"
    with np.load(artifact, allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["parameter_names"], reference["parameter_names"])
        np.testing.assert_array_equal(arrays["point"], reference["initial"])
        gradient = arrays["gradient"].copy()
        assert np.isfinite(gradient).all()
    report = dict(benchmark=str(arguments.benchmark), benchmark_backend=benchmark["backend"],
                  source=benchmark["source"], arrays_sha256=benchmark["arrays_sha256"],
                  reference=str(reference_path), reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  policy="Raw smaller-step per-parameter FD uses relative error <1e-4 with denominator max(abs(FD),1e-8); no automatic step replacement or tolerance relaxation",
                  cases=[], excluded=[], directions=[], refinement_diagnostics=[])
    for case in reference["cases"]:
        if case["status"] == "fixed_bounds":
            report["excluded"].append(dict(name=case["name"], reason=case["reason"]))
            continue
        assert case["status"] == "reference_ready"
        comparison = error(gradient[case["index"]], case["probes"][-1]["derivative"])
        report["cases"].append(dict(name=case["name"], index=case["index"],
                                    reference_step=case["probes"][-1]["step"],
                                    reference_step_relative_change=case["step_halving_relative_change"], **comparison))
    directions_path = Path("reports/gpu_combine_directions.json")
    directions = json.loads(directions_path.read_text())
    assert directions["source"] == benchmark["source"]
    np.testing.assert_array_equal(reference["initial"], directions["initial"])
    np.testing.assert_array_equal(reference["parameter_names"], directions["parameter_names"])
    for case in directions["cases"]:
        report["directions"].append(dict(name=case["label"], **error(
            np.dot(gradient, case["direction"]), case["probes"][-1]["derivative"])))
    refined_path = Path("reports/gpu_combine_gradient_refined.json")
    refined = json.loads(refined_path.read_text())
    assert refined["source"] == benchmark["source"]
    assert refined["parent_reference_sha256"] == report["reference_sha256"]
    report.update(directions_reference_sha256=hashlib.sha256(directions_path.read_bytes()).hexdigest(),
                  refinement_reference_sha256=hashlib.sha256(refined_path.read_bytes()).hexdigest())
    for case in refined["cases"]:
        report["refinement_diagnostics"].append(dict(name=case["name"],
            note="All raw step probes are shown; these do not replace the acceptance reference",
            probes=[dict(step=p["step"], **error(gradient[case["index"]], p["derivative"])) for p in case["probes"]]))
    stable_path = Path("reports/hhvv_stable_fd.json")
    if stable_path.exists():
        stable = json.loads(stable_path.read_text())
        assert stable["status"] == "diagnostic_complete" and stable["source"] == benchmark["source"]
        assert stable["max_combine_delta_absolute_error"] < 1e-6
        local_ad_path = Path(stable["ad_report"])
        assert hashlib.sha256(local_ad_path.read_bytes()).hexdigest() == stable["ad_report_sha256"]
        local_ad = json.loads(local_ad_path.read_text())
        assert local_ad["reference_sha256"] == report["refinement_reference_sha256"]
        raw_by_name = {c["name"]: c for c in reference["cases"]}
        stable_cases = []
        for case in stable["cases"]:
            raw = raw_by_name[case["name"]]
            probe = case["probes"][1]
            assert probe["step"] == raw["probes"][-1]["step"]
            stable_cases.append(dict(name=case["name"], step=probe["step"],
                                     **error(gradient[raw["index"]], probe["derivative"])))
        report["stable_bin_reference"] = dict(
            reference=str(stable_path), reference_sha256=hashlib.sha256(stable_path.read_bytes()).hexdigest(),
            method=stable["method"], scope="Supplementary 25-parameter reference; raw full-gradient gate is retained separately",
            max_relative_error=max(c["relative_error"] for c in stable_cases),
            max_absolute_error=max(c["absolute_error"] for c in stable_cases), cases=stable_cases,
            status="passed" if all(c["relative_error"] < 1e-4 for c in stable_cases) else "failed")
    failures = [c for c in report["cases"] if c["relative_error"] >= 1e-4]
    report.update(raw_fd_failed_names=[c["name"] for c in failures],
                  raw_fd_max_relative_error=max(c["relative_error"] for c in report["cases"]),
                  raw_fd_max_absolute_error=max(c["absolute_error"] for c in report["cases"]),
                  raw_fd_failures_with_step_uncertainty=[c["name"] for c in failures if c["reference_step_relative_change"] >= 1e-4],
                  direction_max_relative_error=max(c["relative_error"] for c in report["directions"]))
    report["raw_fd_status"] = "failed" if failures else "passed"
    report["direction_status"] = "passed" if report["direction_max_relative_error"] < 1e-4 else "failed"
    output = arguments.benchmark.with_name(arguments.benchmark.stem+"_combine_gradient.json")
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(output, report["raw_fd_status"], "failed parameters", len(failures),
          "direction", report["direction_status"], report["direction_max_relative_error"])


if __name__ == "__main__":
    main()
