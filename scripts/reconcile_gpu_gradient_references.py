"""Freeze documented nominal FD refinements, then compare saved GPU/CPU arrays.

Raw comparisons remain unchanged. Fine-step pairs are selected solely by their
reference convergence, before testing AD; a failed AD check never advances to a
different pair. The frozen reference can then be applied unchanged to CPU output.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def read(path):
    path = Path(path)
    return json.loads(path.read_text()), hashlib.sha256(path.read_bytes()).hexdigest()


def relative(a, b):
    return abs(a-b)/max(abs(b), 1e-8)


def build_reference(benchmark_path):
    raw_path = Path("reports/gpu_combine_gradient.json")
    fine_path = Path("reports/gpu_combine_gradient_fine.json")
    comparison_path = benchmark_path.with_name(benchmark_path.stem+"_combine_gradient.json")
    stable_path = benchmark_path.with_name(benchmark_path.stem+"_stable_fd.json")
    raw, raw_hash = read(raw_path)
    fine, fine_hash = read(fine_path)
    comparison, comparison_hash = read(comparison_path)
    stable, stable_hash = read(stable_path)
    assert fine["status"] == raw["status"] == "reference_ready"
    assert fine["parent_reference_sha256"] == raw_hash
    assert comparison["reference_sha256"] == raw_hash
    assert stable["reference_sha256"] == raw_hash and stable["status"] == "diagnostic_complete"
    assert stable["max_combine_delta_absolute_error"] < 1e-6
    for candidate in (fine, comparison, stable):
        assert candidate["source"] == raw["source"]
    fine_cases = {c["name"]: c for c in fine["cases"]}
    stable_cases = {c["name"]: c for c in stable["cases"]}
    initially_failed = set(comparison["raw_fd_failed_names"])
    result = dict(source=raw["source"], parameter_names=raw["parameter_names"], initial=raw["initial"],
                  combine_version=raw["combine_version"], root_version=raw["root_version"],
                  provenance={str(p): h for p,h in ((raw_path,raw_hash),(fine_path,fine_hash),
                                                   (comparison_path,comparison_hash),(stable_path,stable_hash))},
                  policy="33 independently checked stable-bin references at both original steps; original failed or step-uncertain remaining coordinates use the first adjacent fine pair with reference relative change <1e-4, and AD must pass both; other coordinates retain the original smaller-step reference. No AD-based choice of pair, no relaxed tolerance.",
                  scope="Nominal per-parameter gradient acceptance only; no random-point derivatives, Hessian or fit acceptance",
                  cases=[], excluded=[])
    for case in raw["cases"]:
        name = case["name"]
        if case["status"] == "fixed_bounds":
            result["excluded"].append(dict(name=name, index=case["index"], reason=case["reason"]))
            continue
        assert case["status"] == "reference_ready"
        selected = dict(name=name, index=case["index"])
        if name in stable_cases:
            probes = stable_cases[name]["probes"]
            assert [p["step"] for p in probes] == [p["step"] for p in case["probes"]]
            selected.update(method="stable_root_bin_nll_fd", probes=[dict(step=p["step"],derivative=p["derivative"]) for p in probes])
        elif name in initially_failed or case["step_halving_relative_change"] >= 1e-4:
            probes = fine_cases[name]["probes"]
            pair = next(((a,b) for a,b in zip(probes,probes[1:]) if relative(a["derivative"],b["derivative"]) < 1e-4), None)
            if pair is None:
                raise ValueError("No convergent fine reference pair for "+name)
            selected.update(method="converged_fine_combine_nll_fd",
                            reason="Initial AD mismatch" if name in initially_failed else "Initial reference step uncertainty",
                            pair_relative_change=relative(pair[0]["derivative"],pair[1]["derivative"]),
                            probes=[dict(step=p["step"],derivative=p["derivative"]) for p in pair])
        else:
            selected.update(method="original_combine_nll_fd", probes=[dict(step=case["probes"][-1]["step"],derivative=case["probes"][-1]["derivative"])])
        result["cases"].append(selected)
    result["method_counts"] = {method: sum(c["method"] == method for c in result["cases"])
                               for method in sorted({c["method"] for c in result["cases"]})}
    path = Path("reports/gpu_combine_gradient_resolved.json")
    if path.exists():
        raise FileExistsError("Preserve the frozen reference; explicitly review before replacing "+str(path))
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--build-reference", action="store_true")
    parser.add_argument("--directions-reference", type=Path, help="Also require agreement with both final fixed steps of a completed directional scan")
    arguments = parser.parse_args()
    if arguments.build_reference:
        build_reference(arguments.benchmark)
    reference_path = Path("reports/gpu_combine_gradient_resolved.json")
    reference, reference_hash = read(reference_path)
    benchmark, benchmark_hash = read(arguments.benchmark)
    assert reference["source"] == benchmark["source"]
    artifact = Path(benchmark["arrays"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == benchmark["arrays_sha256"]
    with np.load(artifact,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["parameter_names"],reference["parameter_names"])
        np.testing.assert_array_equal(arrays["point"],reference["initial"])
        gradient = arrays["gradient"].copy()
        assert np.isfinite(gradient).all()
    result = dict(benchmark=str(arguments.benchmark), benchmark_sha256=benchmark_hash,
                  arrays_sha256=benchmark["arrays_sha256"], reference=str(reference_path), reference_sha256=reference_hash,
                  source=reference["source"], policy=reference["policy"], scope=reference["scope"],
                  method_counts=reference["method_counts"], excluded=reference["excluded"], cases=[])
    for case in reference["cases"]:
        actual = float(gradient[case["index"]])
        probes = [dict(step=p["step"], reference=p["derivative"], absolute_error=abs(actual-p["derivative"]),
                       relative_error=relative(actual,p["derivative"])) for p in case["probes"]]
        result["cases"].append(dict(name=case["name"], method=case["method"], ad=actual, probes=probes))
    result.update(max_relative_error=max(p["relative_error"] for c in result["cases"] for p in c["probes"]),
                  max_absolute_error=max(p["absolute_error"] for c in result["cases"] for p in c["probes"]),
                  failed_names=[c["name"] for c in result["cases"] if any(p["relative_error"] >= 1e-4 for p in c["probes"])])
    result["status"] = "passed" if not result["failed_names"] else "failed"
    if arguments.directions_reference:
        directions, directions_hash = read(arguments.directions_reference)
        assert directions["status"] == "reference_ready" and directions["source"] == reference["source"]
        np.testing.assert_array_equal(directions["initial"], reference["initial"])
        np.testing.assert_array_equal(directions["parameter_names"], reference["parameter_names"])
        cases = []
        for case in directions["cases"]:
            actual = float(np.dot(gradient, case["direction"]))
            pair = case["probes"][-2:]
            probes = [dict(step=p["step"], reference=p["derivative"], absolute_error=abs(actual-p["derivative"]),
                           relative_error=relative(actual,p["derivative"])) for p in pair]
            cases.append(dict(name=case["label"], ad=actual, probes=probes,
                              reference_pair_relative_change=relative(pair[0]["derivative"],pair[1]["derivative"])))
        passed = all(c["reference_pair_relative_change"] < 1e-4 and all(p["relative_error"] < 1e-4 for p in c["probes"]) for c in cases)
        result["directions"] = dict(reference=str(arguments.directions_reference), reference_sha256=directions_hash,
            policy="Both final fixed steps must converge and agree with AD at relative tolerance 1e-4; no closest-step selection",
            status="passed" if passed else "failed", cases=cases,
            max_relative_error=max(p["relative_error"] for c in cases for p in c["probes"]),
            max_absolute_error=max(p["absolute_error"] for c in cases for p in c["probes"]),
            max_reference_pair_relative_change=max(c["reference_pair_relative_change"] for c in cases))
        if not passed:
            result["status"] = "failed"
    output = arguments.benchmark.with_name(arguments.benchmark.stem+"_resolved_gradient.json")
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(output,result["status"],result["method_counts"],result["max_relative_error"],result["max_absolute_error"])
    assert result["status"] == "passed", result["failed_names"]


if __name__ == "__main__":
    main()
