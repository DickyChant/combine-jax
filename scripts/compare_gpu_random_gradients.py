"""Check captured random gradients against frozen, independent Combine probes."""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def compare_selected(result, derivative, chosen):
    errors = [dict(step=p["step"], **({"estimator": p["estimator"]} if "estimator" in p else {}),
                   reference=p["derivative"],
                   absolute_error=float(abs(derivative-p["derivative"])),
                   relative_error=float(abs(derivative-p["derivative"])/max(abs(p["derivative"]),1e-8)))
              for p in chosen]
    maximum = max(e["relative_error"] for e in errors)
    return dict(result, status="passed" if maximum < 1e-4 else "failed",
                selected_references=errors, max_relative_error=maximum,
                max_absolute_error=max(e["absolute_error"] for e in errors))


def compare_coordinate(case, derivative):
    result = dict(name=case["name"], index=case["index"],
                  reference_status=case["status"],
                  ad_derivative=float(derivative) if np.isfinite(derivative) else None)
    if not np.isfinite(derivative):
        return dict(result, status="failed", reason="Nonfinite AD derivative")
    if case["status"] == "fixed_bounds":
        return dict(result, status="fixed_bounds")
    if case["status"] != "converged":
        return dict(result, status="reference_uncertain")
    first = case["selected_first_probe_index"]
    probes = case["probes"]
    # Recheck that the frozen selection is the FIRST stable consecutive triple.
    selected = None
    for i in range(len(probes)-2):
        triple = probes[i:i+3]
        if not all(p["status"] == "valid" for p in triple):
            continue
        values = [p["derivative"] for p in triple]
        changes = [abs(a-b)/max(abs(b), 1e-8) for a,b in zip(values,values[1:])]
        if max(changes) < 1e-4:
            selected = i
            break
    assert selected == first, "Reference does not select its first converged triple"
    chosen = probes[first:first+3]
    assert len(chosen) == 3 and min(abs(p["derivative"]) for p in chosen) >= 1e-4
    return compare_selected(dict(result,selected_first_probe_index=first),derivative,chosen)


def compare_stable_coordinate(case, stable, derivative):
    assert case["status"] == "roundoff_sensitive"
    assert all(case[k] == stable[k] for k in ("name","index","coordinate"))
    result = compare_coordinate(case,derivative)
    result["reference_method"] = "ROOT component bin expectations and stable NLL differences"
    if result["status"] == "failed" or stable["status"] != "reference_ready":
        return result
    chosen = stable["probes"]
    assert len(chosen) == len(case["probes"]) == 3
    assert [p["step"] for p in chosen] == [p["step"] for p in case["probes"]]
    for probe in chosen:
        checks = probe["delta_checks"]
        assert len(checks) == 4
        assert all(d["absolute_error"] < 1e-6 and abs(d["stable_delta"]-d["combine_delta"]) < 1e-6 for d in checks)
        np.testing.assert_array_equal([d["coordinate"] for d in checks],
                                      [case["coordinate"]+i*probe["step"] for i in (-2,-1,1,2)])
        values = np.array([d["stable_delta"] for d in checks],dtype=np.longdouble)
        recomputed = float((8*(values[2]-values[1])-(values[3]-values[0]))/(12*np.longdouble(probe["step"])))
        assert abs(recomputed-probe["derivative"])/max(abs(probe["derivative"]),1e-8) < 1e-9
    values = [p["derivative"] for p in chosen]
    assert max(abs(a-b)/max(abs(b),1e-8) for a,b in zip(values,values[1:])) < 1e-4
    return compare_selected(result,derivative,chosen)


def compare_refined_coordinate(point, case, refinement, derivative):
    original = compare_coordinate(case, derivative)
    assert case == refinement["case"]["original_reference"]
    for key in ("point_number", "attempt", "theta", "canonical_nll"):
        assert point[key] == refinement["point"][key], f"Changed reference {key}"
    result = dict(name=case["name"], index=case["index"],
                  reference_status=case["status"], original_comparison=original,
                  refined_reference_status=refinement["reference_status"],
                  reference_method=refinement["method"],
                  ad_derivative=float(derivative) if np.isfinite(derivative) else None)
    if not np.isfinite(derivative):
        return dict(result, status="failed", reason="Nonfinite AD derivative")
    if refinement["reference_status"] != "reference_ready":
        return dict(result, status="reference_uncertain")
    assert len(refinement["chosen"]) in (3, 12)
    return compare_selected(result, derivative, refinement["chosen"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--reference", type=Path,
                        default=Path("reports/gpu_combine_random_gradient.json"))
    parser.add_argument("--stable-reference", type=Path,
                        help="Optional independent stable-bin reference for the frozen small-derivative cases")
    parser.add_argument("--refined-full-nll-reference", type=Path,
                        help="Explicit fixed-grid follow-up for the frozen 14 failed cases")
    parser.add_argument("--refined-intensity-reference", type=Path,
                        help="Explicit short-stencil stable-intensity follow-up for two of those cases")
    parser.add_argument("--output", type=Path,
                        help="Separate report path for validation-harness comparisons")
    args = parser.parse_args()
    if bool(args.refined_full_nll_reference) != bool(args.refined_intensity_reference):
        parser.error("Both refinement references must be supplied together")
    if args.refined_full_nll_reference and not args.stable_reference:
        parser.error("The original 24-case stable-bin reference is also required")
    benchmark_bytes = args.benchmark.read_bytes()
    benchmark = json.loads(benchmark_bytes)
    reference_bytes = args.reference.read_bytes()
    reference = json.loads(reference_bytes)
    assert reference["status"] in ("reference_ready", "reference_uncertain")
    assert reference["source"] == benchmark["source"]
    assert sha256(Path(reference["parent_reference"]).read_bytes()) == reference["parent_reference_sha256"]
    if "extends_reference" in reference:
        assert sha256(Path(reference["extends_reference"]).read_bytes()) == reference["extends_reference_sha256"]
    artifact = Path(benchmark["arrays"])
    assert sha256(artifact.read_bytes()) == benchmark["arrays_sha256"]
    report = dict(benchmark=str(args.benchmark), benchmark_sha256=sha256(benchmark_bytes),
                  arrays_sha256=benchmark["arrays_sha256"], source=reference["source"],
                  checker_sha256=sha256(Path(__file__).read_bytes()),
                  reference=str(args.reference), reference_sha256=sha256(reference_bytes),
                  policy="AD must pass all three independently selected references at relative tolerance 1e-4, denominator max(abs(reference),1e-8); unresolved references cannot pass. Fixed-bound AD entries must be finite.",
                  scope="All captured random points versus Combine v11 finite differences; nominal validation is separate",
                  points=[])
    stable_cases = {}
    if args.stable_reference:
        raw = args.stable_reference.read_bytes()
        stable_reference = json.loads(raw)
        assert stable_reference["status"] in ("reference_ready","reference_uncertain")
        assert stable_reference["reference_sha256"] == sha256(reference_bytes)
        assert stable_reference["source"] == reference["source"]
        assert stable_reference["parameter_names"] == reference["parameter_names"]
        assert all(stable_reference[k] == reference[k] for k in ("combine_version","root_version","runtime_flags"))
        for point in stable_reference["points"]:
            for case in point["cases"]:
                key = point["point_number"],case["index"]
                assert key not in stable_cases
                stable_cases[key] = (point,case)
        assert len(stable_cases) == 24
        report.update(stable_reference=str(args.stable_reference),stable_reference_sha256=sha256(raw),
                      stable_reference_scope="Only the original 24 roundoff-sensitive cases; all original three step sizes retained; raw full-NLL references remain preserved")
    used_stable = set()
    refinements = {}
    used_refined = set()
    if args.refined_full_nll_reference:
        from gpu_random_refinements import load_refinements
        refinements, provenance = load_refinements(args.refined_full_nll_reference,
            args.refined_intensity_reference, reference, sha256(reference_bytes))
        assert not set(refinements).intersection(stable_cases)
        report["refinements"] = provenance
        report["refinement_checker_sha256"] = {str(p):sha256(p.read_bytes()) for p in
            (Path(__file__).with_name("gpu_random_refinements.py"), Path(__file__).with_name("fd_reference.py"))}
        report["policy"] += " Explicit follow-ups replace only the frozen 14 cases and require all three or twelve selected estimates; original comparisons are retained."
    overall = Counter()
    with np.load(artifact, allow_pickle=False) as archive:
        # NpzFile reads/decompresses a member again on every indexing operation.
        # Retain the small validation tables once, without loading any Hessian.
        arrays = {key: archive[key] for key in ("parameter_names", "validation_labels",
                  "validation_points", "validation_values", "validation_vg_values", "validation_gradients")}
        np.testing.assert_array_equal(arrays["parameter_names"], reference["parameter_names"])
        labels = list(arrays["validation_labels"])
        assert len(set(labels)) == len(labels)
        assert arrays["validation_gradients"].shape == arrays["validation_points"].shape
        assert len(reference["points"]) == 25 and len(labels) == 26
        assert labels[0] == "nominal"
        for index in reference["points"]:
            blob = Path(index["artifact"]).read_bytes()
            assert sha256(blob) == index["artifact_sha256"]
            raw = gzip.decompress(blob)
            assert sha256(raw) == index["raw_sha256"]
            point = json.loads(raw)
            row = labels.index(f"random_attempt_{point['attempt']}")
            np.testing.assert_array_equal(arrays["validation_points"][row], point["theta"])
            nll_errors = {}
            for key in ("validation_values", "validation_vg_values"):
                value = arrays[key][row]
                nll_errors[key] = float(abs(value-point["canonical_nll"])) if np.isfinite(value) else None
            checks = []
            assert len(point["cases"]) == len(reference["parameter_names"])
            for i, case in enumerate(point["cases"]):
                assert case["index"] == i and case["name"] == reference["parameter_names"][i]
                assert case["coordinate"] == point["theta"][i]
                key = point["point_number"],i
                if key in refinements:
                    checks.append(compare_refined_coordinate(point,case,refinements[key],arrays["validation_gradients"][row,i]))
                    used_refined.add(key)
                elif key in stable_cases:
                    stable_point,stable = stable_cases[key]
                    np.testing.assert_array_equal(stable_point["theta"],point["theta"])
                    assert stable_point["attempt"] == point["attempt"]
                    checks.append(compare_stable_coordinate(case,stable,arrays["validation_gradients"][row,i]))
                    used_stable.add(key)
                else:
                    checks.append(compare_coordinate(case, arrays["validation_gradients"][row,i]))
            counts = Counter(c["status"] for c in checks)
            overall.update(counts)
            ranked = sorted((c for c in checks if "max_relative_error" in c),
                            key=lambda c:c["max_relative_error"], reverse=True)
            failures = [c for c in checks if c["status"] == "failed"]
            uncertain = [c for c in checks if c["status"] == "reference_uncertain"]
            nll_passed = all(e is not None and e < 1e-6 for e in nll_errors.values())
            status = ("failed" if failures or not nll_passed else
                      "reference_uncertain" if uncertain else "passed")
            report["points"].append(dict(point_number=point["point_number"], attempt=point["attempt"],
                                         status=status, status_counts=dict(counts),
                                         nll_absolute_errors=nll_errors,
                                         max_relative_error=max((c["max_relative_error"] for c in ranked),default=0.),
                                         max_absolute_error=max((c["max_absolute_error"] for c in ranked),default=0.),
                                         worst_converged_references=ranked[:10],
                                         refined_checks=[c for c in checks if "original_comparison" in c],
                                         failures=failures, unresolved=uncertain))
    report["status_counts"] = dict(overall)
    assert used_stable == set(stable_cases)
    assert used_refined == set(refinements)
    report["status"] = ("failed" if any(p["status"] == "failed" for p in report["points"]) else
                        "reference_uncertain" if overall["reference_uncertain"] else "passed")
    tag = "_refined" if refinements else "_stable" if args.stable_reference else ""
    output = args.output or args.benchmark.with_name(args.benchmark.stem+"_combine_random_gradients"+tag+".json")
    if refinements and output.exists():
        raise FileExistsError(str(output))
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(output, report["status"], dict(overall))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
