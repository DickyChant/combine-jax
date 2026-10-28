"""Compare a saved nominal Hessian against independently selected Combine probes."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def first_converged_triple(probes):
    assert all(np.isfinite(p["curvature"]) and p["step"] > 0 for p in probes)
    for first in range(len(probes)-2):
        triple = probes[first:first+3]
        changes = [abs(a["curvature"]-b["curvature"])/max(abs(b["curvature"]),1e-8)
                   for a,b in zip(triple,triple[1:])]
        if max(changes) < 1e-4:
            return dict(first_probe_index=first,steps=[p["step"] for p in triple],
                        adjacent_relative_changes=changes)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--reference", type=Path, default=Path("reports/gpu_combine_curvature.json"))
    arguments = parser.parse_args()
    benchmark = json.loads(arguments.benchmark.read_text())
    reference = json.loads(arguments.reference.read_text())
    assert reference["status"] in ("reference_ready", "reference_uncertain")
    assert reference["source"] == benchmark["source"]
    assert reference["parent_reference_sha256"] == hashlib.sha256(Path(reference["parent_reference"]).read_bytes()).hexdigest()
    if "extends_reference" in reference:
        assert reference["extends_reference_sha256"] == hashlib.sha256(Path(reference["extends_reference"]).read_bytes()).hexdigest()
    artifact = Path(benchmark["arrays"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == benchmark["arrays_sha256"]
    with np.load(artifact,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["parameter_names"],reference["parameter_names"])
        np.testing.assert_array_equal(arrays["point"],reference["initial"])
        hessian = arrays["hessian"].copy()
    assert hessian.shape == (len(reference["initial"]),)*2
    assert np.isfinite(hessian).all()
    report = dict(benchmark=str(arguments.benchmark),
                  benchmark_sha256=hashlib.sha256(arguments.benchmark.read_bytes()).hexdigest(),
                  arrays_sha256=benchmark["arrays_sha256"], source=reference["source"],
                  reference=str(arguments.reference),
                  reference_sha256=hashlib.sha256(arguments.reference.read_bytes()).hexdigest(),
                  reference_policy=reference["policy"],
                  policy="Compare d.T @ H @ d with every curvature in the independently selected reference triple; relative tolerance 1e-4, denominator max(abs(reference),1e-8). Unresolved references cannot pass.",
                  scope="18 nominal Hessian projections, including the POI and scale_j; not all-entry Combine Hessian or fit acceptance",
                  hessian_max_asymmetry=float(np.max(np.abs(hessian-hessian.T))), cases=[])
    for case in reference["cases"]:
        direction = np.asarray(case["direction"])
        curvature = float(direction @ hessian @ direction)
        probes = [dict(step=p["step"], reference=p["curvature"],
                       absolute_error=abs(curvature-p["curvature"]),
                       relative_error=abs(curvature-p["curvature"])/max(abs(p["curvature"]),1e-8))
                  for p in case["probes"]]
        selected = case["selected_triple"]
        assert selected == first_converged_triple(case["probes"]), "Reference does not use its first converged triple"
        status = "reference_uncertain"
        if selected is not None:
            first = selected["first_probe_index"]
            chosen = probes[first:first+3]
            assert len(chosen) == 3 and [p["step"] for p in chosen] == selected["steps"]
            assert max(selected["adjacent_relative_changes"]) < 1e-4
            status = "passed" if all(p["relative_error"] < 1e-4 for p in chosen) else "failed"
        report["cases"].append(dict(label=case["label"], status=status, ad_curvature=curvature,
                                    selected_triple=selected, probes=probes))
    assert len(report["cases"]) == 18
    report["failed_directions"] = [c["label"] for c in report["cases"] if c["status"] == "failed"]
    report["unresolved_directions"] = [c["label"] for c in report["cases"] if c["status"] == "reference_uncertain"]
    report["status"] = ("failed" if report["failed_directions"] else
                        "reference_uncertain" if report["unresolved_directions"] else "passed")
    suffix = "_extended" if "extends_reference" in reference else ""
    output = arguments.benchmark.with_name(arguments.benchmark.stem+"_combine_curvature"+suffix+".json")
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(output, report["status"], "failed", report["failed_directions"], "unresolved", report["unresolved_directions"])
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
