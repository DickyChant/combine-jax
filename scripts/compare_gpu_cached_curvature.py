"""Audit cached-density curvature evidence and compare a complete nominal Hessian.

This separate consumer leaves the active reference runner and original comparison
policies unchanged. Its output never replaces an earlier curvature comparison.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_gpu_curvature import first_converged_triple


FINER_STEPS = [1e-6, 5e-7, 2e-7]


def document(path, expected=None):
    raw = Path(path).read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    if expected is not None:
        assert checksum == expected, str(path)
    return json.loads(raw), checksum


def audit_reference(path):
    reference, checksum = document(path)
    if "extends_cached_reference" in reference:
        return audit_finer_reference(path, reference, checksum)
    assert reference["status"] in ("reference_ready", "reference_uncertain"), "Reference is unfinished or failed"
    parent, _ = document(reference["parent_reference"], reference["parent_reference_sha256"])
    directions, _ = document(parent["parent_reference"], parent["parent_reference_sha256"])
    if "extends_reference" in parent:
        document(parent["extends_reference"], parent["extends_reference_sha256"])
    pilot, _ = document(reference["pilot"], reference["pilot_sha256"])
    audit, _ = document(reference["pilot_audit"], reference["pilot_audit_sha256"])
    original, _ = document(audit["original"], audit["original_sha256"])
    assert audit["status"] == "passed" and audit["all_channels_accounted_for"]
    assert audit["report_sha256"] == reference["pilot_sha256"]
    assert pilot["status"] == "diagnostic_complete" and pilot["stable_cached_delta_validation"] == "passed"
    assert pilot["reference_sha256"] == reference["parent_reference_sha256"]
    assert pilot["channel_selection_from_sha256"] == audit["original_sha256"]
    for other in (parent, directions, pilot, original):
        assert reference["source"] == other["source"]
    for key in ("parameter_names", "initial"):
        assert reference[key] == parent[key] == directions[key]
    for key in ("combine_version", "root_version", "runtime_flags", "steps"):
        assert reference[key] == parent[key]
    assert len(reference["steps"]) == 12 and all(np.isfinite(h) and h > 0 for h in reference["steps"])
    assert reference["stable_cached_channels"] == pilot["stable_cached_channels"]
    assert reference["excluded_cached_channels"] == pilot["excluded_cached_channels"]
    stable = [c["index"] for c in reference["stable_cached_channels"]]
    excluded = [c["index"] for c in reference["excluded_cached_channels"]]
    assert len(stable) > 0 and sorted(stable+excluded) == list(range(len(pilot["channels"])))
    assert audit["stable_channels"] == len(stable)
    assert audit["excluded_channels"] == reference["excluded_cached_channels"]
    for value in (reference["canonical_nominal_nll"], reference["nominal_zero_point_nll"], reference["canonical_restoration_absolute_error"]):
        assert np.isfinite(value)
    assert abs(reference["canonical_nominal_nll"]-parent["canonical_nominal_nll"]) < 1e-6
    assert abs(reference["nominal_zero_point_nll"]) < 1e-6
    assert 0 <= reference["canonical_restoration_absolute_error"] < 1e-6
    assert len(reference["cases"]) == len(parent["cases"]) == len(directions["cases"]) == 18
    assert len({c["label"] for c in reference["cases"]}) == 18
    unresolved, errors = [], []
    for case, old, direction in zip(reference["cases"],parent["cases"],directions["cases"],strict=True):
        assert case["label"] == old["label"] == direction["label"]
        assert case["direction"] == old["direction"] == direction["direction"]
        assert len(case["direction"]) == len(reference["initial"]) == len(reference["parameter_names"])
        assert np.isfinite(case["direction"]).all() and np.isfinite(reference["initial"]).all()
        assert len(case["probes"]) == len(old["probes"]) == 12
        for probe, prior, step in zip(case["probes"],old["probes"],reference["steps"],strict=True):
            assert probe["step"] == prior["step"] == step
            assert probe["original_curvature"] == prior["curvature"]
            samples = probe["samples"]
            assert [s["multiple"] for s in samples] == [-2,-1,1,2]
            values = np.array([s["full_nll"] for s in samples])
            deltas = np.array([s["stable_delta"] for s in samples],dtype=np.longdouble)
            assert np.isfinite(values).all() and np.isfinite(deltas).all()
            error = float(np.max(abs(values-prior["zero_point_nll_values"])))
            assert probe["original_sample_max_absolute_error"] == error < 1e-6
            errors.append(error)
            for sample in samples:
                bound = sample["stable_channel_delta_max_absolute_error"]
                assert np.isfinite(bound) and 0 <= bound < 1e-6 and sample["roofit_evaluation_errors"] == 0
                correction = sample["stable_delta"]-(sample["full_nll"]-reference["nominal_zero_point_nll"])
                assert abs(correction) < 1e-6, "Stable full NLL exceeds the original NLL tolerance"
                rounding = 8*np.finfo(float).eps*max(1.,abs(sample["stable_delta"]),abs(sample["full_nll"]))
                assert abs(correction) <= len(stable)*bound+rounding, "Inconsistent full/channel correction"
            # Reconstruct from stored samples with the explicit five-point rule.
            measured = float((-deltas[0]+16*deltas[1]+16*deltas[2]-deltas[3])/(12*np.longdouble(step)**2))
            np.testing.assert_allclose(probe["curvature"],measured,rtol=1e-12,atol=1e-12)
        selected = first_converged_triple(case["probes"])
        assert case["selected_triple"] == selected, "Must use first converged triple"
        assert case["reference_status"] == ("converged" if selected else "unresolved")
        if selected is None:
            unresolved.append(case["label"])
        if case["label"] == pilot["label"]:
            difference = max(abs(a["curvature"]-b["stable_full_curvature"])
                             for a,b in zip(case["probes"],pilot["probes"],strict=True))
            assert case["pilot_curvature_max_absolute_error"] == difference
    assert reference["unresolved_directions"] == unresolved
    assert reference["status"] == ("reference_uncertain" if unresolved else "reference_ready")
    return reference, dict(status="verified", reference_status=reference["status"],
        scope="Provenance, channel partition, stored NLL samples, stencils and reference-only selection; unresolved references remain unresolved",
        reference=str(path),reference_sha256=checksum,source=reference["source"],
        parent_reference_sha256=reference["parent_reference_sha256"],pilot_sha256=reference["pilot_sha256"],
        pilot_audit_sha256=reference["pilot_audit_sha256"],directions=18,probes=216,samples=864,
        stable_channels=len(stable),excluded_channels=len(excluded),
        original_nll_max_absolute_error=max(errors),unresolved_directions=unresolved)


def audit_finer_reference(path, reference, checksum):
    """Audit one fixed extension; the original samples cannot be replaced."""
    assert reference["status"] in ("reference_ready", "reference_uncertain")
    base_path = reference["extends_cached_reference"]
    base, _ = document(base_path, reference["extends_cached_reference_sha256"])
    assert "extends_cached_reference" not in base, "Only the frozen first extension is supported"
    _, base_audit = audit_reference(base_path)
    for key in ("source", "parameter_names", "initial", "combine_version", "root_version", "runtime_flags",
                "parent_reference", "parent_reference_sha256", "pilot", "pilot_sha256",
                "pilot_audit", "pilot_audit_sha256", "stable_cached_channels", "excluded_cached_channels",
                "canonical_nominal_nll", "nominal_zero_point_nll"):
        assert reference[key] == base[key], key
    assert reference["steps"] == base["steps"]+FINER_STEPS
    assert len(reference["cases"]) == len(base["cases"]) == 18
    assert 0 <= reference["canonical_restoration_absolute_error"] < 1e-6
    stable_count = len(base["stable_cached_channels"])
    unresolved = []
    for case, old in zip(reference["cases"], base["cases"], strict=True):
        assert case["label"] == old["label"] and case["direction"] == old["direction"]
        assert len(case["probes"]) == 15 and case["probes"][:12] == old["probes"]
        anchor = case["repeated_anchor"]
        assert anchor["step"] == old["probes"][-1]["step"]
        for current, previous in zip(anchor["samples"], old["probes"][-1]["samples"], strict=True):
            for key in ("multiple", "full_nll", "stable_delta"):
                assert current[key] == previous[key], "Repeated anchor changed"
        for probe, step in zip([anchor]+case["probes"][12:], [base["steps"][-1]]+FINER_STEPS, strict=True):
            assert probe["step"] == step
            samples = probe["samples"]
            assert [s["multiple"] for s in samples] == [-2,-1,1,2]
            values = np.asarray([s["stable_delta"] for s in samples], dtype=np.longdouble)
            assert np.isfinite(values).all()
            for sample in samples:
                assert np.isfinite(sample["full_nll"]) and sample["roofit_evaluation_errors"] == 0
                bound = sample["stable_channel_delta_max_absolute_error"]
                assert np.isfinite(bound) and 0 <= bound < 1e-6
                correction = sample["stable_delta"]-(sample["full_nll"]-reference["nominal_zero_point_nll"])
                assert abs(correction) < 1e-6
                rounding = 8*np.finfo(float).eps*max(1.,abs(sample["stable_delta"]),abs(sample["full_nll"]))
                assert abs(correction) <= stable_count*bound+rounding
            measured = float(np.dot(values, [-1,16,16,-1])/(12*np.longdouble(step)**2))
            np.testing.assert_allclose(probe["curvature"], measured, rtol=1e-12, atol=1e-12)
        selected = first_converged_triple(case["probes"])
        assert case["selected_triple"] == selected
        assert case["reference_status"] == ("converged" if selected else "unresolved")
        if selected is None:
            unresolved.append(case["label"])
    assert reference["unresolved_directions"] == unresolved
    assert reference["status"] == ("reference_uncertain" if unresolved else "reference_ready")
    return reference, dict(status="verified", reference_status=reference["status"],
        scope="Fixed three-step extension of all 18 directions; unchanged original samples and exactly reproduced anchors; no AD consulted",
        reference=str(path),reference_sha256=checksum,source=reference["source"],
        base_audit=base_audit,directions=18,probes=270,samples=1080,new_samples=216,anchor_samples=72,
        stable_channels=stable_count,excluded_channels=len(base["excluded_cached_channels"]),
        original_nll_max_absolute_error=base_audit["original_nll_max_absolute_error"],
        unresolved_directions=unresolved)


def compare(benchmark_path, reference_path):
    reference, audit = audit_reference(reference_path)
    benchmark, checksum = document(benchmark_path)
    assert benchmark["source"] == reference["source"]
    assert "hessian" in benchmark["completed_operations"], "Complete Hessian required"
    artifact = Path(benchmark["arrays"])
    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == benchmark["arrays_sha256"]
    with np.load(artifact,allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["parameter_names"],reference["parameter_names"])
        np.testing.assert_array_equal(arrays["point"],reference["initial"])
        hessian = arrays["hessian"].copy()
    assert hessian.shape == (len(reference["initial"]),)*2 and np.isfinite(hessian).all()
    report = dict(benchmark=str(benchmark_path),benchmark_sha256=checksum,
        arrays_sha256=benchmark["arrays_sha256"],source=reference["source"],reference_audit=audit,
        policy="Compare every independently selected curvature in the first converged triple at relative tolerance 1e-4, denominator max(abs(reference),1e-8). Unresolved directions cannot pass.",
        scope="18 nominal projections of the complete saved Hessian; not all-entry Combine Hessian, fit or performance acceptance",
        hessian_max_asymmetry=float(np.max(abs(hessian-hessian.T))),cases=[])
    for case in reference["cases"]:
        direction = np.asarray(case["direction"])
        value = float(direction @ hessian @ direction)
        checks = [dict(step=p["step"],reference=p["curvature"],absolute_error=abs(value-p["curvature"]),
                       relative_error=abs(value-p["curvature"])/max(abs(p["curvature"]),1e-8)) for p in case["probes"]]
        selected = case["selected_triple"]
        status = "reference_uncertain"
        if selected:
            first = selected["first_probe_index"]
            status = "passed" if all(c["relative_error"] < 1e-4 for c in checks[first:first+3]) else "failed"
        report["cases"].append(dict(label=case["label"],ad_curvature=value,selected_triple=selected,status=status,probes=checks))
    report["failed_directions"] = [c["label"] for c in report["cases"] if c["status"] == "failed"]
    report["unresolved_directions"] = [c["label"] for c in report["cases"] if c["status"] == "reference_uncertain"]
    report["status"] = ("failed" if report["failed_directions"] else
                        "reference_uncertain" if report["unresolved_directions"] else "passed")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark",type=Path,help="Omit to audit the reference without reading a Hessian")
    parser.add_argument("--reference",type=Path,default=Path("reports/gpu_combine_curvature_cached.json"))
    parser.add_argument("--output",type=Path,required=True)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("Preserve prior curvature evidence")
    report = compare(arguments.benchmark,arguments.reference) if arguments.benchmark else audit_reference(arguments.reference)[1]
    with arguments.output.open("x") as stream:
        stream.write(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(arguments.output,report["status"])
    if arguments.benchmark and report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
