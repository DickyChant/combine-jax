"""Bind saved random-gradient validation to an exact benchmark artifact."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from gpu_random_refinements import load_refinements


def random_validation(path, benchmark, benchmark_hash, refined=False):
    suffix = "_refined" if refined else "_stable"
    check_path = path.with_name(path.stem+"_combine_random_gradients"+suffix+".json")
    raw = check_path.read_bytes()
    check = json.loads(raw)
    assert check["benchmark_sha256"] == benchmark_hash
    assert check["arrays_sha256"] == benchmark["arrays_sha256"]
    assert check["source"] == benchmark["source"]
    for key in ("reference", "stable_reference"):
        assert hashlib.sha256(Path(check[key]).read_bytes()).hexdigest() == check[key+"_sha256"]
    assert len(check["points"]) == 25
    provenance = None
    if refined:
        reference = json.loads(Path(check["reference"]).read_bytes())
        declared = check["refinements"]
        overrides, provenance = load_refinements(Path(declared["full_nll_reference"]),
            Path(declared["intensity_reference"]), reference, check["reference_sha256"])
        assert declared == provenance
        assert {Path(p).name for p in check["refinement_checker_sha256"]} == {
            "gpu_random_refinements.py", "fd_reference.py"}
        assert len({p["point_number"] for p in check["points"]}) == 25
        counts = Counter()
        used = set()
        for point in check["points"]:
            counts.update(point["status_counts"])
            assert sum(point["status_counts"].values()) == benchmark["parameters"]
            for case in point["refined_checks"]:
                key = point["point_number"], case["index"]
                assert key not in used and key in overrides
                override = overrides[key]
                assert point["attempt"] == override["point"]["attempt"]
                assert case["name"] == override["case"]["name"]
                assert case["reference_method"] == override["method"]
                assert case["refined_reference_status"] == override["reference_status"]
                used.add(key)
            if check["status"] == "passed":
                assert point["status"] == "passed" and not point["failures"] and not point["unresolved"]
                assert point["max_relative_error"] < 1e-4
                assert all(v is not None and v < 1e-6 for v in point["nll_absolute_errors"].values())
                assert all(c["status"] == "passed" and c["max_relative_error"] < 1e-4
                           for c in point["refined_checks"])
        assert dict(counts) == check["status_counts"] and used == set(overrides)
        if check["status"] == "passed":
            assert set(counts) <= {"passed", "fixed_bounds"}
    return dict(status=check["status"], validation=str(check_path),
                validation_sha256=hashlib.sha256(raw).hexdigest(),
                reference_sha256=check["reference_sha256"],
                stable_reference_sha256=check["stable_reference_sha256"],
                refinement_documents_sha256=None if provenance is None else provenance["documents_sha256"],
                status_counts=check["status_counts"])
