"""Reject altered reference provenance, stencil arithmetic and partial agreement."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from gpu_random_refinements import (checked_document, load_refinements,
                                    verify_full_nll, verify_local_intensity)
from compare_gpu_random_gradients import compare_refined_coordinate


@pytest.fixture(scope="module")
def frozen():
    raw = Path("reports/gpu_combine_random_gradient_extended.json").read_bytes()
    cases, provenance = load_refinements(Path("reports/gpu_random_gradient_failure_fd.json"),
        Path("reports/gpu_random_intensity_fd_dense_local.json"),
        json.loads(raw), hashlib.sha256(raw).hexdigest())
    assert len(cases) == 14 and len(provenance["documents_sha256"]) == 12
    return cases


def test_frozen_references_require_all_selected_estimates(frozen):
    assert sum(len(c["chosen"]) == 3 for c in frozen.values()) == 12
    assert sum(len(c["chosen"]) == 12 for c in frozen.values()) == 2
    for refinement in frozen.values():
        derivative = np.mean([p["derivative"] for p in refinement["chosen"]])
        point, case = refinement["point"], refinement["case"]["original_reference"]
        assert compare_refined_coordinate(point,case,refinement,derivative)["status"] == "passed"
        assert compare_refined_coordinate(point,case,refinement,derivative*1.01)["status"] == "failed"


@pytest.mark.parametrize("count", (3,12))
def test_agreement_with_only_some_estimates_cannot_pass(frozen, count):
    refinement = deepcopy(next(c for c in frozen.values() if len(c["chosen"]) == count))
    for probe in refinement["chosen"]:
        probe["derivative"] = 100.
    refinement["chosen"][-1]["derivative"] = 100.0009
    result = compare_refined_coordinate(refinement["point"], refinement["case"]["original_reference"],
                                        refinement, 99.9904)
    assert result["status"] == "failed"
    assert result["selected_references"][0]["relative_error"] < 1e-4
    assert result["selected_references"][-1]["relative_error"] > 1e-4


@pytest.mark.parametrize("derivative,status", [(1.,"reference_uncertain"),(np.nan,"failed")])
def test_uncertain_and_nonfinite_cannot_pass(frozen, derivative, status):
    refinement = deepcopy(next(iter(frozen.values())))
    refinement.update(reference_status="reference_uncertain", chosen=[])
    result = compare_refined_coordinate(refinement["point"], refinement["case"]["original_reference"],
                                        refinement, derivative)
    assert result["status"] == status


@pytest.mark.parametrize("field", ("step", "derivative", "selected_first_probe_index"))
def test_altered_full_nll_reference_rejected(frozen, field):
    case = deepcopy(frozen[(2,756)]["case"])
    fractions = [p["fraction"] for p in case["probes"]]
    if field == "selected_first_probe_index":
        case[field] += 1
    else:
        case["probes"][0][field] *= 1.01
    with pytest.raises(AssertionError):
        verify_full_nll(case, fractions)


@pytest.mark.parametrize("field", ("step", "estimate", "selected_triple", "theta", "delta"))
def test_altered_intensity_reference_rejected(field):
    case = json.loads(Path("reports/gpu_random_intensity_fd_dense_local.json").read_text())["points"][0]
    parent = json.loads(Path("reports/gpu_random_intensity_fd_dense.json").read_text())["points"][0]
    if field == "step":
        case["probes"][0]["step"] *= 1.01
    elif field == "estimate":
        case["probes"][0]["estimates"]["forward_two"] *= 1.01
    elif field == "selected_triple":
        case[field]["first_probe_index"] += 1
    elif field == "theta":
        case[field][0] += .1
    else:
        parent["probes"][0]["delta_checks"][0]["combine_delta"] += .001
    with pytest.raises(AssertionError):
        verify_local_intensity(case, parent)


def test_changed_linked_document_is_rejected(tmp_path):
    child = tmp_path/"reference.json"
    child.write_text('{"source":"original"}')
    parent = tmp_path/"selection.json"
    parent.write_text(json.dumps(dict(reference=str(child),
        reference_sha256=hashlib.sha256(child.read_bytes()).hexdigest())))
    checked_document(parent)
    child.write_text('{"source":"altered"}')
    with pytest.raises(AssertionError, match="Hash mismatch"):
        checked_document(parent)


def test_changed_original_selection_cannot_be_substituted(frozen):
    refinement = deepcopy(frozen[(2,756)])
    case = deepcopy(refinement["case"]["original_reference"])
    case["index"] += 1
    with pytest.raises(AssertionError):
        compare_refined_coordinate(refinement["point"],case,refinement,1.)
