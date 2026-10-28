"""Synthetic provenance and failure checks for the explicit supplemental gate."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import build_cpu_hessian_reference as builder
from refine_cpu_hessian_fd import STEPS, first_converged_triple


@pytest.fixture
def evidence(tmp_path):
    hessian = np.array([[2., .5], [.5, 3.]])
    matrices = np.tile(hessian, (len(STEPS), 1, 1))
    matrices[0, 0, 0] += .2
    stencils = np.empty((len(STEPS), 2, 4, 2))
    for k, step in enumerate(STEPS):
        for column in range(2):
            stencils[k, column] = np.asarray((-2., -1., 1., 2.))[:, None]*step*matrices[k, :, column]
    original = tmp_path/"original.npz"
    common = dict(parameter_names=np.array(["a", "b"]), parameter_scales=np.ones(2),
                  free_indices=np.arange(2))
    np.savez(original, **common, points=np.zeros((2, 2)),
             original_hessians=np.tile(hessian, (2, 1, 1)),
             default_gradient_fd_hessians=np.tile(matrices[0], (2, 3, 1, 1)))
    case = dict(status="failed", index=2, name="synthetic", arrays=str(original),
                arrays_sha256=builder.digest(original),
                finite_difference_checks=[dict(passed=False), dict(passed=True)])
    metadata = dict(source={"synthetic": True}, source_sha256="synthetic",
                    jax_version="synthetic", jaxlib_version="synthetic",
                    float64=True, backend="cpu")
    gate = dict(metadata, status="failed", cases=[case])
    gate_path = tmp_path/"gate.json"
    gate_path.write_text(json.dumps(gate))
    artifact = tmp_path/"refinement.npz"
    np.savez(artifact, **common, point=np.zeros(2), steps=STEPS, fd_hessians=matrices,
             gradient_stencils=stencils, saved_ad_hessian=hessian)
    selection = first_converged_triple(matrices, np.ones(2))
    errors = [builder.scaled_error(hessian, m, np.ones(2)) for m in matrices]
    refinement = dict(metadata, status="passed", parent=str(gate_path),
        parent_sha256=builder.digest(gate_path),
        runner_sha256=builder.digest(Path(builder.__file__).with_name("refine_cpu_hessian_fd.py")),
        default_disabled_hlo_passes="", steps=list(STEPS),
        selection_completed_before_reading_ad_hessian=True,
        term_index=2, term_name="synthetic", point_index=0,
        parent_arrays=str(original), parent_arrays_sha256=builder.digest(original),
        arrays=str(artifact), arrays_sha256=builder.digest(artifact),
        selection=selection, ad_errors_at_every_step=errors,
        selected_ad_errors=[errors[i] for i in selection["indices"]])
    refinement_path = tmp_path/"refinement.json"
    return gate, gate_path, refinement, refinement_path


def test_explicit_gate_accepts_reconstructed_reference_and_retains_failed_parent(evidence):
    gate, gate_path, refinement, path = evidence
    path.write_text(json.dumps(refinement))
    before = gate_path.read_bytes()
    result = builder.validate_nominal_refinement(gate, gate_path, path)
    assert result["original_gate_status"] == "failed"
    assert result["selected_steps"] == list(STEPS[1:4])
    assert gate_path.read_bytes() == before


@pytest.mark.parametrize("fault", ["failed", "parent_hash", "source", "later_triple",
                                 "stencils", "ad_errors"])
def test_gate_rejects_inconsistent_or_reselected_evidence(evidence, fault):
    gate, gate_path, refinement, path = evidence
    if fault == "failed":
        refinement["status"] = "failed"
    elif fault == "parent_hash":
        refinement["parent_sha256"] = "changed"
    elif fault == "source":
        refinement["source_sha256"] = "changed"
    elif fault == "later_triple":
        refinement["selection"]["indices"] = [2, 3, 4]
    elif fault == "ad_errors":
        refinement["selected_ad_errors"][0] = .5
    elif fault == "stencils":
        with np.load(refinement["arrays"]) as arrays:
            payload = {k: arrays[k].copy() for k in arrays}
        payload["gradient_stencils"][2, 0, 0, 0] += .1
        np.savez(refinement["arrays"], **payload)
        refinement["arrays_sha256"] = builder.digest(refinement["arrays"])
    path.write_text(json.dumps(refinement))
    with pytest.raises(AssertionError):
        builder.validate_nominal_refinement(gate, gate_path, path)
