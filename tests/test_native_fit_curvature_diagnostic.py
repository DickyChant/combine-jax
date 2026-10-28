"""Reject unbound points/artifacts; retain the distinction from fit acceptance."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import diagnose_native_fit_curvature as diagnostic


@pytest.fixture
def native_point(tmp_path):
    arrays = tmp_path/"native.npz"
    point = np.array([1., 2., 3.])
    names = ["x", "y", "fixed"]
    np.savez_compressed(arrays, parameter_names=np.array(names), values=point)
    common = dict(model="counting", source={"sha256":"workspace", "configuration":{}},
        parameter_names=names, discrete_choices={}, channel_masks={},
        combine_version="11.0.0", root_version="6.40.00", runtime_flags={"flag":1})
    parent = dict(common, status="fit_incomplete", stage="finished", arrays=str(arrays),
        arrays_sha256=diagnostic.digest(arrays), canonical_initial_nll=10., canonical_fitted_nll=9.5,
        final=dict(values=point.tolist(), nll_zero_point=-.5, is_valid=False))
    parent_path = tmp_path/"parent.json"
    parent_path.write_text(json.dumps(parent))
    check = dict(common, status="point_checked", point_source="final", point=point.tolist(),
        parent=str(parent_path), parent_sha256=diagnostic.digest(parent_path), parent_status=parent["status"],
        parameter_bounds_checked=True, fixed_parameters_unchanged=True,
        canonical_candidate_nll=9.5, zero_point_candidate_nll=-.5,
        saved_zero_point_absolute_error=0., canonical_offset_absolute_error=0.,
        canonical_restoration_absolute_error=0., probes=[dict(label=label,nll=value,roofit_evaluation_errors=0)
            for label,value in [("canonical_initial",10.), ("zero_point_initial",0.),
                ("saved_point_zero_offset",-.5), ("saved_point_canonical",9.5), ("restored_initial",10.)]])
    path = tmp_path/"check.json"
    path.write_text(json.dumps(check))
    return path


def test_rejected_terminal_fit_can_be_diagnosed_without_promoting_it(native_point):
    check, parent, point = diagnostic.checked_native_point(native_point)
    assert parent["status"] == "fit_incomplete" and not parent["final"]["is_valid"]
    assert check["parent_status"] == "fit_incomplete"
    np.testing.assert_array_equal(point, [1., 2., 3.])


@pytest.mark.parametrize("change", ["running_check", "best_seen", "parent_hash", "array_hash",
    "running_parent", "unfinished_parent", "point", "names", "canonical_nll", "root_error",
    "duplicate_probe", "nominal", "bounds", "fixed", "stack", "mask", "array_dtype", "nonfinite"])
def test_unchecked_changed_or_incomplete_point_rejected(native_point, change):
    check = json.loads(native_point.read_bytes())
    parent_path = Path(check["parent"])
    parent = json.loads(parent_path.read_bytes())
    if change == "running_check": check["status"] = "running"
    elif change == "best_seen": check["point_source"] = "best_seen"
    elif change == "parent_hash": check["parent_sha256"] = "bad"
    elif change == "array_hash": parent["arrays_sha256"] = "bad"
    elif change == "running_parent": parent["status"] = check["parent_status"] = "running"
    elif change == "unfinished_parent": parent["stage"] = "hesse"
    elif change == "point": check["point"][0] += 1.
    elif change == "names": check["parameter_names"] = ["y", "x", "fixed"]
    elif change == "canonical_nll": check["canonical_candidate_nll"] += 1.
    elif change == "root_error": check["probes"][2]["roofit_evaluation_errors"] = 1
    elif change == "duplicate_probe": check["probes"][1] = check["probes"][0].copy()
    elif change == "nominal": parent["canonical_initial_nll"] += 1.
    elif change == "bounds": check["parameter_bounds_checked"] = False
    elif change == "fixed": check["fixed_parameters_unchanged"] = False
    elif change == "stack": parent["root_version"] = check["root_version"] = "6.38.00"
    elif change == "mask": check["channel_masks"] = {"channel":{"masked":True}}
    elif change in ("array_dtype", "nonfinite"):
        point = np.array([1., 2., 3.], dtype=np.float32 if change == "array_dtype" else np.float64)
        if change == "nonfinite": point[0] = np.nan
        np.savez_compressed(parent["arrays"], parameter_names=np.array(parent["parameter_names"]), values=point)
        parent["arrays_sha256"] = diagnostic.digest(parent["arrays"])
    parent_path.write_text(json.dumps(parent))
    if change != "parent_hash": check["parent_sha256"] = diagnostic.digest(parent_path)
    native_point.write_text(json.dumps(check))
    with pytest.raises(AssertionError):
        diagnostic.checked_native_point(native_point)


@pytest.fixture
def saved_blocks(tmp_path):
    point = np.array([.2, .4, .8])
    records = []
    for i, (indices, value, gradient) in enumerate([([0,1],3.,[2.,-3.]), ([1,2],-2.,[4.,7.])]):
        path = tmp_path/f"{i}.npz"
        np.savez_compressed(path, indices=np.array(indices), point=point[indices],
                            nll=value, gradient=np.array(gradient))
        records.append(dict(index=i, arrays=str(path), arrays_sha256=diagnostic.digest(path), parameters=2))
    return dict(terms=records, expected_terms=2), point


def test_reassembles_shared_parameter_gradients_and_offset(saved_blocks):
    report, point = saved_blocks
    result = diagnostic.reassemble_value_gradient(report, point, 101., np.array([2.,1.,7.]), 100.)
    assert all(result.values())


@pytest.mark.parametrize("change", ["incomplete", "checksum", "duplicate_index", "wrong_point", "gradient_shape", "missing_shared_term"])
def test_corrupted_or_incomplete_derivative_assembly_rejected(saved_blocks, change):
    report, point = saved_blocks
    gradient = np.array([2.,1.,7.])
    if change == "incomplete": report["terms"].pop()
    elif change == "checksum": report["terms"][0]["arrays_sha256"] = "changed"
    elif change == "missing_shared_term": gradient[1] = 4.
    else:
        record = report["terms"][0]
        with np.load(record["arrays"], allow_pickle=False) as a:
            arrays = {k:a[k] for k in a.files}
        if change == "duplicate_index": arrays["indices"] = np.array([0,0])
        elif change == "wrong_point": arrays["point"][0] += .1
        elif change == "gradient_shape": arrays["gradient"] = np.array([2.])
        np.savez_compressed(record["arrays"], **arrays)
        record["arrays_sha256"] = diagnostic.digest(record["arrays"])
    with pytest.raises(AssertionError):
        diagnostic.reassemble_value_gradient(report, point, 101., gradient, 100.)


def test_existing_output_rejected_before_loading_point(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("reports").mkdir()
    output = Path("reports/existing.json")
    output.write_text("preserve")
    monkeypatch.setattr(sys, "argv", ["diagnose_native_fit_curvature.py",
        "--point-check", "absent.json", "--output", str(output)])
    with pytest.raises(FileExistsError): diagnostic.main()
    assert output.read_text() == "preserve"
