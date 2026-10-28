"""Profile acceptance uses all points, fixed coordinates, NLLs and POI shifts."""

import copy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import compare_profile_points as checker


def fixture():
    plan = dict(model="synthetic", source={"synthetic": True}, host="synthetic-host",
                parameter_names=["nuisance", "poi", "fixed"], parameter="nuisance", poi="poi",
                parameter_index=0, poi_index=1, discrete_choices={}, channel_masks={},
                initial=[0., 10., 7.], baseline_errors=[1., 2., .1], canonical_baseline_nll=2.,
                fixed_bounds=["fixed"], runner_sha256="synthetic",
                points=[dict(value=x, offset_sigma=x) for x in [-1., 0., 1.]])
    arrays = dict(parameter_names=np.asarray(plan["parameter_names"]), initial=np.asarray(plan["initial"]),
                  fixed_values=np.asarray([-1., 0., 1.]),
                  values=np.asarray([[-1., 10.2, 7.], [0., 10., 7.], [1., 9.8, 7.]]),
                  canonical_nll=np.asarray([3., 2., 3.]))
    return plan, arrays


def test_every_profile_point_and_poi_shift_passes():
    plan, arrays = fixture()
    points = checker.compare_arrays(arrays, copy.deepcopy(arrays), plan)
    assert len(points) == 3 and all(p["status"] == "passed" for p in points)
    assert points[1]["delta_nll_native"] == 0.
    assert points[1]["poi_shift_limit"] == .02


@pytest.mark.parametrize("fault", ["nll", "coordinate", "shift"])
def test_profile_disagreements_cannot_hide_in_aggregate(fault):
    plan, native = fixture()
    jax = copy.deepcopy(native)
    if fault == "nll":
        jax["canonical_nll"][0] += 2e-6
    elif fault == "coordinate":
        jax["values"][0, 1] += 1.
    else:
        # Within the coordinate's 1% limit but outside the POI-shift limit.
        jax["values"][0, 1] += .05
    points = checker.compare_arrays(jax, native, plan)
    assert points[0]["status"] == "failed"
    assert all(p["status"] == "passed" for p in points[1:])
    if fault == "shift":
        assert not points[0]["failed_coordinates"]
        assert points[0]["poi_shift_absolute_error"] > points[0]["poi_shift_limit"]


@pytest.mark.parametrize("fault", [None, "checksum", "fixed_nuisance", "fixed_bound", "point_count",
                                 "failed_fit", "delta", "poi", "nonfinite", "order"])
def test_saved_profile_integrity(tmp_path, fault):
    plan, arrays = fixture()
    report = {key: copy.deepcopy(plan[key]) for key in
              ("model", "source", "host", "parameter_names", "parameter", "poi",
               "discrete_choices", "channel_masks", "initial")}
    report.update(status="profile_computed", engine="cpu", plan_helper_sha256="synthetic",
                  baseline_nll=2., baseline_nll_absolute_error=0., points=[])
    for i, p in enumerate(plan["points"]):
        report["points"].append(dict(index=i, **p, valid_minimum=True,
            canonical_nll=float(arrays["canonical_nll"][i]),
            delta_nll=float(arrays["canonical_nll"][i]-2.),
            poi_value=float(arrays["values"][i, 1]), poi_shift=float(arrays["values"][i, 1]-10.)))
    if fault == "fixed_nuisance":
        arrays["values"][0, 0] = -.9
    elif fault == "fixed_bound":
        arrays["values"][1, 2] = 8.
    elif fault == "point_count":
        report["points"].pop()
    elif fault == "failed_fit":
        report["points"][0]["valid_minimum"] = False
    elif fault == "delta":
        report["points"][0]["delta_nll"] += .1
    elif fault == "poi":
        report["points"][0]["poi_shift"] += .1
    elif fault == "nonfinite":
        arrays["values"][0, 1] = np.nan
    elif fault == "order":
        arrays["parameter_names"] = arrays["parameter_names"][::-1]
    path = tmp_path/"arrays.npz"
    np.savez_compressed(path, **arrays)
    report.update(arrays=str(path), arrays_sha256=checker.digest(path))
    if fault == "checksum":
        path.write_bytes(b"tampered")
    if fault is None:
        result = checker.load_profile(report, plan)
        np.testing.assert_array_equal(result["values"], arrays["values"])
    else:
        with pytest.raises(AssertionError):
            checker.load_profile(report, plan)
