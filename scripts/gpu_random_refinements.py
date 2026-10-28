"""Verify explicit follow-up references without consulting AD distances."""

import hashlib
import json
from pathlib import Path

import numpy as np

from fd_reference import estimates, first_consistent_triple


def checked_document(path, expected=None, documents=None):
    """Verify the saved reference/selection chain, including failed comparisons."""
    documents = {} if documents is None else documents
    path = str(path)
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert expected is None or digest == expected, f"Hash mismatch: {path}"
    if path in documents:
        return documents[path][0]
    document = json.loads(raw)
    documents[path] = (document, digest)
    for key in ("reference", "selection", "selection_from", "extends_reference",
                "refinement_plan", "parent_reference"):
        if key in document:
            checked_document(document[key], document[key+"_sha256"], documents)
    return document


def identity(case):
    return case["point_number"], case["index"]


def verify_full_nll(case, fractions):
    probes = case["probes"]
    assert [p["fraction"] for p in probes] == fractions
    selected = None
    for probe in probes:
        # The frozen follow-up grid contains only central stencils. Do not
        # silently reinterpret a future one-sided or clipped-bound reference.
        assert probe["stencil"] == "central" and probe["direction"] == 1
        assert probe["step"] == probe["fraction"]*case["scale"] > 0
        if probe["status"] != "valid":
            continue
        assert probe["roofit_evaluation_errors"] == 0
        values = np.array(probe["zero_point_nll_values"], dtype=np.longdouble)
        assert values.shape == (4,) and np.isfinite(values).all()
        derivative = float((8*(values[2]-values[1])-(values[3]-values[0]))/
                           (12*np.longdouble(probe["step"])))
        assert derivative == probe["derivative"], "Changed full-NLL stencil"
    for first in range(len(probes)-2):
        triple = probes[first:first+3]
        if not all(p["status"] == "valid" for p in triple):
            continue
        values = [p["derivative"] for p in triple]
        spread = (max(values)-min(values))/max(min(abs(v) for v in values), 1e-8)
        if spread < 1e-5:
            selected = first
            assert spread == case["selected_relative_spread"]
            break
    assert selected == case["selected_first_probe_index"], "Changed first triple"
    assert case["status"] == ("reference_ready" if selected is not None else "reference_uncertain")
    return [] if selected is None else probes[selected:selected+3]


def verify_local_intensity(case, parent):
    for key in ("point_number", "attempt", "name", "index", "theta", "coordinate", "scale"):
        assert case[key] == parent[key], f"Changed reference {key}"
    assert parent["canonical_parent_absolute_error"] < 1e-6
    assert parent["canonical_restoration_absolute_error"] < 1e-6
    assert len(case["probes"]) == len(parent["probes"])
    fractions = [p["fraction"] for p in parent["probes"]]
    assert fractions == sorted(set(fractions), reverse=True)
    for probe, original in zip(case["probes"], parent["probes"], strict=True):
        assert probe["fraction"] == original["fraction"]
        assert probe["step"] == original["step"] == probe["fraction"]*case["scale"] > 0
        checks = original["delta_checks"]
        assert [d["multiple"] for d in checks] == [-4,-3,-2,-1,1,2,3,4]
        for check in checks:
            assert check["coordinate"] == case["coordinate"]+check["multiple"]*probe["step"]
            assert check["roofit_evaluation_errors"] == 0
            error = abs(check["stable_delta"]-check["combine_delta"])
            assert error < 1e-6 and error == check["absolute_error"]
        derived = estimates({d["multiple"]:d["stable_delta"] for d in checks}, probe["step"])
        assert derived == probe["estimates"], "Changed local stencil"
    selected = first_consistent_triple(case["probes"])
    assert selected == case["selected_triple"], "Changed first triple"
    assert case["status"] == ("reference_ready" if selected is not None else "reference_uncertain")
    if selected is None:
        return []
    first = selected["first_probe_index"]
    return [dict(step=p["step"], estimator=name, derivative=value)
            for p in case["probes"][first:first+3] for name, value in p["estimates"].items()]


def load_refinements(full_path, intensity_path, reference, reference_sha256):
    documents = {}
    full = checked_document(full_path, documents=documents)
    local = checked_document(intensity_path, documents=documents)
    for document, _ in documents.values():
        for key in ("source", "combine_version", "root_version", "runtime_flags"):
            if key in document:
                assert document[key] == reference[key], f"Reference mismatch: {key}"
    assert full["reference_sha256"] == reference_sha256
    selection = documents[full["selection"]][0]
    assert selection["reference_sha256"] == reference_sha256
    assert full["fractions"] == selection["fractions"]
    selected = {identity(c):c for c in selection["cases"]}
    assert len(selected) == len(selection["cases"]) == 14
    failed = documents[selection["selection_from"]][0]
    assert selected == {(p["point_number"],c["index"]):
                        dict(point_number=p["point_number"], attempt=p["attempt"],
                             index=c["index"], name=c["name"])
                        for p in failed["points"] for c in p["failures"]}
    parent = documents[local["reference"]][0]
    assert parent["reference_sha256"] == documents[str(full_path)][1]
    intensity_selection = documents[parent["selection"]][0]
    assert intensity_selection["reference_sha256"] == documents[str(full_path)][1]
    intensity_selected = {identity(c):c for c in intensity_selection["cases"]}
    assert len(intensity_selected) == len(intensity_selection["cases"]) == 2
    assert intensity_selected.keys() <= selected.keys()
    diagnostic = documents[intensity_selection["selection_from"]][0]
    assert set(intensity_selected) == {identity(c) for c in diagnostic["cases"] if c["status"] != "passed"}
    # The dense extension must retain every old probe and exactly the declared
    # new fractions. The plan itself is bound to the original reference hash.
    original = documents[parent["extends_reference"]][0]
    plan = documents[parent["refinement_plan"]][0]
    assert plan["reference_sha256"] == parent["extends_reference_sha256"]
    originals = {identity(c):c for c in original["points"]}
    plans = {identity(c):c for c in plan["cases"]}
    parents = {identity(c):c for c in parent["points"]}
    locals_ = {identity(c):c for c in local["points"]}
    assert set(parents) == set(originals) == set(locals_) == set(intensity_selected)
    assert len(parents) == len(parent["points"]) == len(local["points"]) == 2
    assert set(plans) == {k for k,c in originals.items() if c["status"] == "reference_uncertain"}
    for key, case in parents.items():
        old = originals[key]
        if key not in plans:
            assert case == old
            continue
        old_probes = {p["fraction"]:p for p in old["probes"]}
        first = next(i for i,p in enumerate(old["probes"])
                     if (max(p["stable"].values())-min(p["stable"].values()))/
                     max(min(abs(v) for v in p["stable"].values()),1e-8) < 1e-5)
        start, stop = [p["fraction"] for p in old["probes"][first:first+2]]
        assert plans[key]["first_individually_consistent_probe"] == first
        assert plans[key]["interval_fractions"] == [start, stop]
        assert plan["subdivisions"] == 10
        assert plans[key]["added_fractions"] == [stop+(start-stop)*i/10 for i in range(9,0,-1)]
        assert [p for p in case["probes"] if p["fraction"] in old_probes] == old["probes"]
        assert [p["fraction"] for p in case["probes"] if p["fraction"] not in old_probes] == plans[key]["added_fractions"]
        assert case["added_fractions"] == plans[key]["added_fractions"]
    overrides = {}
    for point in full["points"]:
        assert point["canonical_parent_absolute_error"] < 1e-6
        assert point["canonical_restoration_absolute_error"] < 1e-6
        for case in point["cases"]:
            key = point["point_number"], case["index"]
            assert key not in overrides and key in selected
            chosen = verify_full_nll(case, full["fractions"])
            descriptor = selected[key]
            assert descriptor["attempt"] == point["attempt"] and descriptor["name"] == case["name"]
            method = "fixed-grid full Combine NLL; first three references with spread <1e-5"
            status = case["status"]
            if key in intensity_selected:
                assert intensity_selected[key] == descriptor
                intensity = locals_[key]
                assert all(intensity[k] == point[k] for k in ("point_number", "attempt", "theta"))
                assert all(intensity[k] == case[k] for k in ("name", "index", "coordinate", "scale"))
                chosen = verify_local_intensity(intensity, parents[key])
                status = intensity["status"]
                method = "stable ROOT continuous intensities; first twelve short-stencil estimates with spread <1e-5"
            overrides[key] = dict(point=point, case=case, chosen=chosen,
                                  reference_status=status, method=method)
    assert set(overrides) == set(selected)
    provenance = dict(full_nll_reference=str(full_path), intensity_reference=str(intensity_path),
                      documents_sha256={p:entry[1] for p,entry in documents.items()},
                      selection_scope="14 cases selected after raw failures; 12 fixed-grid full-NLL and two stable-intensity references; selections are frozen and never depend on this benchmark's AD distances")
    return overrides, provenance
