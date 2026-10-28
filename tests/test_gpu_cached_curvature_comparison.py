"""Synthetic quartic/cusp references test complete-Hessian acceptance and provenance."""

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_gpu_cached_curvature import FINER_STEPS, audit_reference, compare, main
from compare_gpu_curvature import first_converged_triple


def write(path, data):
    path.write_text(json.dumps(data))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def evidence(tmp_path):
    source = dict(synthetic_polynomial=True)
    common = dict(source=source,parameter_names=["x","y"],initial=[0.,0.],
                  combine_version="synthetic",root_version="synthetic",runtime_flags={})
    vectors = np.random.default_rng(821).normal(size=(18,2))
    vectors[0] = [1.,0.]
    steps = [1e-2,5e-3,2e-3,1e-3,5e-4,2e-4,1e-4,5e-5,2e-5,1e-5,5e-6,2e-6]
    directions = dict(common,cases=[dict(label=str(i),direction=d.tolist()) for i,d in enumerate(vectors)])
    directions_path = tmp_path/"directions.json"
    directions_hash = write(directions_path,directions)
    parent = dict(common,status="reference_ready",steps=steps,canonical_nominal_nll=0.,
                  parent_reference=str(directions_path),parent_reference_sha256=directions_hash,cases=[])
    cases = []
    for i,d in enumerate(vectors):
        case = dict(label=str(i),direction=d.tolist(),probes=[])
        old = dict(label=str(i),direction=d.tolist(),probes=[])
        for step in steps:
            xy = np.array([-2,-1,1,2],dtype=np.longdouble)[:,None]*np.longdouble(step)*d
            x,y = xy.T
            values = np.asarray(7*x+11*y+x*x+3*x*y+5*y*y+x**4+y**4,dtype=float)
            # Independent polynomial values, before any comparator code runs.
            curve = float(np.dot(values.astype(np.longdouble),[-1,16,16,-1])/(12*np.longdouble(step)**2))
            old["probes"].append(dict(step=step,curvature=curve,zero_point_nll_values=values.tolist()))
            case["probes"].append(dict(step=step,curvature=curve,original_curvature=curve,
                original_sample_max_absolute_error=0.,samples=[dict(multiple=k,full_nll=float(v),stable_delta=float(v),
                    stable_channel_delta_max_absolute_error=0.,roofit_evaluation_errors=0)
                    for k,v in zip((-2,-1,1,2),values,strict=True)]))
        case.update(selected_triple=first_converged_triple(case["probes"]),reference_status="converged")
        assert case["selected_triple"] is not None
        cases.append(case);parent["cases"].append(old)
    parent_path = tmp_path/"parent.json";parent_hash=write(parent_path,parent)
    original_path = tmp_path/"original.json";original_hash=write(original_path,dict(source=source))
    stable = [dict(index=0,channel="stable")]
    excluded = [dict(index=1,channel="retained",reason="non-RooAddPdf")]
    pilot = dict(source=source,status="diagnostic_complete",stable_cached_delta_validation="passed",
        reference_sha256=parent_hash,channel_selection_from_sha256=original_hash,
        stable_cached_channels=stable,excluded_cached_channels=excluded,channels=[{},{}],label="7",
        probes=[dict(stable_full_curvature=p["curvature"]) for p in cases[7]["probes"]])
    pilot_path=tmp_path/"pilot.json";pilot_hash=write(pilot_path,pilot)
    audit = dict(status="passed",all_channels_accounted_for=True,report_sha256=pilot_hash,
        original=str(original_path),original_sha256=original_hash,stable_channels=1,excluded_channels=excluded)
    audit_path=tmp_path/"audit.json";audit_hash=write(audit_path,audit)
    reference = dict(common,status="reference_ready",steps=steps,canonical_nominal_nll=0.,nominal_zero_point_nll=0.,
        canonical_restoration_absolute_error=0.,parent_reference=str(parent_path),parent_reference_sha256=parent_hash,
        pilot=str(pilot_path),pilot_sha256=pilot_hash,pilot_audit=str(audit_path),pilot_audit_sha256=audit_hash,
        stable_cached_channels=stable,excluded_cached_channels=excluded,cases=cases,unresolved_directions=[])
    cases[7]["pilot_curvature_max_absolute_error"] = 0.
    reference_path=tmp_path/"reference.json";write(reference_path,reference)
    arrays=tmp_path/"arrays.npz"
    np.savez(arrays,parameter_names=["x","y"],point=[0.,0.],hessian=np.array([[2.,3.],[3.,10.]]))
    benchmark = dict(source=source,completed_operations=["value","value_and_grad","hessian"],arrays=str(arrays),
                     arrays_sha256=hashlib.sha256(arrays.read_bytes()).hexdigest())
    benchmark_path=tmp_path/"benchmark.json";write(benchmark_path,benchmark)
    return reference_path,benchmark_path


@pytest.mark.parametrize("mutation", ("none","running","hash","direction","coverage","sample",
    "stencil","selection","roofit","delta_bound","negative_bound","missing_case","restoration","pilot_difference"))
def test_reference_audit(evidence, mutation):
    path,_ = evidence
    r=json.loads(path.read_text());c=r["cases"][0];p=c["probes"][0]
    if mutation=="running":r["status"]="running"
    elif mutation=="hash":r["parent_reference_sha256"]="different"
    elif mutation=="direction":c["direction"][0]=2.
    elif mutation=="coverage":r["excluded_cached_channels"][0]["index"]=0
    elif mutation=="sample":p["samples"][0]["full_nll"]+=1e-5
    elif mutation=="stencil":p["curvature"]+=.01
    elif mutation=="selection":
        selected=first_converged_triple(c["probes"][1:]);selected["first_probe_index"]+=1;c["selected_triple"]=selected
    elif mutation=="roofit":p["samples"][0]["roofit_evaluation_errors"]=1
    elif mutation=="delta_bound":p["samples"][0]["stable_delta"]+=1e-7
    elif mutation=="negative_bound":p["samples"][0]["stable_channel_delta_max_absolute_error"]=-1e-9
    elif mutation=="missing_case":r["cases"].pop()
    elif mutation=="restoration":r["canonical_restoration_absolute_error"]=1e-3
    elif mutation=="pilot_difference":r["cases"][7]["pilot_curvature_max_absolute_error"]=1.
    write(path,r)
    if mutation=="none":
        _,audit=audit_reference(path)
        assert audit["status"]=="verified" and audit["samples"]==864 and audit["unresolved_directions"]==[]
    else:
        with pytest.raises(AssertionError):audit_reference(path)


def test_consistent_metadata_cannot_hide_double_counted_channel(evidence):
    path,_=evidence
    r=json.loads(path.read_text());pilot_path=Path(r["pilot"]);audit_path=Path(r["pilot_audit"])
    pilot=json.loads(pilot_path.read_text());audit=json.loads(audit_path.read_text())
    # Update all metadata and hashes together: channel accounting itself must
    # reject a duplicated channel and the corresponding omitted contribution.
    for report in (r,pilot,audit):report["excluded_channels" if report is audit else "excluded_cached_channels"][0]["index"]=0
    r["pilot_sha256"]=write(pilot_path,pilot);audit["report_sha256"]=r["pilot_sha256"]
    r["pilot_audit_sha256"]=write(audit_path,audit);write(path,r)
    with pytest.raises(AssertionError):audit_reference(path)


@pytest.mark.parametrize("outcome", ("passed","failed","reference_uncertain","missing_hessian","arrays_hash"))
def test_complete_hessian_comparison(evidence,outcome):
    path,benchmark_path=evidence
    b=json.loads(benchmark_path.read_text())
    if outcome=="failed":
        np.savez(b["arrays"],parameter_names=["x","y"],point=[0.,0.],hessian=1.01*np.array([[2.,3.],[3.,10.]]))
        b["arrays_sha256"]=hashlib.sha256(Path(b["arrays"]).read_bytes()).hexdigest()
    elif outcome=="missing_hessian":b["completed_operations"].remove("hessian")
    elif outcome=="arrays_hash":b["arrays_sha256"]="different"
    elif outcome=="reference_uncertain":
        r=json.loads(path.read_text());c=r["cases"][0]
        for probe in c["probes"]:
            step=probe["step"]
            for sample in probe["samples"]:
                cusp=3e-5*abs(sample["multiple"]*step)
                sample["stable_delta"]+=cusp
                sample["stable_channel_delta_max_absolute_error"]=cusp
            values=np.array([s["stable_delta"] for s in probe["samples"]],dtype=np.longdouble)
            probe["curvature"]=float(np.dot(values,[-1,16,16,-1])/(12*np.longdouble(step)**2))
        assert first_converged_triple(c["probes"]) is None
        c.update(reference_status="unresolved",selected_triple=None)
        r.update(status="reference_uncertain",unresolved_directions=["0"]);write(path,r)
    write(benchmark_path,b)
    if outcome in ("missing_hessian","arrays_hash"):
        with pytest.raises(AssertionError):compare(benchmark_path,path)
    else:
        report=compare(benchmark_path,path)
        assert report["status"]==outcome and len(report["cases"])==18
        if outcome=="reference_uncertain":assert report["unresolved_directions"]==["0"]
        if outcome=="failed":assert len(report["failed_directions"])==18


def test_cli_preserves_previous_evidence(evidence,tmp_path,monkeypatch):
    reference,benchmark=evidence
    output=tmp_path/"comparison.json"
    monkeypatch.setattr(sys,"argv",["compare_gpu_cached_curvature.py","--benchmark",str(benchmark),
                                  "--reference",str(reference),"--output",str(output)])
    main();saved=output.read_bytes()
    assert json.loads(saved)["status"]=="passed"
    with pytest.raises(FileExistsError):main()
    assert output.read_bytes()==saved


def extended_reference(path):
    base=json.loads(path.read_text())
    result=copy.deepcopy(base)
    result.update(extends_cached_reference=str(path),
                  extends_cached_reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  steps=base["steps"]+FINER_STEPS)
    for case in result["cases"]:
        case["repeated_anchor"]=copy.deepcopy(case["probes"][-1])
        for step in FINER_STEPS:
            xy=np.array([-2,-1,1,2],dtype=np.longdouble)[:,None]*np.longdouble(step)*case["direction"]
            x,y=xy.T
            values=np.asarray(7*x+11*y+x*x+3*x*y+5*y*y+x**4+y**4,dtype=float)
            curve=float(np.dot(values.astype(np.longdouble),[-1,16,16,-1])/(12*np.longdouble(step)**2))
            case["probes"].append(dict(step=step,curvature=curve,
                samples=[dict(multiple=k,full_nll=float(v),stable_delta=float(v),
                              stable_channel_delta_max_absolute_error=0.,roofit_evaluation_errors=0)
                         for k,v in zip((-2,-1,1,2),values,strict=True)]))
        case["selected_triple"]=first_converged_triple(case["probes"])
    return result


@pytest.mark.parametrize("mutation", ("none","base_hash","original_sample","anchor","step_grid",
    "new_stencil","new_delta","new_error","selection","coverage","missing_case","unfinished"))
def test_fixed_finer_extension(evidence,mutation):
    base,benchmark=evidence
    r=extended_reference(base)
    if mutation=="base_hash":r["extends_cached_reference_sha256"]="different"
    elif mutation=="original_sample":r["cases"][0]["probes"][0]["curvature"]+=1.
    elif mutation=="anchor":r["cases"][0]["repeated_anchor"]["samples"][0]["full_nll"]+=1e-12
    elif mutation=="step_grid":r["steps"][-1]=1e-7
    elif mutation=="new_stencil":r["cases"][0]["probes"][-1]["curvature"]+=1.
    elif mutation=="new_delta":r["cases"][0]["probes"][-1]["samples"][0]["stable_delta"]+=1e-7
    elif mutation=="new_error":r["cases"][0]["probes"][-1]["samples"][0]["roofit_evaluation_errors"]=1
    elif mutation=="selection":r["cases"][0]["selected_triple"]=None
    elif mutation=="coverage":r["stable_cached_channels"][0]["index"]=1
    elif mutation=="missing_case":r["cases"].pop()
    elif mutation=="unfinished":r["status"]="running"
    output=base.with_name("finer.json");write(output,r)
    if mutation=="none":
        _,audit=audit_reference(output)
        assert audit["status"]=="verified" and audit["samples"]==1080
        assert audit["new_samples"]==216 and audit["anchor_samples"]==72
        assert compare(benchmark,output)["status"]=="passed"
    else:
        with pytest.raises(AssertionError):audit_reference(output)


def add_cusp(case, probes):
    for probe in probes:
        for sample in probe["samples"]:
            delta=3e-5*abs(sample["multiple"]*probe["step"])
            sample["stable_delta"]+=delta
            sample["stable_channel_delta_max_absolute_error"]=delta
        values=np.asarray([s["stable_delta"] for s in probe["samples"]],dtype=np.longdouble)
        probe["curvature"]=float(np.dot(values,[-1,16,16,-1])/(12*np.longdouble(probe["step"])**2))
    case.update(selected_triple=first_converged_triple(case["probes"]),reference_status="unresolved")
    assert case["selected_triple"] is None


def test_finer_samples_do_not_accept_unresolved_curvature(evidence):
    base,benchmark=evidence
    original=json.loads(base.read_text())
    add_cusp(original["cases"][0],original["cases"][0]["probes"])
    original.update(status="reference_uncertain",unresolved_directions=["0"])
    write(base,original)
    result=extended_reference(base)
    add_cusp(result["cases"][0],result["cases"][0]["probes"][12:])
    output=base.with_name("finer_cusp.json");write(output,result)
    _,audit=audit_reference(output)
    assert audit["reference_status"]=="reference_uncertain"
    assert compare(benchmark,output)["unresolved_directions"]==["0"]
