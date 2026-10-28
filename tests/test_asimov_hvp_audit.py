"""Selected-product acceptance must not hide provenance or native-gate failures."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_hvps import compare,checked_hvps
from tensor_benchmark_common import save_json,sha
from test_asimov_hessian_audit import evidence


@pytest.fixture
def hvp_evidence(evidence,tmp_path):
    original,info,hashes,structure=evidence
    source=json.loads(original['cpu'].read_text())
    with np.load(source['arrays']) as saved:data={k:saved[k] for k in saved.files}
    directions=np.array([[.1,0.,0.],[0.,0.,10.],[0.,1.,0.],[.025,0.,-7.5]])
    path=tmp_path/'directions.npz';np.savez(path,directions=directions,parameter_names=data['parameter_names'])
    numeric=tmp_path/'hvps.npz'
    np.savez(numeric,points=data['points'],parameter_names=data['parameter_names'],directions=directions,
        values=np.repeat(data['values'],4),gradients=np.repeat(data['gradients'],4,axis=0),
        hvps=directions@data['hessians'][0].T)
    report=tmp_path/'hvps.json'
    source.update(operation='hvp',directions=4,directions_source_sha256=sha(path),
        arrays=str(numeric),arrays_sha256=sha(numeric),batch_size=1)
    save_json(report,source)
    args=dict(manifest=original['manifest'],benchmark=report,directions=path,
        hessian=original['benchmark'],native_gate=original['native_gate'])
    return args,info,hashes,structure


def test_full_audit_keeps_selected_products_and_native_gates_separate(hvp_evidence):
    args,_,_,_=hvp_evidence;r=compare(**args)
    assert r['status']=='passed_selected_products_and_nominal_native_gates'
    assert r['selected_products']['compared_free_entries']==8
    assert len(r['nominal_native_comparison']['benchmark_directions'])==4
    assert not r['covariance_certified']


@pytest.mark.parametrize('fault',['directions','point','float32','nan','fixed_row','fixed_only','shape'])
def test_invalid_hvp_captures_rejected(hvp_evidence,fault):
    args,info,hashes,structure=hvp_evidence;path=args['benchmark'];r=json.loads(path.read_text())
    with np.load(r['arrays']) as saved:data={k:saved[k] for k in saved.files}
    if fault=='directions':data['directions'][0,0]+=.1
    elif fault=='point':data['points'][0,0]+=.1
    elif fault=='float32':data['hvps']=data['hvps'].astype(np.float32)
    elif fault=='nan':data['hvps'][0,0]=np.nan
    elif fault=='fixed_row':data['hvps'][0,1]=1e-20
    elif fault=='fixed_only':data['hvps'][2,0]=1e-20
    else:data['hvps']=data['hvps'][:2]
    np.savez(r['arrays'],**data);r['arrays_sha256']=sha(r['arrays']);save_json(path,r)
    with pytest.raises(AssertionError):checked_hvps(path,args['directions'],info,hashes,structure)


@pytest.mark.parametrize('field,increment',[('values',1.1e-6),('gradients',.001),('hvps',1.1e-5)])
def test_error_in_any_direction_fails_even_when_other_gates_pass(hvp_evidence,field,increment):
    args,_,_,_=hvp_evidence;path=args['benchmark'];r=json.loads(path.read_text())
    with np.load(r['arrays']) as saved:data={k:saved[k] for k in saved.files}
    if field=='values':data[field][-1]+=increment
    else:data[field][-1,0]+=increment
    np.savez(r['arrays'],**data);r['arrays_sha256']=sha(r['arrays']);save_json(path,r)
    result=compare(**args);assert result['status']=='failed'
    if field=='hvps':assert not result['selected_products']['passed']
    else:assert not result['nominal_native_comparison']['benchmark_directions'][-1]['passed']


def test_observed_manifest_rejected_before_reading_any_captures(tmp_path):
    path=tmp_path/'manifest.json';save_json(path,dict(hhvv=dict(dataset='data_obs')))
    with pytest.raises(ValueError,match='pre-fit Asimov'):
        compare(manifest=path,benchmark='unread',directions='unread',hessian='unread',native_gate='unread')
