import hashlib
import json

import numpy as np
import pytest

from combine_jax.scan_recovery import load_previous_scan,recover_scan,recover_global_fit,summarize_recovery


def previous():
    return [dict(task='old_'+str(x),fixed={'kl':x},value=v,status=0,values=np.array([x,v]),origin='original')
            for x,v in [(1.,5.),(2.,0.),(2.5,1.),(3.,2.),(3.5,3.)]]


def test_recovery_starts_at_lowest_conditional_and_walks_neighbours_both_ways():
    calls=[]
    def fit(label,start,fixed):
        calls.append((label,start.copy(),fixed))
        value=-1. if not fixed else start[1]+.1
        return dict(status=0,value=value,values=np.array([fixed.get('kl',2.),value]))
    r=recover_scan(fit,previous(),poi='kl',minimum=2.5)
    assert calls[0][0]=='recovery_refit';np.testing.assert_array_equal(calls[0][1],[2.,0.])
    assert [c[2].get('kl') for c in calls]==[None,2.5,3.,3.5,3.5,3.,2.5]
    assert [c[1][0] for c in calls[1:]]==[2.,2.5,3.,3.5,3.5,3.]
    assert r['status']=='shape_study_requires_native_comparison'
    assert r['selected'][0]['origin']=='original'
    assert r['selected'][2]['origin']=='ascending_neighbour'
    assert not r['agreement_claimed']
    summary=summarize_recovery(r)
    assert summary['shape_only']['points'][1]==[2.,0.]
    assert summary['baseline_check']['global_value']==-1.
    assert all('values' not in row for row in summary['selected'])


def test_lower_conditional_triggers_another_unrestricted_fit_and_failed_attempt_is_retained():
    def fit(label,start,fixed):
        value=-2. if label=='recovery_refit_final' else (-1. if fixed else 0.)
        return dict(status=3 if label=='down_000' else 0,value=value,values=np.array([fixed.get('kl',2.),value]))
    r=recover_scan(fit,previous(),poi='kl',minimum=2.5)
    assert r['global_fit']['task']=='recovery_refit_final'
    assert r['baseline_check']['valid']
    assert r['status']=='recovery_attempt_failures'
    assert any(a['status']==3 for a in r['attempts'])
    assert all(a['status']==0 for a in r['selected'])


def test_failed_recovery_global_does_not_launch_conditional_fits():
    def fit(label,start,fixed):return dict(status=1,value=1.,values=start)
    r=recover_scan(fit,previous(),poi='kl',minimum=2.5)
    assert r['status']=='recovery_global_not_converged' and len(r['attempts'])==1
    assert not summarize_recovery(r)['baseline_check']['valid']


def test_restart_requires_identical_data_model_and_capture(tmp_path):
    path=tmp_path/'fit_report.json';capture=tmp_path/'point.npz'
    np.savez(capture,parameter_names=['kl','nuisance'],values=[2.,.3],value=12.,status=0)
    report=dict(model_sha256={'model':'modelhash'},dataset_identity={'sha256':'datahash'},dataset='data_obs',
        dataset_selection={'analysis':'hig23006'},fits=[dict(task='point',fixed={'kl':2.},value=12.,status=0,
        capture=capture.name,capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest())])
    path.write_text(json.dumps(report));kwargs=dict(model_sha256=report['model_sha256'],
        dataset_identity=report['dataset_identity'],dataset='data_obs',analysis='hig23006',parameter_names=['kl','nuisance'])
    rows,proof=load_previous_scan(path,**kwargs);assert proof['captures_verified']==1
    np.testing.assert_array_equal(rows[0]['values'],[2.,.3])
    with pytest.raises(ValueError,match='provenance'):load_previous_scan(path,**{**kwargs,'analysis':'hhvv'})
    with pytest.raises(ValueError,match='bindings'):load_previous_scan(path,**{**kwargs,'parameter_names':['nuisance','kl']})
    capture.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):load_previous_scan(path,**kwargs)


def test_global_only_recovery_preserves_every_conditional_and_runs_one_fit():
    original=previous();calls=[]
    def fit(label,start,fixed):
        calls.append((label,start.copy(),fixed))
        return dict(status=0,value=-.1,values=np.array([2.1,-.1]))
    result=recover_global_fit(fit,original,poi='kl')
    assert len(calls)==1 and calls[0][2]=={}
    np.testing.assert_array_equal(calls[0][1],[2.,0.])
    assert all(a is b for a,b in zip(result['selected'],original,strict=True))
    assert result['conditional_fits_rerun']==0
    assert result['status']=='global_recovery_requires_native_comparison'
    assert result['baseline_check']['valid'] and not result['agreement_claimed']
    assert summarize_recovery(result)['global_fit']['value']==-.1


@pytest.mark.parametrize('status,value,expected',[(3,-.1,'recovery_global_not_converged'),
                                               (0,.1,'invalid_global_baseline')])
def test_global_only_failed_or_inadequate_restart_remains_failed(status,value,expected):
    def fit(label,start,fixed):return dict(status=status,value=value,values=start)
    result=recover_global_fit(fit,previous(),poi='kl')
    assert result['status']==expected
    assert result['attempts'][0]['status']==status


def test_global_only_keeps_a_better_existing_unrestricted_fit():
    old=dict(task='old_global',fixed={},value=-2.,status=0,values=np.array([2.,0.]))
    def fit(label,start,fixed):return dict(status=0,value=-.1,values=start)
    result=recover_global_fit(fit,[old,*previous()],poi='kl')
    assert result['global_fit'] is old
    assert result['attempts'][0]['value']==-.1
    assert result['baseline_check']['global_value']==-2.
