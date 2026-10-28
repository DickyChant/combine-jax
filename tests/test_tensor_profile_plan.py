import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from tensor_profile_plan import load_plan,make_plan
from audit_tensor_profile_comparison import add_baseline_shifts
from tensor_benchmark_common import save_json,sha
from combine_jax.root_io import open_workspace
from combine_jax.tensor.fit_policy import POLICY
from combine_jax.tensor.model import from_workspace


@pytest.fixture(scope='module')
def baseline(counting_path,tmp_path_factory):
    folder=tmp_path_factory.mktemp('tensor_profile_baseline')
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
    model.save(folder/'model')
    manifest=folder/'manifest.json'
    manifest.write_text(json.dumps(dict(counting=dict(workspace=str(counting_path),workspace_sha256=sha(counting_path),
        model=str(folder/'model'),dataset='data_obs',configuration={},hashes={}))))
    def run(script,*arguments):
        r=subprocess.run([sys.executable,'scripts/'+script,*map(str,arguments)],capture_output=True,text=True,timeout=180)
        assert r.returncode==0,r.stdout[-1500:]+r.stderr[-1500:]
    tensor=folder/'tensor.json';native=folder/'native.json';point=folder/'point.json'
    run('benchmark_tensor.py','--model',folder/'model','--backend','cpu','--operation','fit',
        '--fit-stopping-criterion','edm','--single-fit','--repeat','1','--output',tensor)
    run('benchmark_combine_fit.py','--manifest',manifest,'--case','counting',
        '--directory',folder/'native_fit','--output',native)
    run('check_tensor_fit_candidate.py','--manifest',manifest,'--case','counting','--fit',tensor,'--output',point)
    arguments=dict(manifest=manifest,case='counting',tensor=tensor,native=native,native_point=point,
        parameter='lumi',poi='r',offsets=[-1.,0.,1.])
    output=folder/'plan.json'
    run('tensor_profile_plan.py','--manifest',manifest,'--case','counting','--tensor',tensor,'--native',native,
        '--native-point',point,'--parameter','lumi','--poi','r','--offsets','-1','0','1','--output',output)
    return arguments,json.loads(output.read_text()),model


def test_native_baseline_defines_common_physical_grid(baseline,tmp_path):
    arguments,plan,model=baseline
    assert make_plan(**arguments)==plan
    assert plan['baseline_comparison']['status']=='passed_same_minimum_gates_no_speed_claim'
    index=plan['parameter_index']
    expected=plan['baseline_values'][index]+np.asarray(plan['offsets'])*plan['baseline_errors'][index]
    np.testing.assert_array_equal(plan['grid'],expected)
    np.testing.assert_array_equal(plan['initial_values'],model.arrays['parameter_info__initial'])
    assert plan['initial']=='imported nominal'
    path=tmp_path/'plan.json';save_json(path,plan);assert load_plan(path)==plan
    print('PROFILE_BASELINE_PLAN',json.dumps(dict(grid=plan['grid'],parameter=plan['parameter'],poi=plan['poi'],
        baseline_nll_difference=plan['baseline_comparison']['minimum_nll_difference'],
        baseline_max_delta_sigma=plan['baseline_comparison']['max_parameter_delta_over_sigma'])),flush=True)


@pytest.mark.parametrize('offsets',[[-1,1],[-1,0,0,1],[0,1,2],[-1,0,float('nan')],[-100,0,100]])
def test_invalid_or_out_of_range_offsets_are_not_clipped(baseline,offsets):
    arguments,_,_=baseline
    with pytest.raises(ValueError):make_plan(**{**arguments,'offsets':offsets})


@pytest.mark.parametrize('fault',['grid','start','baseline','host'])
def test_plan_revalidation_rejects_modified_coordinates(baseline,tmp_path,fault):
    _,original,_=baseline;plan=copy.deepcopy(original)
    if fault=='grid':plan['grid'][0]+=.01
    elif fault=='start':plan['initial_values'][0]+=.01
    elif fault=='baseline':plan['baseline_values'][0]+=.01
    else:plan['host']='different'
    path=tmp_path/'plan.json';save_json(path,plan)
    with pytest.raises(ValueError,match='revalidated baseline'):load_plan(path)


def test_reported_native_edm_flag_cannot_replace_the_numeric_gate(baseline,tmp_path):
    arguments,_,_=baseline
    native=json.loads(Path(arguments['native']).read_text())
    assert native['fit']['edm_target_met'];native['fit']['edm']=1e-4
    path=tmp_path/'native.json';save_json(path,native)
    with pytest.raises(AssertionError):make_plan(**{**arguments,'native':path})


def test_hhvv_label_rejects_observed_input_before_reading_fit_files(baseline,tmp_path):
    arguments,_,_=baseline
    manifest=json.loads(Path(arguments['manifest']).read_text())
    path=tmp_path/'manifest.json';save_json(path,{'hhvv':manifest['counting']})
    with pytest.raises(ValueError,match='nominal pre-fit Asimov'):
        make_plan(**{**arguments,'manifest':path,'case':'hhvv','native':tmp_path/'never_read.json'})


def test_shared_plan_drives_both_profile_runners_and_poi_shift_audit(baseline,tmp_path):
    arguments,plan,_=baseline
    plan_path=Path(arguments['manifest']).parent/'plan.json'
    model=Path(arguments['manifest']).parent/'model'
    def run(script,*args):
        result=subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],capture_output=True,text=True,timeout=180)
        assert result.returncode==0,result.stdout[-1500:]+result.stderr[-1500:]
    tensor=tmp_path/'profile.json'
    run('benchmark_tensor_profile.py','--model',model,'--plan',plan_path,'--backend','cpu','--output',tensor)
    natives=[];checks=[]
    for i in range(len(plan['grid'])):
        native=tmp_path/f'native_{i}.json';point=tmp_path/f'point_{i}.json'
        run('benchmark_combine_fit.py','--manifest',arguments['manifest'],'--case','counting',
            '--profile-plan',plan_path,'--profile-index',i,'--directory',tmp_path/f'fit_{i}','--output',native)
        run('check_tensor_fit_candidate.py','--manifest',arguments['manifest'],'--case','counting',
            '--fit',tensor,'--profile-index',i,'--output',point)
        natives.append(native);checks.append(point)
    output=tmp_path/'audit.json'
    run('audit_tensor_profile_comparison.py','--manifest',arguments['manifest'],'--case','counting',
        '--tensor',tensor,'--native',*natives,'--native-point',*checks,'--output',output)
    report=json.loads(output.read_text())
    assert report['status']=='passed_all_profile_minima_no_speed_claim' and report['baseline_linked_poi_shifts']
    assert len(report['points'])==3 and all(r['poi_shift']['passed'] for r in report['points'])
    print('PROFILE_PLAN_EXECUTION',json.dumps(dict(parameter=report['parameter'],grid=report['grid'],
        max_nll_difference=report['max_minimum_nll_difference'],max_delta_sigma=report['max_parameter_delta_over_sigma'],
        max_poi_shift_error=max(r['poi_shift']['absolute_error'] for r in report['points']))),flush=True)


def test_small_poi_shift_cannot_hide_behind_large_absolute_poi():
    plan=dict(host='host',dataset='generic',model_sha256={'model':'same'},parameter='nuisance',grid=[1.],
        parameter_names=['nuisance','poi'],poi_index=1,poi='poi',baseline_values=[0.,100.],baseline_errors=[1.,2.],baseline_canonical_nll=3.)
    tensor=dict(host='host',dataset='generic',model_sha256=plan['model_sha256'],parameter='nuisance',grid=[1.],
        profile_plan=dict(path='plan',sha256='same'))
    arrays=dict(parameter_names=np.asarray(plan['parameter_names']),values=np.asarray([[1.,100.25]]),value=np.asarray([4.]))
    natives=[dict(profile_plan=dict(path='plan',sha256='same',index=0),fit=dict(values=[1.,100.2],canonical_nll=4.))]
    rows=[dict(passed=True)]
    add_baseline_shifts(rows,plan,tensor,arrays,natives)
    assert not rows[0]['passed'] and rows[0]['poi_shift']['tolerance']==.02
    assert rows[0]['tensor_delta_nll']==rows[0]['native_delta_nll']==1.
