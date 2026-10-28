import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_tensor_profile_comparison import compare_points
from tensor_benchmark_common import sha
from combine_jax.root_io import open_workspace
from combine_jax.tensor.fit_policy import POLICY
from combine_jax.tensor.model import from_workspace
from test_bin_statistics import template_path


@pytest.fixture(scope='module',params=['counting','automcstats'])
def captured(request,counting_path,template_path,tmp_path_factory):
    kind=request.param;directory=tmp_path_factory.mktemp('profile_comparison_'+kind)
    workspace=template_path if kind=='automcstats' else counting_path
    card=Path(workspace).parent/'card.txt' if kind=='automcstats' else Path('tests/cards/counting.txt')
    parameter,grid=('r',[.8,1.,1.2]) if kind=='automcstats' else ('lumi',[-.5,0.,.5])
    with open_workspace(workspace) as ws:model=from_workspace(ws,card,fit_policy=POLICY)
    model.save(directory/'model')
    manifest=directory/'manifest.json'
    manifest.write_text(json.dumps({kind:dict(workspace=str(workspace),workspace_sha256=sha(workspace),
        model=str(directory/'model'),configuration={},dataset='data_obs',hashes={})}))
    def run(script,*args):
        completed=subprocess.run([sys.executable,'scripts/'+script,*map(str,args)],capture_output=True,text=True,timeout=180)
        assert completed.returncode==0,completed.stdout[-1500:]+completed.stderr[-1500:]
    tensor_path=directory/'tensor.json'
    run('benchmark_tensor_profile.py','--model',directory/'model','--parameter',parameter,
        '--values',*grid,'--backend','cpu','--output',tensor_path)
    native_paths=[];check_paths=[]
    for i,value in enumerate(grid):
        native=directory/f'native_{i}.json';check=directory/f'check_{i}.json'
        run('benchmark_combine_fit.py','--manifest',manifest,'--case',kind,'--fixed-parameter',parameter,value,
            '--directory',directory/f'fit_{i}','--output',native)
        run('check_tensor_fit_candidate.py','--manifest',manifest,'--case',kind,'--fit',tensor_path,
            '--profile-index',i,'--output',check)
        native_paths.append(native);check_paths.append(check)
    output=directory/'audit.json'
    run('audit_tensor_profile_comparison.py','--manifest',manifest,'--case',kind,'--tensor',tensor_path,
        '--native',*native_paths,'--native-point',*check_paths,'--output',output)
    tensor=json.loads(tensor_path.read_text())
    with np.load(tensor['arrays'],allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    natives=[json.loads(p.read_text()) for p in native_paths];checks=[json.loads(p.read_text()) for p in check_paths]
    report=json.loads(output.read_text())
    print('PROFILE_GRID_NATIVE_AUDIT',json.dumps(dict(case=kind,grid=grid,status=report['status'],
        max_minimum_nll_difference=report['max_minimum_nll_difference'],
        max_parameter_delta_over_sigma=report['max_parameter_delta_over_sigma'])),flush=True)
    return model,tensor,arrays,natives,checks,report


def test_all_three_profile_points_pass_native_minimum_gates(captured):
    model,tensor,arrays,natives,checks,report=captured
    assert report['status']=='passed_all_profile_minima_no_speed_claim'
    assert len(report['points'])==3 and all(p['passed'] for p in report['points'])
    assert compare_points(model,tensor,arrays,natives,checks)==report['points']


@pytest.mark.parametrize('fault',['missing','order','grid','status','edm','native_edm','start','fixed','point','dataset','host'])
def test_profile_integrity_cannot_accept_subset_or_different_state(captured,fault):
    model,tensor,arrays,natives,checks,_=captured
    tensor,arrays,natives,checks=copy.deepcopy((tensor,arrays,natives,checks))
    if fault=='missing':natives.pop()
    elif fault=='order':natives.reverse()
    elif fault=='grid':arrays['grid'][0]+=.01
    elif fault=='status':arrays['status'][0]=1
    elif fault=='edm':arrays['corrected_estimated_edm'][0]=1e-4
    elif fault=='native_edm':natives[0]['fit']['edm']=1e-4
    elif fault=='start':natives[0]['initial_values'][0]+=.01
    elif fault=='fixed':arrays['values'][0,list(arrays['parameter_names']).index(tensor['parameter'])]+=.01
    elif fault=='point':checks[0]['point'][0]+=.01
    elif fault=='dataset':natives[0]['dataset']='different'
    elif fault=='host':natives[0]['host']='different'
    with pytest.raises((AssertionError,ValueError)):
        compare_points(model,tensor,arrays,natives,checks)


def test_one_bad_minimum_fails_the_profile_without_hiding_other_points(captured):
    model,tensor,arrays,natives,checks,_=captured
    natives=copy.deepcopy(natives);natives[0]['fit']['canonical_nll']+=.001
    rows=compare_points(model,tensor,arrays,natives,checks)
    assert not rows[0]['passed'] and all(row['passed'] for row in rows[1:])
