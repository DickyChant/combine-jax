"""Real CLI checks for covariance rows and compile-only nonexecution."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.parameter_info import model_parameter_info,fit_coordinates
from combine_jax.likelihood import from_workspace as old_import


@pytest.mark.parametrize('mode',['solve','compile_only','limited'])
def test_counting_covariance_capture_or_compile_only(counting_path,tmp_path,mode):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'));old=old_import(workspace)
        coordinates=fit_coordinates(model_parameter_info(model));free=coordinates['free_indices']
        expected=np.zeros((1,len(old.initial)))
        expected[:,free]=np.linalg.inv(old.hessian(old.initial)[np.ix_(free,free)])[[0]]
        name=old.parameter_names[free[0]]
    model.save(tmp_path/'model');output=tmp_path/'result.json'
    command=[sys.executable,'scripts/benchmark_tensor.py','--model',str(tmp_path/'model'),'--backend','cpu',
        '--operation','covariance_rows','--covariance-parameters',name,'--repeat','1','--curvature-recompute-point','--output',str(output)]
    if mode=='compile_only':command.append('--compile-only')
    if mode=='limited':command+=['--covariance-max-iterations','1']
    p=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert p.returncode==0,p.stdout+p.stderr
    r=json.loads(output.read_text())
    if mode=='compile_only':
        assert r['status']=='compiled_not_executed'
        assert 'first_execute_seconds' not in r['timings'] and not output.with_suffix('.npz').exists()
    else:
        assert r['status']=='measured_pending_validation'
        assert r['covariance_result']['all_rows_solved']==(mode=='solve')
        with np.load(r['arrays']) as saved:
            if mode=='solve':np.testing.assert_allclose(saved['covariance_rows'],expected,atol=1e-8,rtol=1e-8)
            else:
                assert np.isnan(saved['covariance_rows']).all()
                np.testing.assert_array_equal(saved['covariance_status'],1)
            np.testing.assert_array_equal(saved['covariance_parameter_names'],[name])
            assert 'hessians' not in saved.files and 'fit_values' not in saved.files


@pytest.mark.parametrize('options',[[],['--covariance-parameters','r','r'],
    ['--covariance-parameters','r','--operation','nll'],
    ['--covariance-parameters','r','--covariance-max-iterations','0']])
def test_covariance_options_reject_before_loading(tmp_path,options):
    p=subprocess.run([sys.executable,'scripts/benchmark_tensor.py','--model','unused','--backend','cpu',
        '--operation','covariance_rows','--output',str(tmp_path/'result.json'),*options],capture_output=True,text=True,timeout=30)
    assert p.returncode==2 and not (tmp_path/'result.json').exists()
