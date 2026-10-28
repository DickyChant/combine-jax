"""Standalone curvature strategies retain explicit CLI provenance and gates."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from combine_jax.likelihood import from_workspace as old_import
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace


@pytest.mark.parametrize('linearization',['reuse','recompute'])
def test_real_curvature_capture_and_cache(counting_path,tmp_path,linearization):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'))
        old=old_import(workspace)
        initial=old.initial.copy()
        expected_value,expected_gradient=old.value_and_grad(initial)
        expected_hessian=old.hessian(initial)
    model.save(tmp_path/'model');output=tmp_path/'measurement.json'
    command=[sys.executable,'scripts/benchmark_tensor.py','--model',str(tmp_path/'model'),
        '--backend','cpu','--operation','hessian','--batch-size','1','--repeat','1',
        '--hessian-linearization',linearization,'--cache',str(tmp_path/'cache'),'--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(output.read_text())
    assert report['status']=='measured_pending_validation' and not report['root_imported']
    assert report['hessian_linearization']==linearization and report['batch_size']==1
    assert len(report['timings']['warm_seconds'])==1
    assert list((tmp_path/'cache').rglob('*-cache'))
    with np.load(report['arrays'],allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved['points'],initial[None,:])
        np.testing.assert_allclose(saved['values'],[expected_value],rtol=0,atol=1e-10)
        np.testing.assert_allclose(saved['gradients'],expected_gradient[None,:],rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(saved['hessians'],expected_hessian[None,:,:],rtol=1e-10,atol=1e-10)


@pytest.mark.parametrize('operation',['fit','nll','value_gradient'])
def test_recompute_option_cannot_silently_affect_other_operations(tmp_path,operation):
    output=tmp_path/'rejected.json'
    result=subprocess.run([sys.executable,'scripts/benchmark_tensor.py','--model','unused',
        '--backend','cpu','--operation',operation,'--hessian-linearization','recompute',
        '--output',str(output)],capture_output=True,text=True,timeout=30)
    assert result.returncode==2
    assert 'applies only to standalone Hessians' in result.stderr
    assert not output.exists()
