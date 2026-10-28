"""Matrix-free inverse rows must reproduce dense curvature or fail explicitly."""
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.covariance_rows import make_covariance_rows
from combine_jax.tensor.model import from_workspace,make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info,fit_coordinates
from combine_jax.likelihood import from_workspace as old_import
from combine_jax.root_io import open_workspace
from test_bin_statistics import template_path


def test_coupled_scaled_system_uses_runtime_inputs_and_device_loops(monkeypatch):
    def objective(x,a):
        return .5*jnp.sum(a['diagonal']*x*x)+.5*jnp.sum(a['coupling']*(x[1:]-x[:-1])**2)
    arrays=dict(diagonal=np.arange(1.,7.),coupling=np.arange(1.,6.)*.3)
    theta=np.arange(6,dtype=np.float64)/10
    free=np.array([0,2,3,5],dtype=np.int32);scale=np.array([.1,2.,.4,1.])
    columns=np.array([2,0],dtype=np.int32)
    hessian=np.array(jax.hessian(objective)(theta,arrays))[np.ix_(free,free)]
    expected=np.zeros((2,6));expected[:,free]=np.linalg.inv(hessian)[columns]
    def forbidden(*args,**kwargs):raise AssertionError('Dense Hessian called')
    monkeypatch.setattr(jax,'hessian',forbidden)
    function=make_covariance_rows(objective,max_iterations=40,relative_tolerance=1e-12)
    result=jax.jit(function)(theta,arrays,free,scale,columns)
    np.testing.assert_array_equal(result['status'],0)
    np.testing.assert_allclose(result['rows'],expected,atol=1e-12,rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(result['rows'])[:,[1,4]],0.)
    assert np.all(result['residual_norm']<=result['residual_target'])
    np.testing.assert_array_equal(result['hvp_evaluations'],result['iterations']+1)
    # Runtime values can change without rebuilding the function or recompiling.
    other={k:v*2 for k,v in arrays.items()}
    updated=jax.jit(function)(theta,other,free,scale,columns)
    np.testing.assert_allclose(updated['rows'],expected/2,atol=1e-12,rtol=1e-12)
    trace=jax.make_jaxpr(function)(theta,arrays,free,scale,columns)
    assert not trace.consts and 'scan[' in str(trace) and 'while[' in str(trace)
    assert 'callback' not in str(trace)


@pytest.mark.parametrize('diagonal,expected',[(np.array([1.,-1.]),2),(np.array([1.,0.]),2),(np.array([1.,np.nan]),3)])
def test_nonpositive_or_nonfinite_curvature_never_emits_a_covariance(diagonal,expected):
    function=make_covariance_rows(lambda x,a:.5*jnp.sum(a*x*x))
    r=jax.jit(function)(np.ones(2),diagonal,np.arange(2,dtype=np.int32),np.ones(2),np.array([1],dtype=np.int32))
    assert r['status'][0]==expected and np.isnan(r['rows']).all()
    assert r['hvp_evaluations'][0]==2


def test_iteration_limit_retains_residual_without_silent_inverse_approximation():
    function=make_covariance_rows(lambda x,a:jnp.sum(x*x)+.7*x[0]*x[1],max_iterations=1,relative_tolerance=1e-12)
    r=jax.jit(function)(np.ones(2),{},np.arange(2,dtype=np.int32),np.ones(2),np.array([0],dtype=np.int32))
    assert r['status'][0]==1 and r['residual_norm'][0]>r['residual_target'][0]
    assert np.isnan(r['rows']).all()


def test_tiny_recurrence_residual_cannot_hide_true_residual_failure():
    function=make_covariance_rows(lambda x,a:.5*x[0]**2+.55*x[1]**2+.1*x[0]*x[1],
        max_iterations=20,relative_tolerance=1e-28)
    r=jax.jit(function)(np.ones(2),{},np.arange(2,dtype=np.int32),np.ones(2),np.array([0],dtype=np.int32))
    assert r['recurrence_residual_norm'][0]<=r['residual_target'][0]
    assert r['residual_norm'][0]>r['residual_target'][0] and r['status'][0]==5
    assert np.isnan(r['rows']).all()


@pytest.mark.parametrize('fault',['duplicate','out_of_range','scale_zero','scale_nan','column','theta_nan'])
def test_invalid_runtime_coordinates_fail_without_hvp(fault):
    theta=np.ones(2);free=np.arange(2,dtype=np.int32);scale=np.ones(2);columns=np.array([0],dtype=np.int32)
    if fault=='duplicate':free[1]=0
    elif fault=='out_of_range':free[1]=2
    elif fault=='scale_zero':scale[0]=0
    elif fault=='scale_nan':scale[0]=np.nan
    elif fault=='column':columns[0]=2
    else:theta[0]=np.nan
    function=make_covariance_rows(lambda x,a:jnp.sum(x*x))
    r=jax.jit(function)(theta,{},free,scale,columns)
    assert r['status'][0]==4 and r['hvp_evaluations'][0]==0 and np.isnan(r['rows']).all()


@pytest.mark.parametrize('kind',['counting','automcstats','binned'])
@pytest.mark.parametrize('policy',['ad','structural','structural_sequential'])
@pytest.mark.parametrize('recompute',[False,True])
def test_generic_card_inverse_rows_match_old_exact_hessian(counting_path,template_path,tmp_path,kind,policy,recompute):
    if kind=='counting':path,card=counting_path,Path('tests/cards/counting.txt')
    elif kind=='automcstats':path,card=template_path,Path(template_path).parent/'card.txt'
    else:
        # Keep ROOT's fixture-construction ownership out of the AD test process.
        # The same generator and card are used, with a checked subprocess exit.
        built=subprocess.run([sys.executable,'scripts/build_binned_fixture.py',str(tmp_path/'fixture')],
            capture_output=True,text=True,timeout=120)
        assert built.returncode==0,built.stdout+built.stderr
        path=tmp_path/'fixture/workspace.root';card=path.parent/'card.txt'
    with open_workspace(path) as workspace:
        model=from_workspace(workspace,card);old=old_import(workspace)
        coordinates=fit_coordinates(model_parameter_info(model));free=coordinates['free_indices'];scale=coordinates['scale']
        point=old.initial;hessian=old.hessian(point)[np.ix_(free,free)]
        assert np.linalg.eigvalsh(hessian).min()>0
        columns=np.array([0,len(free)-1],dtype=np.int32)
        function=make_covariance_rows(make_model_function(model,binned_gradient=policy,row_chunk_size=3),
            max_iterations=100,relative_tolerance=1e-11,recompute_point=recompute)
        r=jax.jit(function)(point,dict(model.arrays),free,scale,columns)
        np.testing.assert_array_equal(r['status'],0)
        expected=np.zeros((len(columns),len(point)));expected[:,free]=np.linalg.inv(hessian)[columns]
        np.testing.assert_allclose(r['rows'],expected,atol=1e-8,rtol=1e-9)
        assert np.all(r['residual_norm']<=r['residual_target'])


@pytest.mark.parametrize('kwargs',[dict(max_iterations=0),dict(max_iterations=True),dict(relative_tolerance=-1),
    dict(relative_tolerance=np.inf),dict(relative_tolerance=0,absolute_tolerance=0)])
def test_invalid_solver_configuration(kwargs):
    with pytest.raises(ValueError):make_covariance_rows(lambda x,a:jnp.sum(x*x),**kwargs)
