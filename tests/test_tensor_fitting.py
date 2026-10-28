"""Device fits: raw exact covariance, held parameters and native card oracles."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from iminuit import Minuit

from combine_jax.graph import Parameter
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import pack_parameter_info,fit_coordinates,attach_parameter_info,model_parameter_info,COORDINATE_FIELDS
from combine_jax.tensor.fitting import make_fit_function,prepare_fit,physical_parameters
from combine_jax.tensor.model import from_workspace
from combine_jax.likelihood import from_workspace as old_import
from combine_jax.root_io import open_workspace,load_caching_nll
from test_bin_statistics import template_path


def numeric_coordinates(info,**kw):
    c=fit_coordinates(info,**kw)
    return jax.device_put({n:c[n] for n in COORDINATE_FIELDS})


def test_fused_scaled_bound_roundoff_preserves_physical_values_and_raw_curvature():
    info=pack_parameter_info([Parameter('rate',1.,0.,10.,.1,'poi'),
                             Parameter('shape',2.,-10.,10.,1.,'nuisance')])
    c=numeric_coordinates(info)
    lower=jnp.array([-10.,-1.3])
    mapped=jax.jit(physical_parameters)(lower,c)
    assert float(mapped[0])==0.
    # Clipping is only a floating-point value correction. The derivative stays
    # the affine scale, including at the physical bound; it is not halved/zeroed.
    np.testing.assert_array_equal(jax.jit(jax.jacfwd(physical_parameters))(lower,c),np.diag([.1,1.]))
    def objective(x,a):
        d=x[1]-.7
        return jnp.where(x[0]>=0,.5*x[0]**2+x[0]+.5*d*d+.1*x[0]*d,jnp.nan)
    result=jax.device_get(jax.jit(make_fit_function(objective,max_iterations=200,
        gradient_tolerance=1e-9))(c,{}))
    assert result['status']==0 and result['covariance_status']==0
    assert result['iteration']<200 and float(result['values'][0])==0.
    np.testing.assert_allclose(result['values'],[0.,.7],atol=1e-9,rtol=0.)
    expected=np.array([[1.,.1],[.1,1.]])
    np.testing.assert_allclose(result['hessian'],expected,atol=1e-14,rtol=0.)
    np.testing.assert_allclose(result['covariance'],np.linalg.inv(expected),atol=1e-14,rtol=0.)


def test_physical_covariance_scaling_fixed_rows_and_runtime_inputs():
    info=pack_parameter_info([Parameter('x',0.,-10.,10.,.3,'poi'),Parameter('held',2.,2.,2.,0.,'other'),
        Parameter('y',0.,-10.,10.,2.,'nuisance')])
    c=numeric_coordinates(info)
    def objective(x,a):
        d=jnp.stack([x[0],x[2]])-a['center']
        return .5*d@a['matrix']@d+(x[1]-a['held'])**2
    fn=make_fit_function(objective,gradient_tolerance=1e-9);compiled=jax.jit(fn)
    a=dict(matrix=jnp.array([[4.,1.],[1.,2.]]),center=jnp.array([.6,-.4]),held=jnp.array(2.))
    result=jax.device_get(compiled(c,a));cov=np.zeros((3,3));cov[np.ix_([0,2],[0,2])]=np.linalg.inv(a['matrix'])
    assert result['status']==0 and result['covariance_status']==0
    np.testing.assert_allclose(result['values'],[.6,2.,-.4],atol=1e-9)
    np.testing.assert_allclose(result['covariance'],cov,atol=1e-13)
    np.testing.assert_allclose(result['hessian'],a['matrix'],atol=1e-13)
    np.testing.assert_array_equal(result['curvature_errors'],result['errors'])
    np.testing.assert_array_equal(result['reported_error_status'],[0,1,0])
    assert result['reported_errors'][1]==0.
    assert result['inverse_residual']<1e-13 and not result['active_bounds'].any()
    c2=numeric_coordinates(info,fixed={'x':.8});second=jax.device_get(compiled(c2,a))
    assert second['status']==0 and second['covariance_status']==0
    np.testing.assert_allclose(second['values'],[.8,2.,-.5],atol=1e-9)
    np.testing.assert_allclose(second['covariance'],np.diag([0.,0.,.5]),atol=1e-13)
    trace=jax.make_jaxpr(fn)(c,a)
    assert not trace.consts and 'callback' not in str(trace)
    assert any(e.primitive.name=='while' for e in trace.jaxpr.eqns)


@pytest.mark.parametrize('curvature,expected',[(0.,4),(-1.,4),(2.,0)])
def test_raw_curvature_at_stationary_point_is_never_repaired(curvature,expected):
    info=pack_parameter_info([Parameter('x',0.,-2.,2.,.5,'poi')]);c=numeric_coordinates(info)
    fn=jax.jit(make_fit_function(lambda x,a:.5*a*x[0]**2))
    result=jax.device_get(fn(c,jnp.array(curvature)))
    assert result['status']==0 and result['covariance_status']==expected
    assert result['converged'] and bool(result['valid_covariance'])==(expected==0)
    if expected:
        assert np.isnan(result['covariance']).all() and np.isnan(result['reported_errors']).all()
        np.testing.assert_array_equal(result['reported_error_status'],3)
    else:np.testing.assert_allclose(result['covariance'],[[.5]],atol=1e-14)


def test_nonfinite_curvature_and_active_bound_are_explicit():
    info=pack_parameter_info([Parameter('x',0.,-1.,1.,1.,'poi')]);c=numeric_coordinates(info)
    fn=jax.jit(make_fit_function(lambda x,a:jnp.abs(x[0])**1.5+a))
    result=jax.device_get(fn(c,jnp.array(0.)))
    assert result['converged'] and result['covariance_status']==2 and not result['valid_covariance']
    assert np.isnan(result['covariance']).all()
    bounded=jax.device_get(jax.jit(make_fit_function(lambda x,a:(x[0]-a)**2))(c,jnp.array(2.)))
    assert bounded['status']==0 and bounded['covariance_status']==0 and bounded['active_bounds'][0]
    np.testing.assert_allclose(bounded['values'],[1.],atol=1e-14)
    np.testing.assert_allclose(bounded['covariance'],[[.5]],atol=1e-14)
    assert bounded['reported_error_status'][0]==2 and np.isnan(bounded['reported_errors'][0])


def test_failed_fit_optional_covariance_and_all_fixed():
    info=pack_parameter_info([Parameter('x',1.,-2.,2.,1.,'poi')])
    objective=lambda x,a:jnp.sum(x*x)+a
    for options,expected in [({'max_iterations':0},1),({'covariance':False},6)]:
        r=jax.device_get(jax.jit(make_fit_function(objective,**options))(numeric_coordinates(info),jnp.array(0.)))
        assert r['covariance_status']==expected and np.isnan(r['covariance']).all()
        assert r['status']==(1 if expected==1 else 0)
    c=numeric_coordinates(info,fixed={'x':1.})
    fn=jax.jit(make_fit_function(objective))
    r=jax.device_get(fn(c,jnp.array(0.)))
    assert r['covariance_status']==7 and r['status']==0 and r['value']==1.
    np.testing.assert_array_equal(r['covariance'],[[0.]])
    bad=jax.device_get(fn(c,jnp.array(jnp.nan)))
    assert bad['status']==3 and bad['covariance_status']==1 and np.isnan(bad['covariance']).all()


@pytest.mark.parametrize('kind',['counting','automcstats'])
@pytest.mark.parametrize('linearization',['reuse','recompute'])
def test_device_fit_uncertainties_and_profiles_against_native(counting_path,template_path,tmp_path,monkeypatch,kind,linearization):
    root=load_caching_nll();path=counting_path if kind=='counting' else template_path
    card=Path(__file__).parent/'cards/counting.txt' if kind=='counting' else Path(path).parent/'card.txt'
    with open_workspace(path) as workspace:
        oracle=old_import(workspace)
        from unittest.mock import Mock
        extract=Mock(wraps=old_import)
        monkeypatch.setattr('combine_jax.likelihood.from_workspace',extract)
        model=from_workspace(workspace,card);assert extract.call_count==1
        info=pack_parameter_info(oracle.parameters)
        model.save(tmp_path/'model');model=TensorBundle.load(tmp_path/'model')
        assert model_parameter_info(model).metadata==info.metadata
        for n,a in info.arrays.items():np.testing.assert_array_equal(model_parameter_info(model).arrays[n],a)
        with pytest.raises(ValueError,match='already has'):attach_parameter_info(model,info)
        base=TensorBundle({k:v for k,v in model.metadata.items() if k!='parameter_info'},
            {k:v for k,v in model.arrays.items() if not k.startswith('parameter_info__')})
        with pytest.raises(ValueError,match='lacks parameter'):prepare_fit(base)
        native=root.cacheutils.CachingSimNLL(workspace.obj('ModelConfig').GetPdf(),workspace.data('data_obs'),
            workspace.obj('ModelConfig').GetNuisanceParameters())
        native.setAnalyticBarlowBeeston(True);native.clearConstantZeroPoint();native.clearZeroPoint()
        def objective(*point):
            for name,value in zip(oracle.parameter_names,point,strict=True):workspace.var(name).setVal(float(value))
            return float(native.getVal())
        reference=Minuit(objective,*oracle.initial,name=oracle.parameter_names)
        reference.errordef=.5;reference.tol=1e-8;reference.strategy=2
        for p in oracle.parameters:
            reference.limits[p.name]=(p.minimum,p.maximum);reference.errors[p.name]=p.error if p.error>0 else .1
        reference.migrad();reference.hesse();assert reference.valid and reference.fmin.has_accurate_covar
        compiled,c,a=prepare_fit(model,gradient_tolerance=1e-7,curvature_linearization=linearization)
        result=jax.device_get(compiled(c,a));assert result['status']==0 and result['covariance_status']==0
        values=np.array(reference.values);errors=np.array(reference.errors);cov=np.array(reference.covariance)
        value_diff=np.abs(result['values']-values);value_tol=np.maximum(.01*np.abs(values),.01*errors)
        assert np.all(value_diff<value_tol)
        np.testing.assert_allclose(result['value'],reference.fval,atol=1e-6,rtol=0)
        np.testing.assert_allclose(result['errors'],errors,rtol=.01,atol=0.)
        np.testing.assert_array_equal(result['reported_error_status'],0)
        np.testing.assert_allclose(result['reported_errors'],errors,rtol=.01,atol=0.)
        np.testing.assert_allclose(result['covariance'],cov,rtol=.01,atol=1e-5)
        # Compare exact tensor curvature with the retained graph oracle at the fit.
        old_hessian=np.asarray(oracle.hessian(result['values']))
        np.testing.assert_allclose(result['hessian'],old_hessian,rtol=1e-9,atol=1e-9)
        summary=dict(kind=kind,reference='Combine v11 CachingSimNLL, native objective minimized by iminuit',
            names=list(oracle.parameter_names),device_values=result['values'].tolist(),native_values=values.tolist(),
            device_errors=result['errors'].tolist(),native_errors=errors.tolist(),device_nll=float(result['value']),
            device_reported_errors=result['reported_errors'].tolist(),
            native_nll=float(reference.fval),nll_absolute_error=float(abs(result['value']-reference.fval)),
            max_parameter_error_over_tolerance=float(np.max(value_diff/value_tol)),
            max_error_relative_difference=float(np.max(np.abs(result['errors']-errors)/errors)),
            device_covariance=result['covariance'].tolist(),native_covariance=cov.tolist(),
            old_hessian_max_absolute_error=float(np.max(np.abs(result['hessian']-old_hessian))),
            covariance_inverse_residual=float(result['inverse_residual']),
            iterations=int(result['iteration']),scaled_projected_gradient_max=float(np.max(np.abs(result['projected_gradient']))),
            covariance_status=int(result['covariance_status']),native_edm=float(reference.fmin.edm),profiles=[])
        index=oracle.parameter_names.index('r')
        for endpoint in result['values'][index]+np.array([-.7,.7])*result['errors'][index]:
            assert info.arrays['lower'][index]<endpoint<info.arrays['upper'][index]
            profile_fn,pc,pa=prepare_fit(model,initial=result['values'],fixed={'r':float(endpoint)},
                gradient_tolerance=1e-7,curvature_linearization=linearization)
            prof=jax.device_get(profile_fn(pc,pa))
            assert prof['status']==0 and prof['covariance_status']==0
            assert np.all(prof['covariance'][index]==0.)
            reference.values=values;reference.values['r']=endpoint;reference.fixed['r']=True
            reference.migrad();assert reference.valid
            np.testing.assert_allclose(prof['value'],reference.fval,atol=1e-6,rtol=0.)
            np.testing.assert_allclose(prof['values'],np.array(reference.values),atol=1e-4,rtol=1e-4)
            summary['profiles'].append(dict(r=float(endpoint),device_nll=float(prof['value']),native_nll=float(reference.fval),
                nll_absolute_error=float(abs(prof['value']-reference.fval)),delta_nll=float(prof['value']-result['value']),
                iterations=int(prof['iteration']),status=int(prof['status'])))
        native.setAnalyticBarlowBeeston(False)
        print('FIT_VALIDATION_JSON='+json.dumps(summary,allow_nan=False))
