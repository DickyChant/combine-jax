"""Device profile scans: analytic curves, runtime reuse and native card gates."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from iminuit import Minuit

from combine_jax.graph import Parameter
from combine_jax.tensor.profiling import make_profile_scan_function, prepare_profile_scan
from combine_jax.tensor.parameter_info import pack_parameter_info, fit_coordinates, COORDINATE_FIELDS, model_parameter_info
from combine_jax.tensor.model import from_workspace
from combine_jax.root_io import open_workspace, load_caching_nll
from test_bin_statistics import template_path


def coordinates(info, points):
    rows=[fit_coordinates(info,fixed={'x':float(x)}) for x in points]
    return jax.device_put({key:np.stack([row[key] for row in rows]) for key in COORDINATE_FIELDS})


def test_correlated_quadratic_profile_reuses_runtime_grid_and_model_buffers():
    info=pack_parameter_info([Parameter('x',0.,-4.,5.,.7,'poi'),Parameter('y',0.,-4.,4.,.4,'nuisance')])
    def objective(theta, arrays):
        delta=theta-arrays['center']
        return .5*delta@arrays['hessian']@delta
    fn=make_profile_scan_function(objective,gradient_tolerance=1e-10)
    first=np.array([-.5,0.,1.3,2.]);c=coordinates(info,first)
    arrays=dict(center=jnp.array([1.3,-.7]),hessian=jnp.array([[4.,1.2],[1.2,2.]]))
    compiled=jax.jit(fn).lower(c,arrays).compile()
    for points,center in [(first,[1.3,-.7]),(first+.2,[.8,.1])]:
        data={**arrays,'center':jnp.array(center)}
        result=jax.device_get(compiled(coordinates(info,points),data))
        np.testing.assert_array_equal(result['status'],0)
        np.testing.assert_array_equal(result['values'][:,0],points)
        np.testing.assert_allclose(result['values'][:,1],center[1]-.6*(points-center[0]),atol=1e-9,rtol=0)
        np.testing.assert_allclose(result['value'],1.64*(points-center[0])**2,atol=1e-12,rtol=0)
        assert np.max(np.abs(result['projected_gradient']))<=1e-10
        assert not any(key in result for key in ('covariance','hessian','inverse'))
    trace=jax.make_jaxpr(fn)(c,arrays)
    assert not trace.consts and 'callback' not in str(trace)
    assert any(e.primitive.name=='scan' for e in trace.jaxpr.eqns)


def test_scan_keeps_failed_points_explicit_and_handles_no_free_nuisances():
    info=pack_parameter_info([Parameter('x',0.,-4.,4.,1.,'poi'),Parameter('y',0.,-4.,4.,1.,'nuisance')])
    c=coordinates(info,[-1.,1.])
    objective=lambda theta,arrays:(theta[1]-theta[0])**2
    result=jax.device_get(jax.jit(make_profile_scan_function(objective,max_iterations=0))(c,{}))
    np.testing.assert_array_equal(result['status'],1)
    assert np.all(np.max(np.abs(result['projected_gradient']),axis=1)>0.)
    single=pack_parameter_info([Parameter('x',0.,-4.,4.,1.,'poi')])
    result=jax.device_get(jax.jit(make_profile_scan_function(lambda theta,a:theta[0]**2))(
        coordinates(single,[-1.,0.,2.]),{}))
    np.testing.assert_array_equal(result['status'],0)
    np.testing.assert_array_equal(result['value'],[1.,0.,4.])
    assert result['projected_gradient'].shape==(3,0)


@pytest.mark.parametrize('kind',['counting','automcstats'])
def test_profile_scan_native_minima(counting_path,template_path,kind):
    root=load_caching_nll();path=counting_path if kind=='counting' else template_path
    card=Path(__file__).parent/'cards/counting.txt' if kind=='counting' else Path(path).parent/'card.txt'
    with open_workspace(path) as workspace:
        model=from_workspace(workspace,card);info=model_parameter_info(model)
        names=info.metadata['names'];initial=info.arrays['initial']
        grid=np.array([0.,.4,1.1,2.])
        function,c,a=prepare_profile_scan(model,'r',grid,gradient_tolerance=1e-7)
        compiled=function.lower(c,a).compile()
        result=jax.device_get(compiled(c,a))
        np.testing.assert_array_equal(result['status'],0)
        np.testing.assert_array_equal(result['values'][:,names.index('r')],grid)
        assert np.max(np.abs(result['projected_gradient']))<=1e-7
        native=root.cacheutils.CachingSimNLL(workspace.obj('ModelConfig').GetPdf(),workspace.data('data_obs'),
            workspace.obj('ModelConfig').GetNuisanceParameters())
        native.setAnalyticBarlowBeeston(True);native.clearConstantZeroPoint();native.clearZeroPoint()
        def objective(*point):
            for name,value in zip(names,point,strict=True):workspace.var(name).setVal(float(value))
            return float(native.getVal())
        ref=Minuit(objective,*initial,name=names);ref.errordef=.5;ref.tol=1e-8;ref.strategy=2
        for i,name in enumerate(names):
            ref.limits[name]=(info.arrays['lower'][i],info.arrays['upper'][i])
            ref.errors[name]=max(float(info.arrays['error'][i]),.1)
        ref.fixed['r']=True
        points=[]
        for i,poi in enumerate(grid):
            ref.values=initial;ref.values['r']=poi;ref.migrad();assert ref.valid
            np.testing.assert_allclose(result['value'][i],ref.fval,atol=1e-6,rtol=0)
            np.testing.assert_allclose(result['values'][i],np.array(ref.values),atol=1e-4,rtol=1e-4)
            assert abs(objective(*result['values'][i])-result['value'][i])<1e-6
            points.append(dict(poi=float(poi),device_nll=float(result['value'][i]),native_nll=float(ref.fval),
                native_valid=bool(ref.valid),nll_absolute_error=float(abs(result['value'][i]-ref.fval)),
                iterations=int(result['iteration'][i]),projected_gradient_max=float(np.max(np.abs(result['projected_gradient'][i]),initial=0.))))
        native.setAnalyticBarlowBeeston(False)
        print('PROFILE_SCAN_VALIDATION_JSON='+json.dumps(dict(kind=kind,points=points),allow_nan=False))
        for bad in ([],[np.nan],[[1.]],[-1.],[1000.]):
            with pytest.raises(ValueError,match='Profile grid'):
                prepare_profile_scan(model,'r',bad)
        with pytest.raises(ValueError,match='Unknown profile parameter'):
            prepare_profile_scan(model,'not_a_parameter',[1.])
        with pytest.raises(ValueError,match='additional fixed'):
            prepare_profile_scan(model,'r',[1.],fixed={'r':1.})
        with pytest.raises(ValueError,match='omit final covariance'):
            prepare_profile_scan(model,'r',[1.],covariance=True)
