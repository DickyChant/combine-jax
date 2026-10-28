"""Device column loops preserve exact coupled curvature and native card gates."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.curvature import make_value_gradient_hessian
from combine_jax.tensor.model import from_workspace,make_model_function
from combine_jax.likelihood import from_workspace as old_import
from combine_jax.root_io import open_workspace,load_caching_nll
from combine_jax.validation import finite_difference
from test_bin_statistics import template_path


@pytest.mark.parametrize('n',[0,3,10])
@pytest.mark.parametrize('width',[1,4,12])
@pytest.mark.parametrize('linearization',['reuse','recompute'])
def test_analytic_dense_hessian_partial_batches_empty_and_runtime_arrays(n,width,linearization):
    def objective(x,a):return jnp.sum(a['weights']*jnp.exp(x))+jnp.sum(x)**4
    x=jnp.linspace(-.3,.7,n);function=make_value_gradient_hessian(objective,column_batch_size=width,linearization=linearization)
    compiled=jax.jit(function)
    for scale in (1.,1.7):
        weights=np.linspace(.2,.9,n)*scale;arrays={'weights':jnp.asarray(weights)}
        value,gradient,hessian=map(np.asarray,compiled(x,arrays));total=float(np.sum(x));diagonal=weights*np.exp(x)
        np.testing.assert_allclose(value,np.sum(diagonal)+total**4,rtol=1e-13)
        np.testing.assert_allclose(gradient,diagonal+4*total**3,rtol=1e-13)
        np.testing.assert_allclose(hessian,np.diag(diagonal)+np.full((n,n),12*total**2),atol=1e-13,rtol=1e-13)
    trace=jax.make_jaxpr(function)(x,arrays);assert not trace.consts
    assert sum(e.primitive.name=='while' for e in trace.jaxpr.eqns)==bool(n)


@pytest.mark.parametrize('linearization',['reuse','recompute'])
def test_fixed_coordinates_and_input_contract(linearization):
    def objective(x,a):return jnp.exp(x[0])*a['weight']+(x[0]+x[2])**4
    arrays={'weight':jnp.array(2.)};x=jnp.array([.2,.7,.4])
    fn=make_value_gradient_hessian(objective,linearization=linearization)
    value,gradient,hessian=map(np.asarray,jax.jit(fn)(x,arrays))
    np.testing.assert_array_equal(gradient[1],0.);np.testing.assert_array_equal(hessian[1,:],0.);np.testing.assert_array_equal(hessian[:,1],0.)
    np.testing.assert_allclose(hessian,jax.hessian(objective)(x,arrays),atol=1e-13)
    for width in (0,-1,1.5,True):
        with pytest.raises(ValueError,match='positive integer'):make_value_gradient_hessian(objective,column_batch_size=width)
    for bad in (x.astype(jnp.float32),x[None,:]):
        with pytest.raises(ValueError,match='float64 parameter vector'):jax.jit(fn)(bad,arrays)
    with pytest.raises(ValueError,match='linearization'):make_value_gradient_hessian(objective,linearization='unknown')


@pytest.mark.parametrize('kind',['counting','automcstats'])
@pytest.mark.parametrize('linearization',['reuse','recompute'])
def test_complete_native_cards_and_old_hessian(counting_path,template_path,kind,linearization):
    root=load_caching_nll();path=counting_path if kind=='counting' else template_path
    card=Path(__file__).parent/'cards/counting.txt' if kind=='counting' else Path(path).parent/'card.txt'
    with open_workspace(path) as workspace:
        old=old_import(workspace);model=from_workspace(workspace,card)
        arrays=jax.device_put(dict(model.arrays));function=make_value_gradient_hessian(make_model_function(model),column_batch_size=4,linearization=linearization)
        compiled=jax.jit(function);pdf=workspace.obj('ModelConfig').GetPdf()
        reference=root.cacheutils.CachingSimNLL(pdf,workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        reference.setAnalyticBarlowBeeston(True);reference.clearConstantZeroPoint();reference.clearZeroPoint()
        def native(point):
            for n,v in zip(old.parameter_names,point,strict=True):workspace.var(n).setVal(float(v))
            return float(reference.getVal())
        for r in (1.,1.4,2.):
            point=old.initial.copy();point[old.parameter_names.index('r')]=r
            value,gradient,hessian=map(np.asarray,compiled(point,arrays))
            np.testing.assert_allclose(value,native(point),atol=1e-10,rtol=0.)
            np.testing.assert_allclose(gradient,finite_difference(native,point,np.full(len(point),2e-4)),atol=1e-8,rtol=1e-4)
            np.testing.assert_allclose(hessian,old.hessian(point),atol=1e-10,rtol=1e-10)
            fd=finite_difference(lambda q:old.value_and_grad(q)[1],point,np.full(len(point),2e-4))
            np.testing.assert_allclose(hessian,fd,atol=1e-8,rtol=1e-6)
        reference.setAnalyticBarlowBeeston(False)
