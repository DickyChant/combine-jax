from array import array

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from combine_jax.tensor.binned_channels import make_binned_function
from test_tensor_binned_channels import bind_model,check_model


def fixture(root,reverse=False):
    x=root.RooRealVar('joint_x','',125.,110.,160.)
    knots=array('d',[110.,118.,125.,139.,145.,153.,160.]);contents=array('d',[1.2,.8,1.,.7,.8,.5,.3])
    spline=root.RooSpline1D('joint_spline','',x,len(knots),knots,contents,'CSPLINE')
    wrapper=root.RooGenericPdf('joint_wrapper','@0',root.RooArgList(spline))
    coefs=[root.RooRealVar('joint_c'+str(i),'',.03,-.15,.15) for i in range(3)]
    cheby=root.RooChebychev('joint_cheby','',x,root.RooArgList(*coefs))
    children=[wrapper,cheby]
    if reverse:children.reverse()
    pdf=root.RooProdPdf('joint_pdf','',root.RooArgList(*children))
    return locals()


@pytest.mark.parametrize('reverse',[False,True])
def test_joint_native_values_raw_integral_and_density_gradients(root,reverse):
    f=fixture(root,reverse);x=f['x'];pdf=f['pdf'];coefs=f['coefs']
    registry=default_registry();nodes=registry.export([pdf],['joint_x'])
    assert 'joint_spline_cheby' in nodes['joint_pdf'].attrs
    coordinates=np.linspace(111.,159.,9);norm=root.RooArgSet(x);raw_integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,[p.GetName() for p in coefs],theta,{'joint_x':coordinates})
        return jnp.r_[ctx.value('joint_pdf'),ctx.integral('joint_pdf',('joint_x',)),ctx.density('joint_pdf',('joint_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for c,v in zip(coefs,theta,strict=True):c.setVal(float(v))
        raw=[];dens=[]
        for xx in coordinates:
            x.setVal(float(xx));raw.append(pdf.getVal());dens.append(pdf.getVal(norm))
        return np.r_[raw,raw_integral.getVal(),dens]
    points=[np.array([p.getVal() for p in coefs])]+list(np.random.default_rng(451).uniform(-.15,.15,(25,3)))
    for theta in points:
        np.testing.assert_allclose(value(theta),native(theta),atol=2e-12,rtol=3e-12)
        np.testing.assert_allclose(gradient(theta),finite_difference(native,theta,np.full(3,1e-5)).T,atol=2e-8,rtol=2e-6)


def test_joint_tensor_native_nll_and_streamed_gradient_hvp(root):
    root=load_caching_nll();f=fixture(root)
    m=bind_model(root,f['pdf'],[f['x']],{'joint_x':np.linspace(111.,159.,9)})
    check_model(m)
    entry=next(e for e in m['binned'].metadata['program'] if e['operation']=='numeric_linear_basis_integral')
    assert m['binned'].arrays[entry['buffers']['refinement_table']].shape==(1,20,4)
    arrays=m['arrays'];theta=m['old'].initial;direction=jnp.linspace(.2,.7,len(theta))
    reference=jax.jit(jax.value_and_grad(m['function']))(theta,arrays)
    expected=jax.jit(lambda t:jax.jvp(jax.grad(lambda q:m['function'](q,arrays)),(t,),(direction,))[1])(theta)
    for policy in ('structural','structural_sequential'):
        kernel=make_binned_function(m['binned'],gradient=policy)
        def fun(t):return kernel(m['sf'](t,arrays['s']),arrays['b'])[0]
        result=jax.jit(jax.value_and_grad(fun))(theta)
        np.testing.assert_allclose(result[0],reference[0],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(result[1],reference[1],atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda t:jax.jvp(jax.grad(fun),(t,),(direction,))[1])(theta),expected,atol=1e-9,rtol=1e-10)


def test_joint_custom_normalization_rejected(root):
    f=fixture(root);config=root.RooNumIntConfig(f['pdf'].defaultIntegratorConfig())
    config.setEpsAbs(1e-9);f['pdf'].setIntegratorConfig(config)
    with pytest.raises(UnsupportedModelError,match='custom numerical configuration'):
        default_registry().export([f['pdf']],['joint_x'])
