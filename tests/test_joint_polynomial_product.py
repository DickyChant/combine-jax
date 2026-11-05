"""Nested spline/Bernstein/Chebychev normalization against ROOT."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import subprocess
import sys

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.binned_channels import make_binned_function
from combine_jax.tensor.numeric_shapes import validate_numeric_fields
from test_joint_spline_product import fixture as spline_fixture
from test_tensor_binned_channels import bind_model, check_model


def fixture(root, nested=True, reverse=False):
    held=spline_fixture(root);x=held['x']
    coefs=[root.RooRealVar('jb_c'+str(i),'',1.+.1*i,.7,1.3) for i in range(3)]
    bern=root.RooBernsteinFast[3]('jb_bern','',x,root.RooArgList(*coefs))
    inner=root.RooProdPdf('jb_inner','',root.RooArgList(held['wrapper'],bern))
    children=[inner,held['cheby']]
    if reverse:children.reverse()
    pdf=root.RooProdPdf('jb_outer','',root.RooArgList(*children)) if nested else inner
    parameters=[*coefs,*held['coefs']] if nested else coefs
    return locals()


@pytest.mark.parametrize('nested,reverse',[(False,False),(True,False),(True,True)])
def test_polynomial_joint_native_raw_integral_density_gradient(root,nested,reverse):
    f=fixture(root,nested,reverse);pdf=f['pdf'];x=f['x'];params=f['parameters']
    registry=default_registry();nodes=registry.export([pdf],['joint_x'])
    attributes=nodes[pdf.GetName()].attrs['joint_spline_cheby']
    assert tuple(attributes['coefficient_groups'])==((3,3) if nested else (3,))
    assert attributes['refinement_table'].shape==(20,16 if nested else 4)
    coordinates=np.linspace(111.,159.,9);norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,[p.GetName() for p in params],theta,{'joint_x':coordinates})
        return jnp.r_[ctx.value(pdf.GetName()),ctx.integral(pdf.GetName(),('joint_x',)),ctx.density(pdf.GetName(),('joint_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for p,v in zip(params,theta,strict=True):p.setVal(float(v))
        raw=[];density=[]
        for xx in coordinates:x.setVal(float(xx));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.r_[raw,integral.getVal(),density]
    points=[np.array([p.getVal() for p in params])]+list(np.random.default_rng(962).uniform(
        [p.getMin() for p in params],[p.getMax() for p in params],(25,len(params))))
    for i,theta in enumerate(points):
        expected=native(theta)
        np.testing.assert_allclose(value(theta),expected,atol=2e-12,rtol=3e-12)
        if i<5:
            # Subtract the local reference value before forming the same FD
            # stencil, so the large raw product integral cannot create a
            # spurious derivative of its parameter-independent terms.
            fd=finite_difference(lambda p:native(p)-expected,theta,np.full(len(theta),2e-4)).T
            np.testing.assert_allclose(gradient(theta),fd,atol=2e-8,rtol=2e-6)


def test_nested_polynomial_tensor_nll_gradient_hessian_hvp(root):
    root=load_caching_nll();f=fixture(root)
    model=bind_model(root,f['pdf'],[f['x']],{'joint_x':np.linspace(111.,159.,9)},weights=np.ones(9))
    check_model(model)
    batch=next(b for b in model['binned'].metadata['program'] if b['operation']=='numeric_product_basis_integral')
    assert model['binned'].arrays[batch['buffers']['refinement_table']].shape==(1,20,16)
    arrays=model['arrays'];theta=model['old'].initial;direction=jnp.linspace(.1,.3,len(theta))
    reference=jax.jit(jax.value_and_grad(model['function']))(theta,arrays)
    expected=jax.jit(lambda t:jax.jvp(jax.grad(lambda q:model['function'](q,arrays)),(t,),(direction,))[1])(theta)
    for policy in ('structural','structural_sequential'):
        kernel=make_binned_function(model['binned'],gradient=policy)
        def fun(t):return kernel(model['sf'](t,arrays['s']),arrays['b'])[0]
        result=jax.jit(jax.value_and_grad(fun))(theta)
        np.testing.assert_allclose(result[0],reference[0],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(result[1],reference[1],atol=1e-9,rtol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda t:jax.jvp(jax.grad(fun),(t,),(direction,))[1])(theta),expected,atol=1e-9,rtol=1e-10)


def test_nested_polynomial_transport_and_invalid_geometry(root,tmp_path):
    root=load_caching_nll();f=fixture(root)
    m=bind_model(root,f['pdf'],[f['x']],{'joint_x':np.array([112.,125.,148.])},weights=np.ones(3))
    m['scalar'].save(tmp_path/'scalar');m['binned'].save(tmp_path/'binned')
    script='''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalar');b=TensorBundle.load(p/'binned')
value=evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))
np.testing.assert_allclose(value,[float(sys.argv[2])],atol=1e-10,rtol=0.)
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path),str(m['native'](m['old'].initial))],
                   check=True,capture_output=True)
    batch=next(b for b in m['binned'].metadata['program'] if b['operation']=='numeric_product_basis_integral')
    buffers={name:m['binned'].arrays[key] for name,key in batch['buffers'].items()}
    for groups in ([-1,7],[3,2],[3.,3],None):
        with pytest.raises(ValueError,match='Invalid polynomial coefficient groups'):
            validate_numeric_fields(batch['operation'],buffers,{**batch['spec'],'coefficient_groups':groups},1)
    with pytest.raises(ValueError,match='Invalid product integration basis table'):
        validate_numeric_fields(batch['operation'],{**buffers,'refinement_table':buffers['refinement_table'][...,:-1]},batch['spec'],1)
