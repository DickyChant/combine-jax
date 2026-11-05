"""Joint raw-product normalization keeps the native Romberg procedure."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import subprocess
import sys

from combine_jax.registry import Evaluation
from combine_jax.graph import UnsupportedModelError
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.binned_channels import make_binned_function
from test_tensor_binned_channels import bind_model, check_model


def fixture(root, kind, reverse=False):
    x=root.RooRealVar('jn_x','',125.,110.,150.)
    a=root.RooRealVar('jn_a','',-.03,-.05,-.01)
    if kind=='modz':
        b=root.RooRealVar('jn_b','',1.,-2.,2.)
        beta=root.RooFormulaVar('jn_beta','1e-5*@0',root.RooArgList(b))
        c=root.RooRealVar('jn_c','',2.,1.8,2.2)
        base=root.RooModZPdf('jn_base','',x,a,beta,c)
    else:
        b=root.RooRealVar('jn_b','',-.015,-.025,-.005)
        c=root.RooRealVar('jn_c','',.4,.2,.8)
        base=root.RooSumTwoExpPdf('jn_base','',x,a,b,c)
    coefs=[root.RooRealVar('jn_coef'+str(i),'',.02,-.05,.05) for i in range(3)]
    cheby=root.RooChebychev('jn_cheby','',x,root.RooArgList(*coefs))
    children=[base,cheby]
    if reverse:children.reverse()
    pdf=root.RooProdPdf('jn_pdf','',root.RooArgList(*children))
    return locals()


@pytest.mark.parametrize('kind', ['modz','twoexp'])
@pytest.mark.parametrize('reverse', [False,True])
def test_joint_values_integral_density_gradient(root,kind,reverse):
    f=fixture(root,kind,reverse);pdf=f['pdf'];x=f['x'];parameters=[f['a'],f['b'],f['c'],*f['coefs']]
    names=[p.GetName() for p in parameters];registry=default_registry();nodes=registry.export([pdf],['jn_x'])
    assert nodes['jn_pdf'].attrs['joint_numeric_product']['shape']==kind
    coordinates=np.linspace(111.,149.,9);norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,{'jn_x':coordinates})
        return jnp.r_[ctx.value('jn_pdf'),ctx.integral('jn_pdf',('jn_x',)),ctx.density('jn_pdf',('jn_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for p,v in zip(parameters,theta,strict=True):p.setVal(float(v))
        raw=[];density=[]
        for xx in coordinates:x.setVal(float(xx));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.r_[raw,integral.getVal(),density]
    rng=np.random.default_rng(631)
    points=[np.array([p.getVal() for p in parameters])]+list(rng.uniform([p.getMin() for p in parameters],[p.getMax() for p in parameters],(25,len(parameters))))
    for i,theta in enumerate(points):
        np.testing.assert_allclose(value(theta),native(theta),atol=2e-12,rtol=3e-12)
        if i<5:
            steps=np.full(len(theta),1e-5)
            np.testing.assert_allclose(gradient(theta),finite_difference(native,theta,steps).T,atol=2e-8,rtol=2e-6)


@pytest.mark.parametrize('kind', ['modz','twoexp'])
def test_joint_tensor_cached_nll_and_streamed_hvp(root,kind):
    root=load_caching_nll();f=fixture(root,kind)
    m=bind_model(root,f['pdf'],[f['x']],{'jn_x':np.linspace(111.,149.,9)},weights=np.ones(9))
    check_model(m)
    theta=m['old'].initial;arrays=m['arrays'];direction=jnp.linspace(.1,.3,len(theta))
    reference=jax.jit(jax.value_and_grad(m['function']))(theta,arrays)
    expected=jax.jit(lambda t:jax.jvp(jax.grad(lambda q:m['function'](q,arrays)),(t,),(direction,))[1])(theta)
    for policy in ('structural','structural_sequential'):
        kernel=make_binned_function(m['binned'],gradient=policy)
        def fun(t):return kernel(m['sf'](t,arrays['s']),arrays['b'])[0]
        result=jax.jit(jax.value_and_grad(fun))(theta)
        np.testing.assert_allclose(result[0],reference[0],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(result[1],reference[1],atol=1e-8,rtol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda t:jax.jvp(jax.grad(fun),(t,),(direction,))[1])(theta),expected,atol=1e-7,rtol=1e-9)


def test_joint_transport_evaluates_without_root_or_jax(root,tmp_path):
    root=load_caching_nll();f=fixture(root,'modz')
    m=bind_model(root,f['pdf'],[f['x']],{'jn_x':np.array([112.,125.,148.])},weights=np.ones(3))
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


def test_joint_custom_native_integrator_is_explicitly_rejected(root):
    f=fixture(root,'twoexp');config=root.RooNumIntConfig(f['pdf'].defaultIntegratorConfig())
    config.setEpsAbs(1e-9);f['pdf'].setIntegratorConfig(config)
    with pytest.raises(UnsupportedModelError,match='custom numerical configuration'):
        default_registry().export([f['pdf']],['jn_x'])
