import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize('degree',[0,1,3,7])
@pytest.mark.parametrize('reference_range',[False,True])
def test_chebyshev_native_polynomial_integral_gradient_roundtrip(root,tmp_path,degree,reference_range):
    x=root.RooRealVar('chx','',.1,-1.,3.)
    coefs=[root.RooRealVar(f'chc{i}','',.02,-.04,.04) for i in range(degree)]
    pdf=root.RooChebychev('cheby','',x,root.RooArgList(*coefs))
    if reference_range:
        x.setRange('reference',-2.,4.);pdf.selectNormalizationRange('reference',True)
    w=root.RooWorkspace('w');getattr(w,'import')(pdf)
    path=tmp_path/'cheby.root';w.writeToFile(str(path));handle=root.TFile.Open(str(path));w=handle.Get('w')
    x,pdf=w.var('chx'),w.pdf('cheby');coefs=[w.var(f'chc{i}') for i in range(degree)]
    registry=default_registry();nodes=registry.export([pdf],['chx']);names=tuple(v.GetName() for v in coefs)
    data={'chx':np.linspace(-1.,3.,13)};norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,data)
        return ctx.value('cheby'),ctx.integral('cheby',('chx',)),ctx.density('cheby',('chx',))
    function=jax.jit(function);gradient=jax.jit(jax.jacrev(lambda t:function(t)[2]))
    def reference(theta):
        for variable,value in zip(coefs,theta,strict=True):variable.setVal(float(value))
        raw,density=[],[]
        for coordinate in data['chx']:
            x.setVal(float(coordinate));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.array(raw),integral.getVal(),np.array(density)
    for point in np.random.default_rng(411).uniform(-.039,.039,(25,degree)):
        for actual,expected in zip(function(point),reference(point),strict=True):
            np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-14)
        if degree:
            fd=finite_difference(lambda t:reference(t)[2],point,np.full(degree,2e-4)).T
            np.testing.assert_allclose(gradient(point),fd,rtol=1e-7,atol=1e-10)
    handle.Close()


@pytest.mark.parametrize('degree',[0,3])
def test_chebyshev_tensor_native_nll_gradients_hessian(root,degree):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model,check_model
    root=load_caching_nll();x=root.RooRealVar('chx','',.1,-1.,3.)
    coefs=[root.RooRealVar(f'chc{i}','',.02,-.04,.04) for i in range(degree)]
    pdf=root.RooChebychev('cheby','',x,root.RooArgList(*coefs))
    x.setRange('reference',-2.,4.);pdf.selectNormalizationRange('reference',True)
    model=bind_model(root,pdf,[x],{'chx':np.linspace(-.9,2.9,9)})
    check_model(model)


@pytest.mark.parametrize('mode',['numerical','range'])
def test_chebyshev_unsupported_normalization_explicit(root,mode):
    x=root.RooRealVar('chx','',.1,-1.,3.);c=root.RooConstVar('chc','',.1)
    pdf=root.RooChebychev('cheby','',x,root.RooArgList(c))
    if mode=='numerical':pdf.forceNumInt(True)
    else:x.setRange('subset',0.,1.);pdf.setNormRange('subset')
    with pytest.raises(UnsupportedModelError,match='forced numerical or named-range'):
        default_registry().export([pdf])
