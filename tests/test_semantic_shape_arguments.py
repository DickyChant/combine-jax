"""Proxy bookkeeping labels cannot replace a PDF's actual argument members."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def relabel(pdf):
    for i in range(pdf.numProxies()):
        proxy=pdf.getProxy(i)
        rename=proxy.SetName if hasattr(proxy,'SetName') else proxy.setName
        rename('stored_proxy_'+str(i))


@pytest.mark.parametrize('kind',['cbshape','landau','chebyshev'])
def test_semantic_shape_arguments_ignore_proxy_labels(root,kind):
    x=root.RooRealVar('semantic_x','',.4,-4.,5.)
    specs={'cbshape':[('mean',.2,-.5,.5),('sigma',1.1,.8,1.4),('alpha',1.7,1.4,2.),('power',2.4,2.,3.)],
           'landau':[('mean',.2,-.5,.5),('sigma',1.1,.8,1.4)],
           'chebyshev':[('a',.1,-.15,.15),('b',.03,-.1,.1)]}[kind]
    parameters=[root.RooRealVar('semantic_'+name,'',v,lo,hi) for name,v,lo,hi in specs]
    pdf=(root.RooCBShape('semantic_pdf','',x,*parameters) if kind=='cbshape' else
         root.RooLandau('semantic_pdf','',x,*parameters) if kind=='landau' else
         root.RooChebychev('semantic_pdf','',x,root.RooArgList(*parameters)))
    norm=root.RooArgSet(x);native_integral=pdf.createIntegral(norm)
    registry=default_registry();before=registry.export([pdf],['semantic_x'])
    relabel(pdf)
    graph=registry.export([pdf],['semantic_x'])
    assert graph['semantic_pdf'].inputs==before['semantic_pdf'].inputs
    coords=np.array([-.33,.37,2.41]);names=[p.GetName() for p in parameters]
    def function(theta):
        c=Evaluation(graph,registry,names,theta,{'semantic_x':coords})
        return jnp.r_[c.value('semantic_pdf'),c.integral('semantic_pdf',('semantic_x',)),c.density('semantic_pdf',('semantic_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for p,v in zip(parameters,theta,strict=True):p.setVal(float(v))
        raw=[];dens=[]
        for v in coords:x.setVal(float(v));raw.append(pdf.getVal());dens.append(pdf.getVal(norm))
        return np.r_[raw,native_integral.getVal(),dens]
    points=[np.array([p.getVal() for p in parameters])]
    points.extend(np.random.default_rng(6501).uniform([p.getMin() for p in parameters],
        [p.getMax() for p in parameters],(25,len(parameters))))
    for point in points:
        np.testing.assert_allclose(value(point),native(point),atol=2e-13,rtol=3e-12)
        steps=np.full(len(point),1e-5)
        if kind=='landau':
            from test_landau import branch_safe_steps
            steps=np.minimum(steps,branch_safe_steps(point,coords,(-4.,5.)))
        np.testing.assert_allclose(gradient(point),finite_difference(native,point,steps).T,atol=2e-8,rtol=2e-6)


def test_joint_spline_uses_semantic_chebyshev_coefficients(root,monkeypatch):
    import test_joint_spline_product as joint
    original=joint.fixture
    def fixture(*args,**kwargs):
        f=original(*args,**kwargs);relabel(f['cheby']);return f
    monkeypatch.setattr(joint,'fixture',fixture)
    joint.test_joint_native_values_raw_integral_and_density_gradients(root,False)
