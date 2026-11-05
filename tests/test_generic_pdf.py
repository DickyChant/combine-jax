import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize('kind',['landau','breit_wigner','photon','fixed_landau'])
def test_generic_native_values_normalization_and_gradient(root,kind):
    if 'landau' in kind:
        x=root.RooRealVar('gen_x','',.2,-6.,30.)
        p=[root.RooRealVar('gen_mean','',.2,-.5,.5),root.RooRealVar('gen_width','',1.2,.8,2.)]
        expression='TMath::Landau(@0,@1,@2)'
    else:
        x=root.RooRealVar('gen_x','',125.,110.,150.)
        p=[root.RooRealVar('gen_power','',-.3,-1.2,-.05)]
        expression='2.5*exp(@0*@1/100.0)/(pow(@0-91.2,2)+pow(2.5/2,2))'
        if kind=='photon':
            p.append(root.RooRealVar('gen_fraction','',.2,.01,.5))
            expression='2.5*exp(@1*@0/100)/(pow(@0-91,2)+pow(2.5/2.0,2))+1000*@2*@2*exp(@1*@0)/pow(@0,2)'
    if kind=='fixed_landau':
        for variable in p:variable.setConstant(True)
    pdf=root.RooGenericPdf('gen_pdf',expression,root.RooArgList(x,*p))
    registry=default_registry();nodes=registry.export([pdf],['gen_x'])
    parameters=[] if kind=='fixed_landau' else p;names=tuple(v.GetName() for v in parameters)
    coords=np.linspace(x.getMin(),x.getMax(),9);norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        context=Evaluation(nodes,registry,names,theta,{'gen_x':coords})
        return jnp.r_[context.value('gen_pdf'),context.integral('gen_pdf',('gen_x',)),context.density('gen_pdf',('gen_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for var,v in zip(parameters,theta,strict=True):var.setVal(float(v))
        raw=[];density=[]
        for coordinate in coords:x.setVal(float(coordinate));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.r_[raw,integral.getVal(),density]
    points=[np.array([v.getVal() for v in parameters])]
    if parameters:points.extend(np.random.default_rng(912).uniform([v.getMin() for v in parameters],[v.getMax() for v in parameters],(25,len(parameters))))
    for point in points:
        np.testing.assert_allclose(value(point),native(point),atol=2e-12,rtol=3e-12)
        if parameters:
            fd=finite_difference(native,point,np.full(len(point),1e-5)).T
            np.testing.assert_allclose(gradient(point),fd,atol=2e-8,rtol=2e-6)
    assert bool(nodes['gen_pdf'].attrs['fixed_normalizations'])==(kind=='fixed_landau')
