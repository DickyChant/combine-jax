import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize('mode',['a','ac','abc','abcm','abcmw','bernstein','abc_bernstein'])
def test_modz_native_value_integral_gradient(root,mode):
    x=root.RooRealVar('modz_x','',125.,110.,160.)
    specs=[('a',-.03,-.06,.01),('b',1e-4,-2e-4,2e-4),('c',2.,1.3,2.8),('m',91.2,88.,94.),('w',2.5,1.8,3.2)]
    variables={n:root.RooRealVar('modz_'+n,'',v,lo,hi) for n,v,lo,hi in specs}
    names={'a':'a','ac':'ac','abc':'abc','abcm':'abcm','abcmw':'abcmw','bernstein':'a','abc_bernstein':'abc'}[mode]
    parameters=[variables[n] for n in names];arguments=[x,*parameters]
    coefficients=[]
    if 'bernstein' in mode:
        coefficients=[root.RooRealVar('modz_coef'+str(i),'',.01,-.025,.025) for i in range(3)]
        arguments.append(root.RooArgList(*coefficients));parameters.extend(coefficients)
    pdf=root.RooModZPdf('modz_pdf','',*arguments)
    registry=default_registry();nodes=registry.export([pdf],['modz_x'])
    names=tuple(v.GetName() for v in parameters);coords=np.linspace(110.,160.,9)
    norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,{'modz_x':coords})
        return jnp.r_[ctx.value('modz_pdf'),ctx.integral('modz_pdf',('modz_x',)),ctx.density('modz_pdf',('modz_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for var,v in zip(parameters,theta,strict=True):var.setVal(float(v))
        raw=[];density=[]
        for coordinate in coords:x.setVal(float(coordinate));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.r_[raw,integral.getVal(),density]
    points=[np.array([v.getVal() for v in parameters])]+list(np.random.default_rng(957).uniform(
        [v.getMin() for v in parameters],[v.getMax() for v in parameters],(25,len(parameters))))
    steps=np.array([1e-8 if name=='modz_b' else 1e-5 for name in names])
    for point in points:
        np.testing.assert_allclose(value(point),native(point),atol=2e-12,rtol=3e-12)
        fd=finite_difference(native,point,steps).T
        np.testing.assert_allclose(gradient(point),fd,atol=2e-8,rtol=2e-6)
