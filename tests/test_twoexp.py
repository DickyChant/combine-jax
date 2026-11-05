from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.twoexp_math import component


def make_twoexp(root, offset='default'):
    x = root.RooRealVar('tx', '', 125., 110., 160.)
    a1 = root.RooRealVar('ta1', '', -.045, -.2, .1)
    a2 = root.RooRealVar('ta2', '', -.025, -.2, .1)
    fraction = root.RooRealVar('tf', '', .63, 0., 1.)
    mean = root.RooRealVar('tm', '', 91.2, 80., 100.)
    args = [x, a1, a2, fraction]
    if offset == 'none': args.append(False)
    elif offset == 'custom': args.append(mean)
    pdf = root.RooSumTwoExpPdf('twoexp', '', *args)
    return dict(x=x, pdf=pdf, variables=[a1, a2, fraction]+([mean] if offset=='custom' else []), held=[mean])


@pytest.mark.parametrize('offset', ['default', 'none', 'custom'])
@pytest.mark.parametrize('bounds', [None, (118., 147.)])
def test_twoexp_native_values_integrals_gradients(root, offset, bounds):
    f = make_twoexp(root, offset);pdf,x,variables=f['pdf'],f['x'],f['variables']
    registry = default_registry();graph = dict(registry.export([pdf], ['tx']))
    if bounds:
        x.setRange('part', *bounds)
        native_integral = pdf.createIntegral(root.RooArgSet(x), root.RooFit.Range('part'))
        graph['tx'] = replace(graph['tx'], attrs={**graph['tx'].attrs, 'minimum':bounds[0], 'maximum':bounds[1]})
    else:
        native_integral = pdf.createIntegral(root.RooArgSet(x))
    coords=np.array([110.1, 118., 125., 145., 159.9]);names=tuple(v.GetName() for v in variables)
    def function(theta):
        context = Evaluation(graph, registry, names, theta, {'tx':coords})
        return jnp.concatenate((context.value('twoexp'), jnp.atleast_1d(context.integral('twoexp', ('tx',)))))
    value=jax.jit(function);jac=jax.jit(jax.jacrev(function))
    def native(theta):
        for v,p in zip(variables,theta,strict=True):v.setVal(float(p))
        result=[]
        for coord in coords:x.setVal(float(coord));result.append(pdf.getVal())
        return np.array(result+[native_integral.getVal()])
    initial=np.array([v.getVal() for v in variables]);rng=np.random.default_rng(8125)
    points=[initial]
    for _ in range(25):
        point=np.array([rng.uniform(-.13,.03),rng.uniform(-.12,.04),rng.uniform(.1,.9)])
        if offset=='custom':point=np.r_[point,rng.uniform(83.,97.)]
        points.append(point)
    for slopes in [(0.,0.),(0.,-.035),(-.04,0.)]:
        point=initial.copy();point[:2]=slopes;points.append(point)
    for point in points:
        np.testing.assert_allclose(value(point),native(point),rtol=3e-13,atol=3e-14)
        columns=[]
        for i in range(len(point)):
            step=np.zeros_like(point);step[i]=1e-5
            columns.append((native(point-2*step)-8*native(point-step)+8*native(point+step)-native(point+2*step))/(12e-5))
        np.testing.assert_allclose(jac(point),np.array(columns).T,rtol=2e-6,atol=5e-8)


def test_twoexp_zero_slope_exact_limiting_gradient_and_hessian():
    low,high=110.,160.;x=jnp.array([111.,125.,150.]);mid=.5*(low+high);width=high-low
    fn=lambda slope:component(x,slope,low,high,91.2,jnp)
    np.testing.assert_allclose(fn(0.),np.full(3,1/width),rtol=0.,atol=0.)
    np.testing.assert_allclose(jax.jacfwd(fn)(0.),(x-mid)/width,rtol=1e-14,atol=1e-14)
    np.testing.assert_allclose(jax.jacfwd(jax.jacfwd(fn))(0.),((x-mid)**2-width**2/12)/width,rtol=1e-13,atol=1e-13)


def test_twoexp_preserves_native_overflow(root):
    f=make_twoexp(root);f['variables'][1].setRange(-10.,10.);f['variables'][1].setVal(10.)
    registry=default_registry();graph=registry.export([f['pdf']],['tx'])
    actual=Evaluation(graph,registry,(),(),{'tx':np.array([125.])}).value('twoexp')
    # RooAbsPdf::getVal(nullptr) masks this evaluation error to zero in ROOT
    # 6.40. Compare the class formula itself, before RooFit's error reporting.
    assert root.gInterpreter.Declare('''
      class TwoExpRawTest : public RooSumTwoExpPdf {
      public:
        TwoExpRawTest(RooSumTwoExpPdf const& p) : RooSumTwoExpPdf(p,"raw_test") {}
        double raw() const { return evaluate(); }
      };
    ''')
    assert not np.isfinite(root.TwoExpRawTest(f['pdf']).raw())
    assert not np.isfinite(np.asarray(actual)).any()
