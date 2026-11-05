import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_cbshape(root, bounds=(-10.,12.), *, negative_alpha=False, negative_sigma=False):
    x=root.RooRealVar('cbx','',.5*(bounds[0]+bounds[1]),*bounds)
    specs=[('mean',.2,-.7,.7),('sigma',1.1,.5,2.),('alpha',1.7,.8,2.5),('power',2.4,1.2,5.)]
    variables=[]
    for name,value,lo,hi in specs:
        if (name=='alpha' and negative_alpha) or (name=='sigma' and negative_sigma):value,lo,hi=-value,-hi,-lo
        variables.append(root.RooRealVar('cb_'+name,'',value,lo,hi))
    pdf=root.RooCBShape('cbshape','',x,*variables)
    return dict(x=x,pdf=pdf,variables=variables)


@pytest.mark.parametrize('bounds,negative_alpha,negative_sigma',[
    ((-10.,12.),False,False),((-12.,-6.),False,False),((6.,12.),False,False),
    ((-.1,.4),False,False),((-10.,12.),True,False),((-10.,12.),False,True),
])
def test_cbshape_native_value_integral_and_gradient(root,bounds,negative_alpha,negative_sigma):
    f=make_cbshape(root,bounds,negative_alpha=negative_alpha,negative_sigma=negative_sigma)
    variables=f['variables'];names=tuple(v.GetName() for v in variables)
    registry=default_registry();nodes=registry.export([f['pdf']],['cbx'])
    coords=np.linspace(*bounds,9);norm=root.RooArgSet(f['x']);integral=f['pdf'].createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,{'cbx':coords})
        return jnp.r_[ctx.value('cbshape'),ctx.integral('cbshape',('cbx',)),ctx.density('cbshape',('cbx',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for v,p in zip(variables,theta,strict=True):v.setVal(float(p))
        raw,densities=[],[]
        for x in coords:
            f['x'].setVal(float(x));raw.append(f['pdf'].getVal());densities.append(f['pdf'].getVal(norm))
        return np.r_[raw,integral.getVal(),densities]
    points=[np.array([v.getVal() for v in variables])]
    points.extend(np.random.default_rng(2028).uniform([v.getMin() for v in variables],
                  [v.getMax() for v in variables],(25,len(variables))))
    for point in points:
        baseline=native(point)
        np.testing.assert_allclose(value(point),baseline,rtol=3e-12,atol=2e-13)
        # ROOT's saturated erf may floor the integral to 1e-300. Subtracting
        # the center before differencing avoids spurious 1e265 derivatives
        # of exactly constant 1e279 densities in unrelated parameters.
        # In this far-tail-only window the native integral is either the
        # 1e-300 floor or a cancellation of erf values near one. Normalized
        # density differences then fail step-stability (recorded diagnostic),
        # so only raw-value/integral derivatives have an adequate FD oracle.
        # Density VALUES still pass the full check above at every point.
        stop=len(coords)+1 if bounds==(6.,12.) else None
        steps=(1e-5,2e-5) if stop else (2e-4,)
        for step in steps:
            fd=finite_difference(lambda t:native(t)-baseline,point,np.full(len(point),step)).T
            np.testing.assert_allclose(gradient(point)[:stop],fd[:stop],rtol=2e-6,atol=1e-8)


@pytest.mark.parametrize('power',[1.-1e-6,1.,1.+1e-6])
def test_cbshape_native_near_one_integral_and_derivative(root,power):
    f=make_cbshape(root);f['variables'][3].setRange(.1,5.);f['variables'][3].setVal(power)
    norm=root.RooArgSet(f['x'])
    for lo,hi in [(-12.,-6.),(-10.,12.),(6.,12.)]:
        f['x'].setRange(lo,hi);registry=default_registry();nodes=registry.export([f['pdf']])
        integral=f['pdf'].createIntegral(norm)
        fn=jax.jit(lambda t:Evaluation(nodes,registry,('cb_power',),t).integral('cbshape',('cbx',)))
        def native(t):f['variables'][3].setVal(float(t[0]));return integral.getVal()
        point=np.array([power])
        np.testing.assert_allclose(fn(point),native(point),rtol=3e-12,atol=2e-13)
        fd=finite_difference(native,point,np.array([1e-7]))
        np.testing.assert_allclose(jax.grad(fn)(point),fd,rtol=1e-6,atol=2e-8)


def test_cbshape_conventions_are_explicit_and_differ(root):
    f=make_cbshape(root,(-12.,-6.));f['variables'][3].setRange(.1,5.);f['variables'][3].setVal(1.+5e-6)
    results=[]
    for convention in ('root630','root640'):
        registry=default_registry(cbshape_convention=convention);nodes=registry.export([f['pdf']])
        assert nodes['cbshape'].attrs['convention']==convention
        results.append(float(Evaluation(nodes,registry,(),()).integral('cbshape',('cbx',))))
    assert abs(results[0]-results[1])>1e-7
