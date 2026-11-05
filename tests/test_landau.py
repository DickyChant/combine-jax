import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax import landau_math
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_landau(root,bounds=(-8.,400.)):
    x=root.RooRealVar('landau_x','',1.,*bounds)
    mean=root.RooRealVar('landau_mean','',.2,-.7,.7)
    width=root.RooRealVar('landau_width','',1.1,.5,2.)
    pdf=root.RooLandau('landau_pdf','',x,mean,width)
    return dict(x=x,variables=[mean,width],pdf=pdf)


def branch_safe_steps(theta,coords,bounds):
    """Keep the native five-point stencil inside its piecewise formula branch."""
    def branches(t):
        return (np.searchsorted([-5.5,-1.,1.,5.,12.,50.,300.],(coords-t[0])/t[1],side='right'),
                np.searchsorted([-5.5,-1.,1.,4.,12.,50.,300.],(np.array(bounds)-t[0])/t[1],side='right'))
    center=branches(theta);steps=np.full(2,2e-4)
    for i in range(2):
        for attempt in range(12):
            shift=np.zeros(2);shift[i]=2*steps[i]
            if all(np.array_equal(a,b) for direction in (-1,1)
                   for a,b in zip(center,branches(theta+direction*shift),strict=True)):
                break
            steps[i]*=.25
        else:raise AssertionError('FD point lies on a native Landau branch boundary')
    return steps


@pytest.mark.parametrize('cumulative',[False,True])
def test_landau_math_all_native_branches_and_boundaries(root,cumulative):
    edges=np.array([-5.5,-1.,1.,4.,5.,12.,50.,300.])
    points=np.r_[-1000.,-100.,-24.,-9.,-7.,-3.,0.,2.,7.,20.,100.,1000.,1e8,
                 edges-1e-8,edges,edges+1e-8]
    function=jax.jit(lambda v:landau_math.standard(v,jnp,cumulative=cumulative))
    native=root.Math.landau_cdf if cumulative else root.Math.landau_pdf
    expected=np.array([native(float(v)) for v in points])
    np.testing.assert_allclose(function(points),expected,atol=1e-15,rtol=3e-13)
    np.testing.assert_allclose(landau_math.standard(points,np,cumulative=cumulative),expected,atol=1e-15,rtol=3e-13)
    # FD points avoid branch jumps; boundary values above are checked exactly.
    smooth=np.array([-8.,-6.,-3.,-.3,2.,7.,20.,100.,500.])
    derivative=jax.jit(jax.vmap(jax.grad(lambda v:landau_math.standard(v,jnp,cumulative=cumulative))))
    step=1e-3
    fd=np.array([(native(float(v-2*step))-8*native(float(v-step))
                 +8*native(float(v+step))-native(float(v+2*step)))/(12*step) for v in smooth])
    np.testing.assert_allclose(derivative(smooth),fd,rtol=2e-6,atol=2e-11)


@pytest.mark.parametrize('bounds',[(-8.,400.),(-5.,3.),(10.,50.),(300.,1000.)])
def test_landau_registry_native_values_integrals_and_gradient(root,bounds):
    f=make_landau(root,bounds);x=f['x'];variables=f['variables']
    registry=default_registry();nodes=registry.export([f['pdf']],['landau_x'])
    coords=np.linspace(*bounds,9);norm=root.RooArgSet(x);integral=f['pdf'].createIntegral(norm)
    names=tuple(v.GetName() for v in variables)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,{'landau_x':coords})
        return jnp.r_[ctx.value('landau_pdf'),ctx.integral('landau_pdf',('landau_x',)),ctx.density('landau_pdf',('landau_x',))]
    value=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def native(theta):
        for var,v in zip(variables,theta,strict=True):var.setVal(float(v))
        raw=[];density=[]
        for coordinate in coords:
            x.setVal(float(coordinate));raw.append(f['pdf'].getVal());density.append(f['pdf'].getVal(norm))
        return np.r_[raw,integral.getVal(),density]
    points=[np.array([v.getVal() for v in variables])]
    points.extend(np.random.default_rng(762).uniform([v.getMin() for v in variables],[v.getMax() for v in variables],(25,2)))
    for theta in points:
        np.testing.assert_allclose(value(theta),native(theta),atol=2e-13,rtol=3e-12)
        steps=branch_safe_steps(theta,coords,bounds)
        for factor in ((1.,.5) if np.any(steps<2e-4) else (1.,)):
            fd=finite_difference(native,theta,steps*factor).T
            np.testing.assert_allclose(gradient(theta),fd,atol=2e-8,rtol=2e-6)


def test_landau_nonpositive_width_and_cdf_derivative_convention(root):
    f=make_landau(root);f['variables'][1].setRange(-2.,2.)
    for sigma in (-1.,0.):
        f['variables'][1].setVal(sigma)
        assert f['pdf'].getVal()==0.
        assert float(landau_math.value(.3,.2,sigma,jnp))==0.
    # ROOT normalizes with a separately parameterized DISLAN expression, whose
    # derivative is not exactly its DENLAN PDF. Preserve that native behavior.
    derivative=float(jax.grad(lambda v:landau_math.standard(v,jnp,cumulative=True))(.2))
    assert abs(derivative-root.Math.landau_pdf(.2))>1e-10
