from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.romberg import RombergConfig,extract_config,integrate_numpy
from combine_jax.romberg_jax import integrate,solve
from combine_jax.validation import finite_difference


@pytest.mark.parametrize('mode',['default','no_extrapolation','fixed','strict'])
def test_native_romberg_integrals_and_derivatives(root,mode):
    x=root.RooRealVar('romberg_x','',1.,-3.,5.)
    mean=root.RooRealVar('romberg_mean','',.2,-1.,1.)
    width=root.RooRealVar('romberg_width','',1.2,.4,2.)
    pdf=root.RooGenericPdf('romberg_pdf','exp(-0.5*pow((@0-@1)/@2,2))',root.RooArgList(x,mean,width))
    settings=root.RooNumIntConfig(pdf.defaultIntegratorConfig())
    section=settings.getConfigSection('RooIntegrator1D')
    if mode=='no_extrapolation':section.find('extrapolation').setLabel('None')
    if mode=='fixed':section.find('fixSteps').setVal(8)
    if mode=='strict':settings.setEpsAbs(1e-12);settings.setEpsRel(1e-12)
    pdf.setIntegratorConfig(settings);config=RombergConfig(**extract_config(pdf))
    norm=pdf.createIntegral(root.RooArgSet(x))
    def native(p):mean.setVal(float(p[0]));width.setVal(float(p[1]));return norm.getVal()
    def shape(p,x):return jnp.exp(-.5*((x-p[0])/p[1])**2)
    def numpy_shape(p,x):return np.exp(-.5*((x-p[0])/p[1])**2)
    function=lambda p:integrate(shape,p,-3.,5.,config)
    value=jax.jit(function);gradient=jax.jit(jax.grad(function));hessian=jax.jit(jax.hessian(function))
    points=[np.array([.2,1.2])]+list(np.random.default_rng(793).uniform([-1.,.4],[1.,2.],(25,2)))
    for index,point in enumerate(points):
        expected=native(point);nvalue,level,mode_number,converged=integrate_numpy(numpy_shape,point,-3.,5.,config)
        actual,steps,actual_mode,done=solve(shape,jnp.r_[point,-3.,5.],config)
        assert converged and bool(done) and int(steps)==level and int(actual_mode)==mode_number
        np.testing.assert_allclose(value(point),expected,rtol=2e-12,atol=2e-12)
        np.testing.assert_allclose([actual,nvalue],expected,rtol=2e-12,atol=2e-12)
        fd=finite_difference(native,point,np.full(2,1e-4))
        np.testing.assert_allclose(gradient(point),fd,rtol=2e-6,atol=2e-8)
        if index<2:
            # The selected native quadrature's AD Hessian versus FD of the
            # already native-validated gradient, and a streamed HVP direction.
            fd_hess=finite_difference(lambda p:np.asarray(gradient(p)),point,np.full(2,2e-4)).T
            np.testing.assert_allclose(hessian(point),fd_hess,rtol=2e-6,atol=2e-8)
            direction=np.array([.3,-.7])
            np.testing.assert_allclose(jax.jvp(jax.grad(function),(point,),(direction,))[1],hessian(point)@direction,atol=2e-11,rtol=2e-11)


def test_romberg_fixed_zero_and_unconverged_modes():
    polynomial=lambda p,x:p[0]*x**4
    config=RombergConfig(max_steps=3,eps_abs=1e-14,eps_rel=1e-14)
    theta=np.array([2.])
    expected,steps,mode,done=integrate_numpy(polynomial,theta,0.,1.,config)
    assert steps==3 and mode==0 and not done
    np.testing.assert_allclose(integrate(polynomial,theta,0.,1.,config),expected,atol=1e-15)
    np.testing.assert_allclose(jax.grad(lambda p:integrate(polynomial,p,0.,1.,config))(theta),[expected/2.],atol=1e-15)
    zero_config=replace(config,min_steps_zero=1)
    zero=lambda p,x:jnp.zeros_like(x)*p[0]
    value,steps,mode,done=solve(zero,jnp.array([2.,0.,1.]),zero_config)
    assert value==0. and steps==1 and mode==2 and done
    np.testing.assert_array_equal(jax.grad(lambda p:integrate(zero,p,0.,1.,zero_config))(theta),[0.])


def test_romberg_batching_and_runtime_geometry():
    shape=lambda p,x:jnp.exp(-p[0]*x)
    # This test compares to the mathematical integral, so explicitly tighten
    # the integration accuracy. Default native tolerances are tested above;
    # at eps=1e-7 their integral legitimately differs by 1.53e-10 here.
    config=RombergConfig(eps_abs=1e-12,eps_rel=1e-12)
    function=lambda p,lo,hi:integrate(shape,p,lo,hi,config)
    params=jnp.array([[.2],[.4],[.7]])
    low=jnp.array([0.,1.,2.]);high=jnp.array([2.,4.,6.])
    actual=jax.jit(jax.vmap(function))(params,low,high)
    expected=(jnp.exp(-params[:,0]*low)-jnp.exp(-params[:,0]*high))/params[:,0]
    np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12)
    derivatives=jax.jit(jax.vmap(jax.grad(function,argnums=(1,2))))(params,low,high)
    np.testing.assert_allclose(derivatives,(-jnp.exp(-params[:,0]*low),jnp.exp(-params[:,0]*high)),atol=2e-11,rtol=2e-11)
