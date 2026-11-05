"""The fast DoubleCB has distinct native tail and far-range integral branches."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_fast_cb(root, bounds=(-10., 12.)):
    x = root.RooRealVar('fcx', '', 0., *bounds)
    config = [('mean', .2, -.7, .7), ('width', 1.1, .5, 2.), ('alpha1', 1.8, .8, 2.5),
              ('n1', 2.2, 1.2, 5.), ('alpha2', 2.1, .8, 2.5), ('n2', 3.4, 1.2, 5.)]
    variables = [root.RooRealVar('fc'+name, '', value, low, high) for name,value,low,high in config]
    pdf = root.RooDoubleCBFast('fastcb', '', x, *variables)
    return locals()


@pytest.mark.parametrize('bounds', [(-10.,12.),(-12.,-6.),(6.,12.),(-.1,.4),(-3000.,4000.)])
def test_fast_cb_native_value_integral_gradients(root, bounds):
    f = make_fast_cb(root,bounds);x,pdf,variables = [f[n] for n in ('x','pdf','variables')]
    names = tuple(v.GetName() for v in variables)
    coordinates = {'fcx': np.linspace(*bounds,9)}
    registry = default_registry();nodes = registry.export([pdf],['fcx'])
    norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    def function(theta):
        ctx=Evaluation(nodes,registry,names,theta,coordinates)
        return jnp.concatenate((jnp.atleast_1d(ctx.integral('fastcb',('fcx',))),ctx.value('fastcb'),ctx.density('fastcb',('fcx',))))
    function=jax.jit(function);gradient=jax.jit(jax.jacrev(function))
    def reference(theta):
        for variable,value in zip(variables,theta,strict=True):variable.setVal(float(value))
        raw,density=[],[]
        for value in coordinates['fcx']:
            x.setVal(float(value));raw.append(pdf.getVal());density.append(pdf.getVal(norm))
        return np.array([integral.getVal(),*raw,*density])
    for theta in np.random.default_rng(2013).uniform([v.getMin() for v in variables],[v.getMax() for v in variables],(25,len(variables))):
        np.testing.assert_allclose(function(theta),reference(theta),rtol=3e-12,atol=2e-13)
        fd=finite_difference(reference,theta,np.full(len(theta),2e-4)).T
        np.testing.assert_allclose(gradient(theta),fd,rtol=2e-6,atol=1e-8)


@pytest.mark.parametrize('power', [1.-1e-6,1.,1.+1e-6,1.2,2.])
def test_fast_cb_log_branch_and_far_tail_threshold(root,power):
    f=make_fast_cb(root);pdf,x,variables=[f[n] for n in ('pdf','x','variables')]
    for index in (3,5):variables[index].setRange(.1,5.);variables[index].setVal(power)
    variables[0].setVal(0.);variables[1].setVal(1.)
    registry=default_registry()
    for low,high in [(-20.,30.),(-1000.,20.),(-1000.01,20.),(-20.,1000.),(-20.,1000.01),(-2000.,3000.)]:
        x.setRange(low,high);nodes=registry.export([pdf]);norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
        function=jax.jit(lambda t:Evaluation(nodes,registry,('fcn1','fcn2'),t).integral('fastcb',('fcx',)))
        np.testing.assert_allclose(function(np.array([power,power])),integral.getVal(),rtol=3e-12,atol=2e-13)
        if abs(power-1)<1e-5 and low==-20. and high==30.:
            def ref(t):
                variables[3].setVal(float(t[0]));variables[5].setVal(float(t[1]));return integral.getVal()
            fd=finite_difference(ref,np.array([power,power]),np.full(2,1e-7))
            np.testing.assert_allclose(jax.grad(function)(np.array([power,power])),fd,rtol=1e-6,atol=2e-8)
            ref([power,power])


def test_fast_cb_native_nonfinite_intermediate_is_not_repaired(root):
    f=make_fast_cb(root);pdf,variables=f['pdf'],f['variables']
    variables[3].setRange(.1,5.);variables[3].setVal(1.00002)
    variables[2].setVal(.8)
    registry=default_registry();nodes=registry.export([pdf]);ctx=Evaluation(nodes,registry,(),())
    integral=pdf.createIntegral(root.RooArgSet(f['x']))
    # n just outside the logarithmic branch can underflow the source's
    # intermediate fast_pow. Retain a nonfinite result instead of smoothing it.
    assert not np.isfinite(integral.getVal())
    assert not np.isfinite(ctx.integral('fastcb',('fcx',)))


@pytest.mark.parametrize('conditional',[False,True])
def test_fast_cb_tensor_native_nll_gradients_hessian_and_root_roundtrip(root,tmp_path,conditional):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model,check_model
    root=load_caching_nll();f=make_fast_cb(root);pdf=f['pdf'];axes=[f['x']]
    coordinates={'fcx':np.linspace(-8.,9.,9)}
    if conditional:
        y=root.RooRealVar('fcy','',.1,-1.,1.);axes.append(y)
        mean=root.CMSHggFormulaC1('fc_conditional_mean','',root.RooArgList(y,f['variables'][0]),root.std.vector('double')([.2,.1]))
        width=root.RooFormulaVar('fc_conditional_width','@0*(1+0.1*@1)',root.RooArgList(f['variables'][1],y))
        child=root.RooDoubleCBFast('conditional_fastcb','',f['x'],mean,width,*f['variables'][2:])
        marginal=root.RooUniform('fcy_pdf','',root.RooArgSet(y))
        prod=root.RooProdPdf('fc_conditional_product','',root.RooArgSet(marginal),root.RooFit.Conditional(root.RooArgSet(child),root.RooArgSet(f['x'])))
        pdf=root.SimpleProdPdf('fc_simple','',prod)
        coordinates['fcy']=np.repeat([-.7,0.,.8],3)
    workspace=root.RooWorkspace('w');getattr(workspace,'import')(pdf)
    path=tmp_path/'fast.root';workspace.writeToFile(str(path));handle=root.TFile.Open(str(path));w=handle.Get('w')
    model=bind_model(root,w.pdf(pdf.GetName()),[w.var(a.GetName()) for a in axes],coordinates)
    check_model(model)
    handle.Close()
