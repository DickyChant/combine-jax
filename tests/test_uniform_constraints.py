import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.likelihood import from_workspace as oracle_import
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy
from test_tensor_likelihood_terms import constraint_fixture


@pytest.mark.parametrize('axes',[(),('u_x',),('u_x','u_y')])
def test_uniform_constraint_normalization_and_zero_derivatives(root,axes):
    x=root.RooRealVar('u_x','',.2,-2.,3.);y=root.RooRealVar('u_y','',.4,-1.,5.)
    pdf=root.RooUniform('uniform_constraint','',root.RooArgSet(x,y))
    scalars,constraints,function,oracle=constraint_fixture(pdf,dict(u_x=x,u_y=y),('u_x','u_y'),axes)
    norm=root.RooArgSet(*(dict(u_x=x,u_y=y)[n] for n in axes))
    for theta in ([.2,.4],[-1.2,3.5],[2.7,4.9]):
        point=np.array(theta);x.setVal(theta[0]);y.setVal(theta[1])
        np.testing.assert_allclose(jax.jit(function)(point),-pdf.getLogVal(norm),atol=1e-14,rtol=1e-14)
        np.testing.assert_allclose(function(point),oracle(point),atol=1e-14,rtol=1e-14)
        np.testing.assert_array_equal(jax.grad(function)(point),np.zeros(2))
        np.testing.assert_array_equal(jax.hessian(function)(point),np.zeros((2,2)))
    assert constraints.metadata['folded_constraints']==['uniform_constraint']
    assert constraints.metadata['schedule'][0]['inputs'] is None


def test_uniform_constraints_complete_asimov_workspace(root):
    root=load_caching_nll()
    x=root.RooRealVar('count_x','',.5,0.,1.);r=root.RooRealVar('r','',1.,.2,3.)
    nuisance=root.RooRealVar('nuisance','',1.,.5,2.)
    global_obs=root.RooRealVar('global_obs','',.7,0.,1.);global_obs.setConstant(True)
    rate=root.RooFormulaVar('rate','12*@0*@1',root.RooArgList(r,nuisance))
    shape=root.RooUniform('shape','',root.RooArgSet(x))
    channel=root.RooAddPdf('channel','',root.RooArgList(shape),root.RooArgList(rate))
    constraint=root.RooUniform('constraint','',root.RooArgSet(nuisance,global_obs))
    constrained=root.RooProdPdf('constrained','',root.RooArgList(channel,constraint))
    category=root.RooCategory('channel_category','');category.defineType('bin0')
    pdf=root.RooSimultaneous('model','',category);pdf.addPdf(constrained,'bin0')
    weight=root.RooRealVar('weight','',12.,0.,100.)
    data=root.RooDataSet('asimov','',root.RooArgSet(x,category,weight),root.RooFit.WeightVar(weight))
    data.add(root.RooArgSet(x,category,weight),12.)
    workspace=root.RooWorkspace('w');getattr(workspace,'import')(pdf);getattr(workspace,'import')(data)
    config=root.RooStats.ModelConfig('ModelConfig',workspace);config.SetPdf(workspace.pdf('model'))
    config.SetObservables(root.RooArgSet(workspace.var('count_x'),workspace.cat('channel_category')))
    config.SetParametersOfInterest(root.RooArgSet(workspace.var('r')))
    config.SetNuisanceParameters(root.RooArgSet(workspace.var('nuisance')))
    config.SetGlobalObservables(root.RooArgSet(workspace.var('global_obs')))
    getattr(workspace,'import')(config)
    oracle=oracle_import(workspace,data='asimov');model=from_workspace(workspace,data='asimov')
    reference=root.cacheutils.CachingSimNLL(workspace.pdf('model'),workspace.data('asimov'),config.GetNuisanceParameters())
    reference.clearZeroPoint();reference.clearConstantZeroPoint()
    function=make_model_function(model);arrays=jax.device_put(dict(model.arrays));value=jax.jit(function)
    rng=np.random.default_rng(612)
    points=[oracle.initial]+list(rng.uniform([.6,.3],[1.9,2.8],(25,2)))
    for point in points:
        for name,v in zip(oracle.parameter_names,point,strict=True):workspace.var(name).setVal(float(v))
        native=reference.getVal()
        np.testing.assert_allclose(value(point,arrays),native,atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(evaluate_model_numpy(model,point),native,atol=1e-10,rtol=1e-12)
        np.testing.assert_allclose(jax.grad(function)(point,arrays),oracle.value_and_grad(point)[1],atol=1e-11,rtol=1e-11)
