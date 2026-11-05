"""Fixed MultiPdf choices may share parameters; profiling stays uncertified."""
import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy
from combine_jax.validation import finite_difference


def fixture(root,indices):
    w=root.RooWorkspace('w')
    for expression in ('x[0.5,0,1]','kl[1,-15,26]',
                       'Exponential::shared(x,shared_slope[-0.3,-1,0.2])',
                       'Exponential::private_a(x,slope_a[-0.6,-1,0.2])',
                       'Exponential::private_b(x,slope_b[0.1,-1,0.2])',
                       'expr::rate_a("12*exp(0.02*(@0-1))",kl)',
                       'expr::rate_b("15*exp(0.02*(@0-1))",kl)'):
        w.factory(expression)
    held=[]
    for label,index in zip(('a','b'),indices,strict=True):
        category=root.RooCategory('index_'+label,'')
        pdf=root.RooMultiPdf('multi_'+label,'',category,root.RooArgList(w.pdf('shared'),w.pdf('private_'+label)))
        held.extend((category,pdf))
        getattr(w,'import')(pdf,root.RooFit.RecycleConflictNodes())
        w.cat('index_'+label).setIndex(index)
        w.factory(f'SUM::channel_{label}(rate_{label}*multi_{label})')
    w.factory('SIMUL::model_s(CMS_channel[a=0,b=1],a=channel_a,b=channel_b)')
    config=root.RooStats.ModelConfig('ModelConfig',w)
    config.SetPdf(w.pdf('model_s'));config.SetObservables(root.RooArgSet(w.var('x'),w.cat('CMS_channel')))
    config.SetParametersOfInterest(root.RooArgSet(w.var('kl')))
    config.SetNuisanceParameters(root.RooArgSet(*(w.var(n) for n in ('shared_slope','slope_a','slope_b'))))
    config.SetGlobalObservables(root.RooArgSet());getattr(w,'import')(config)
    weight=root.RooRealVar('weight','',1.,0.,100.)
    columns=root.RooArgSet(w.var('x'),w.cat('CMS_channel'),weight)
    data=root.RooDataSet('toy_asimov','',columns,root.RooFit.WeightVar(weight))
    coordinates=np.linspace(.05,.95,10);weights=np.exp(-.3*coordinates);weights/=weights.sum()
    for label,rate in (('a',12.),('b',15.)):
        w.cat('CMS_channel').setLabel(label)
        for x,v in zip(coordinates,rate*weights,strict=True):
            w.var('x').setVal(float(x));data.add(columns,float(v))
    getattr(w,'import')(data);assert not w.data('data_obs')
    return w,held


@pytest.mark.parametrize('indices',[(0,0),(0,1),(1,0),(1,1)])
def test_fixed_shared_categories_native_nll_and_gradient(indices):
    root=load_caching_nll();w,held=fixture(root,indices)
    model=from_workspace(w,data='toy_asimov',discrete_policy='fixed')
    names=model.metadata['parameter_names']
    expected={'kl'}|({'shared_slope'} if 0 in indices else set())
    expected|={f'slope_{label}' for label,i in zip(('a','b'),indices,strict=True) if i==1}
    assert set(names)==expected and 'discrete_profile' not in model.metadata
    choices=model.metadata['provenance']['discrete_choices']
    assert {a['category']:a['index'] for a in choices.values()}==dict(zip(('index_a','index_b'),indices,strict=True))
    assert not any(a.get('profiled',False) for a in choices.values())
    nll=root.cacheutils.CachingSimNLL(w.pdf('model_s'),w.data('toy_asimov'),w.obj('ModelConfig').GetNuisanceParameters())
    nll.clearZeroPoint();nll.clearConstantZeroPoint()
    def native(theta):
        for name,value in zip(names,theta,strict=True):w.var(name).setVal(float(value))
        return float(nll.getVal())
    kernel=jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
    arrays=jax.device_put(dict(model.arrays));initial=model.arrays['parameter_info__initial']
    scale=np.array([1. if n=='kl' else .05 for n in names])
    points=[initial]+list(initial+np.random.default_rng(8306).uniform(-1.,1.,(25,len(names)))*scale)
    for i,theta in enumerate(points):
        value,gradient=kernel(theta,arrays)
        np.testing.assert_allclose(value,native(theta),atol=1e-9,rtol=0.)
        np.testing.assert_allclose(evaluate_model_numpy(model,theta),value,atol=1e-10,rtol=0.)
        if i<5:
            np.testing.assert_allclose(gradient,finite_difference(native,theta,np.full(len(theta),2e-4)),atol=2e-8,rtol=1e-4)
    with pytest.raises(UnsupportedModelError,match='shared between categories'):
        from_workspace(w,data='toy_asimov',discrete_policy='profile')
    assert tuple(w.cat('index_'+label).getIndex() for label in ('a','b'))==indices
