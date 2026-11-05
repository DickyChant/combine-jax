"""Independent per-channel copies of a data coordinate must bind exactly."""
from array import array

import jax
import numpy as np
import pytest

from combine_jax.graph import parameters, walk, walk_roots
from combine_jax.likelihood import from_workspace as oracle_import
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.model import from_workspace, make_model_function, evaluate_model_numpy
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def pair(root):
    x = [root.RooRealVar('alias_x', '', v, 0., 4.) for v in (1., 3.)]
    for item in x:item.setBins(4)
    slope = root.RooRealVar('alias_slope', '', -.3, -2., -.1)
    pdf = [root.RooExponential('alias_pdf'+str(i), '', v, slope) for i,v in enumerate(x)]
    return x, slope, pdf


def test_only_declared_observables_can_alias(root):
    x, slope, pdf = pair(root)
    with pytest.raises(ValueError, match='Distinct graph nodes.*alias_x'):
        walk_roots(pdf)
    nodes = default_registry().export(pdf, ['alias_x'])
    coords = np.array([.1, .7, 1.4, 3.8])
    context = Evaluation(nodes, default_registry(), ['alias_slope'], [-.3], {'alias_x':coords})
    for i,p in enumerate(pdf):
        expected=[]
        for v in coords:x[i].setVal(float(v));expected.append(p.getVal(root.RooArgSet(x[i])))
        np.testing.assert_allclose(context.density(p.GetName(), ['alias_x']),expected,atol=1e-14,rtol=1e-14)
    # The same-name guard also operates between roots, rather than only within each.
    assert len(nodes) == 4


@pytest.mark.parametrize('difference', ['range','bins','named_range','named_bins','constant','class'])
def test_observable_geometry_conflicts_still_rejected(root,difference):
    x, slope, pdf = pair(root)
    if difference=='range':x[1].setRange(0., 5.)
    elif difference=='bins':x[1].setBinning(root.RooBinning(3,array('d',[0.,1.,3.,4.])))
    elif difference=='named_range':x[1].setRange('restricted',.5,3.5)
    elif difference=='named_bins':x[1].setBins(2,'alternate')
    elif difference=='constant':x[1].setConstant(True)
    else:
        x[1]=root.RooConstVar('alias_x','',1.)
        pdf[1]=root.RooExponential('alias_pdf1','',x[1],slope)
    with pytest.raises(ValueError,match='Distinct graph nodes.*alias_x'):
        default_registry().export(pdf,['alias_x'])


def test_duplicate_parameter_and_function_identities_still_rejected(root):
    x, slope, pdf = pair(root)
    slope2=root.RooRealVar('alias_slope','',-.3,-2.,-.1)
    pdf[1]=root.RooExponential('alias_pdf1','',x[1],slope2)
    with pytest.raises(ValueError,match='Distinct graph nodes.*alias_slope'):
        default_registry().export(pdf,['alias_x'])
    pdf[1]=root.RooExponential('alias_pdf0','',x[1],slope)
    with pytest.raises(ValueError,match='Distinct graph nodes.*alias_pdf0'):
        default_registry().export(pdf,['alias_x'])


def test_parameterized_observable_alias_ranges_rejected(root):
    x, slope, pdf = pair(root)
    low=root.RooRealVar('alias_low','',.2,0.,1.)
    high=root.RooRealVar('alias_high','',3.8,3.,4.)
    for v in x:v.setRange('moving',low,high)
    with pytest.raises(ValueError,match='Distinct graph nodes.*alias_x'):
        default_registry().export(pdf,['alias_x'])


def histogram_workspace(root):
    """Two CMSHistSum channels with owned observable copies, as in HIG."""
    w=root.RooWorkspace('w');x=root.RooRealVar('alias_x','',.5,0.,3.);x.setBins(3)
    rate=root.RooRealVar('alias_rate','',1.,.3,2.)
    channel=root.RooCategory('alias_channel','');channel.defineType('a');channel.defineType('b')
    sim=root.RooSimultaneous('model_s','',channel);held=[x,rate,channel,sim]
    templates=([4.,9.,2.],[7.,3.,5.])
    for label,contents in zip(('a','b'),templates,strict=True):
        h=root.TH1D('hist_'+label,'',3,0.,3.);h.SetDirectory(0)
        for b,v in enumerate(contents,1):h.SetBinContent(b,v)
        f=root.CMSHistFunc('func_'+label,'',x,h,True);f.prepareStorage()
        total=root.CMSHistSum('total_'+label,'',x,root.RooArgList(f),root.RooArgList(rate))
        one=root.RooConstVar('one_'+label,'',1.)
        p=root.RooRealSumPdf('pdf_'+label,'',root.RooArgList(total),root.RooArgList(one),True)
        sim.addPdf(p,label);held.extend([h,f,total,one,p])
    getattr(w,'import')(sim)
    weight=root.RooRealVar('weight','',1.,0.,100.)
    columns=root.RooArgSet(w.var('alias_x'),w.cat('alias_channel'),weight)
    data=root.RooDataSet('asimov','',columns,root.RooFit.WeightVar(weight))
    for label,contents in zip(('a','b'),templates,strict=True):
        w.cat('alias_channel').setLabel(label)
        for b,v in enumerate(contents):w.var('alias_x').setVal(b+.5);data.add(columns,v)
    getattr(w,'import')(data)
    mc=root.RooStats.ModelConfig('ModelConfig',w);mc.SetPdf(w.pdf('model_s'))
    mc.SetObservables(root.RooArgSet(w.var('alias_x'),w.cat('alias_channel')))
    mc.SetParametersOfInterest(root.RooArgSet(w.var('alias_rate')))
    mc.SetNuisanceParameters(root.RooArgSet());mc.SetGlobalObservables(root.RooArgSet())
    getattr(w,'import')(mc)
    # RooWorkspace normally recycles by name on import. Install the independent
    # owned coordinates afterwards, mimicking the serialized HIG graph.
    for label in ('a','b'):
        clone=root.RooRealVar('alias_x','',.5,0.,3.);clone.setBins(3)
        total=w.function('total_'+label)
        assert not total.redirectServers(root.RooArgSet(clone))
        assert root.addressof(total.getXVar())==root.addressof(clone)
        held.append(clone)
    return w,held


def test_cloned_histogram_observables_full_tensor_native_gates(root):
    root=load_caching_nll();w,held=histogram_workspace(root)
    with pytest.raises(ValueError,match='Distinct graph nodes.*alias_x'):walk(w.pdf('model_s'))
    info=parameters(w.pdf('model_s'),['alias_x','alias_channel'],pois=['alias_rate'])
    assert [p.name for p in info]==['alias_rate']
    oracle=oracle_import(w,data='asimov');model=from_workspace(w,data='asimov')
    arrays=jax.device_put(dict(model.arrays));fn=make_model_function(model)
    value_and_grad=jax.jit(jax.value_and_grad(fn))
    native=root.cacheutils.CachingSimNLL(w.pdf('model_s'),w.data('asimov'),w.obj('ModelConfig').GetNuisanceParameters())
    native.clearZeroPoint();native.clearConstantZeroPoint()
    def reference(theta):
        w.var('alias_rate').setVal(float(theta[0]));return float(native.getVal())
    rng=np.random.default_rng(2810)
    for point in [np.ones(1), *rng.uniform(.4,1.8,(25,1))]:
        val,grad=value_and_grad(point,arrays)
        np.testing.assert_allclose(val,reference(point),atol=1e-6,rtol=0.)
        np.testing.assert_allclose(evaluate_model_numpy(model,point),reference(point),atol=1e-10,rtol=0.)
        np.testing.assert_allclose(oracle.value(point),reference(point),atol=1e-10,rtol=0.)
        fd=finite_difference(reference,point,np.full(1,1e-4))
        np.testing.assert_allclose(grad,fd,atol=1e-8,rtol=1e-4)
    del native
