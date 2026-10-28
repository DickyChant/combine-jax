"""Batched analytic channels against native v11 and the unchanged AD oracle."""
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.binned_channels import pack_binned_channels,make_binned_function,validate_binned_channels
from test_resonance import make_doublecb,make_erfexp
from test_tensor_binned_channels import bind_model,check_model


def shape_fixture(root,family):
    x=root.RooRealVar('ax','',.2,-4.,5.)
    original_x=x  # Retain ownership when the mean/coefficient is the observable.
    mean=root.RooRealVar('acenter','',.3,-.5,.8)
    sigma=root.RooRealVar('asigma','',1.1,.6,1.5)
    slope=root.RooRealVar('aslope','',0.,-.4,.3)
    if family=='gaussian':shape=root.RooGaussian('analytic','',x,mean,sigma)
    elif family=='gaussian_mean':
        mean.setRange(-4.,5.);x.setRange(-.5,.8)
        shape=root.RooGaussian('analytic','',x,mean,sigma);x=mean
    elif family=='exponential':shape=root.RooExponential('analytic','',x,slope)
    elif family=='exponential_coefficient':
        x.setRange(.5,1.5);x.setVal(1.)
        shape=root.RooExponential('analytic','',x,slope);x=slope
    elif family=='power':
        x.setRange(1.,5.);x.setVal(2.);slope.setRange(-2.5,.5);slope.setVal(-1.)
        shape=root.RooPower('analytic','',x,slope)
    elif family.startswith('doublecb'):
        fixture=make_doublecb(root,shifted=family=='doublecb_xp');x=fixture['x'];shape=fixture['pdf']
        if family=='doublecb_xp':x.setConstant(True);x=fixture['xp']
        if family=='doublecb_signed':
            alpha,power=fixture['variables'][2:4];alpha.setRange(-.99,-.8);alpha.setVal(-.9)
            power.setRange(4.,6.);power.setVal(5.)
    elif family=='erfexp':fixture=make_erfexp(root);x=fixture['x'];shape=fixture['pdf']
    else:raise ValueError(family)
    coordinates={str(x.GetName()):np.linspace(x.getMin(),x.getMax(),9)[1:-1]}
    return locals()


@pytest.mark.parametrize('family',['gaussian','gaussian_mean','exponential','exponential_coefficient',
    'power','doublecb','doublecb_xp','doublecb_signed','erfexp'])
def test_analytic_nll_native_gradient_hessian(root,family):
    root=load_caching_nll();fixture=shape_fixture(root,family)
    model=bind_model(root,fixture['shape'],[fixture['x']],fixture['coordinates']);check_model(model)


@pytest.mark.parametrize('power',[1.,1.-3e-6,1.+3e-6,1.-3e-5,1.+3e-5])
def test_doublecb_native_log_tail_branch(root,power):
    root=load_caching_nll();fixture=make_doublecb(root)
    for v in (fixture['variables'][3],fixture['variables'][5]):
        v.setRange(.2,5.);v.setVal(power);v.setConstant(True)
    model=bind_model(root,fixture['pdf'],[fixture['x']],{'rx':np.array([-5.5,-3.,0.,4.5,6.5])})
    check_model(model)


def test_analytic_shared_batches_fixed_arrays_and_neutral_transport(root,tmp_path):
    root=load_caching_nll();fixture=shape_fixture(root,'doublecb');model=bind_model(root,fixture['shape'],[fixture['x']],fixture['coordinates'])
    copies=[{**model['rows'][0],'name':f'copy{i}'} for i in range(4)]
    one=model['binned'];many=pack_binned_channels(model['nodes'],copies,model['scalar'])
    values=np.asarray(model['sf'](model['old'].initial,model['arrays']['s']))
    traces=[jax.make_jaxpr(make_binned_function(b))(values,jax.device_put(dict(b.arrays))) for b in (one,many)]
    assert len(traces[0].jaxpr.eqns)==len(traces[1].jaxpr.eqns) and all(t.consts==[] for t in traces)
    np.testing.assert_allclose(jax.jit(make_binned_function(many))(values,jax.device_put(dict(many.arrays))),model['native'](model['old'].initial),atol=1e-10)
    for p in model['info']:model['raw'][p.name].setConstant(True)
    fixed=bind_model(root,fixture['shape'],[fixture['x']],fixture['coordinates'])
    assert not any(b['operation'].startswith(('analytic_','gridanalytic_')) for b in fixed['binned'].metadata['program'])
    model['scalar'].save(tmp_path/'scalars');many.save(tmp_path/'channels')
    script='''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalars');b=TensorBundle.load(p/'channels')
assert np.all(np.isfinite(evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)
    bad=dict(many.arrays);batch=next(b for b in many.metadata['program'] if b['operation'].startswith(('analytic_','gridanalytic_')))
    key=batch['buffers']['sign'];bad[key]=np.zeros_like(bad[key])
    with pytest.raises(ValueError,match='Invalid analytic integration geometry'):validate_binned_channels(TensorBundle(many.metadata,bad))


def test_fixed_conditional_width_folds_and_matches_native(root):
    root=load_caching_nll();fixture=shape_fixture(root,'gaussian');x=fixture['x']
    y=root.RooRealVar('conditional_y','',0.,-1.,1.)
    width=root.RooFormulaVar('conditional_width','2+@0',root.RooArgList(y))
    shape=root.RooGaussian('conditional_analytic','',x,fixture['mean'],width)
    uniform=root.RooUniform('marginal_y','',root.RooArgSet(y))
    pdf=root.RooProdPdf('conditional_product','',root.RooArgSet(uniform),root.RooFit.Conditional(root.RooArgSet(shape),root.RooArgSet(x)))
    coordinates={**fixture['coordinates'],'conditional_y':np.linspace(-.8,.8,7)}
    model=bind_model(root,pdf,[x,y],coordinates);check_model(model)
    assert any(not q['folded'] for q in model['binned'].metadata['analytic_grid_queries'])  # Mean still floats.
    assert not any(q['operation'].startswith('gridexpr_') for q in model['binned'].metadata['program'])


def test_negated_exponential_cache_mismatch_is_explicit(root):
    root=load_caching_nll();fixture=shape_fixture(root,'exponential')
    shape=root.RooExponential('negated_exponential','',fixture['x'],fixture['slope'],True)
    with pytest.raises(UnsupportedModelError,match='negated_exponential.*differs between RooFit and v11 optimized CachingNLL'):
        bind_model(root,shape,[fixture['x']],fixture['coordinates'])


@pytest.mark.parametrize('degree,restricted',[(0,False),(1,False),(2,False),(4,False),(7,False),(4,True)])
def test_bernstein_precomputed_basis_and_integrals(root,degree,restricted):
    root=load_caching_nll();x=root.RooRealVar('bx','',120.,100.,180.)
    coefficients=[root.RooRealVar(f'bc{i}','',1.+.1*i,.2,2.5) for i in range(degree+1)]
    shape=root.RooBernstein('bernstein','',x,root.RooArgList(*coefficients))
    if restricted:
        x.setRange('reference',100.,180.);shape.selectNormalizationRange('reference',True);x.setRange(110.,170.)
    model=bind_model(root,shape,[x],{'bx':np.linspace(x.getMin(),x.getMax(),9)[1:-1]});check_model(model)
