"""Tensor histogram/profile values and derivatives against native Combine."""
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import parameters,walk,UnsupportedModelError
from combine_jax.data import bind_data
from combine_jax.likelihood import Channel,Likelihood,from_workspace as old_import
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.root_io import open_workspace,load_caching_nll
from combine_jax.validation import finite_difference
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.templates import pack_templates,make_template_function,evaluate_templates,profile_solution,validate_templates
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy,unpack_model
from test_tensor_likelihood_terms import scalar_export
from test_histograms import make_histfunc
from test_bin_statistics import template_path


@pytest.mark.parametrize('log,active,rebin,freeze',[
    (False,False,None,False),(False,True,None,True),(True,True,None,False),
    (True,False,(0.,2.,7.),True),(False,True,(0.,1.25,4.,7.),False),
    (True,True,(0.,1.25,4.,7.),True)])
def test_histogram_native_nll_and_derivatives(root,log,active,rebin,freeze):
    root=load_caching_nll()
    x,a,b,hist=make_histfunc(root,log=log,active=active,rebin=rebin)
    if freeze:b.setVal(-.31);b.setConstant(True)
    scale=root.RooRealVar('scale','',1.,.1,4.)
    pdf=root.RooRealSumPdf('sum','',root.RooArgList(hist),root.RooArgList(scale),True)
    weight=root.RooRealVar('weight','',1.,0.,20.)
    variables=root.RooArgSet(x,weight)
    data=root.RooDataSet('data','',variables,root.RooFit.WeightVar(weight))
    for coord,w in [(.25,2.3),(1.25,4.7),(3.,0.),(5.5,0.)]:x.setVal(coord);data.add(variables,w)
    registry=default_registry();graph=registry.export([pdf],['hx']);raw=walk(pdf)
    info=parameters(pdf,['hx']);names=tuple(p.name for p in info)
    nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in graph.items()}
    binding=bind_data(data,['hx']);old=Likelihood(graph,registry,info,[Channel('hist','sum',('hx',),binding)],[])
    s=scalar_export(nodes,info,lambda n:float(raw[n].getVal()),observables=['hx'])
    rows=[dict(name='hist',pdf='sum',observables=('hx',),coordinates=dict(binding.coordinates),weights=binding.weights)]
    t=pack_templates(nodes,rows,s);sf=make_function(s);tf=make_template_function(t)
    arrays=jax.device_put(dict(s=dict(s.arrays),t=dict(t.arrays)))
    def fn(theta,arrays):return tf(sf(theta,arrays['s']),arrays['t'])[0][0]
    value=jax.jit(fn);grad=jax.jit(jax.grad(fn));hess=jax.jit(jax.hessian(fn))
    reference=root.cacheutils.CachingAddNLL('native','',pdf,data,True)
    reference.clearConstantZeroPoint();reference.clearZeroPoint()
    for i in range(4):
        point=np.array([np.random.default_rng(210+i).uniform(-1.,1.) if n!='scale' else 1.+.2*i for n in names])
        for n,v in zip(names,point,strict=True):raw[n].setVal(float(v))
        np.testing.assert_allclose(value(point,arrays),reference.getVal(),atol=1e-10,rtol=0.)
        np.testing.assert_allclose(grad(point,arrays),old.value_and_grad(point)[1],atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(hess(point,arrays),old.hessian(point),atol=1e-10,rtol=1e-10)
        np.testing.assert_allclose(evaluate_templates(t,np.array(sf(point,arrays['s'])))[0],[reference.getVal()],atol=1e-10)
    assert jax.make_jaxpr(fn)(point,arrays).consts==[]


@pytest.mark.parametrize('profile,global_value,frozen',[(True,0.,False),(False,0.,False),(True,.8,False),(True,.8,True)])
def test_complete_automcstats_native_nll_gradient_hessian(template_path,tmp_path,profile,global_value,frozen):
    root=load_caching_nll()
    with open_workspace(template_path) as workspace:
        workspace.var('prop_binch_bin0_In').setVal(global_value)
        if frozen:
            workspace.var('prop_binch_bin0').setVal(.4);workspace.var('prop_binch_bin0').setConstant(True)
        old=old_import(workspace,analytic_barlow_beeston=profile)
        model=from_workspace(workspace,Path(template_path).parent/'card.txt',analytic_barlow_beeston=profile)
        model.save(tmp_path/'model');model=TensorBundle.load(tmp_path/'model')
        assert model.metadata['parameter_names']==list(old.parameter_names)
        fn=make_model_function(model);arrays=jax.device_put(dict(model.arrays))
        value=jax.jit(fn);gradient=jax.jit(jax.grad(fn));hessian=jax.jit(jax.hessian(fn))
        ref=root.cacheutils.CachingSimNLL(workspace.pdf('model_s'),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        ref.setAnalyticBarlowBeeston(profile);ref.clearZeroPoint();ref.clearConstantZeroPoint()
        def native(point):
            for n,v in zip(old.parameter_names,point,strict=True):workspace.var(n).setVal(float(v))
            return float(ref.getVal())
        rng=np.random.default_rng(1819)
        for i in range(5):
            point=old.initial.copy()
            if i:
                for k,p in enumerate(old.parameters):
                    point[k]=rng.uniform(max(p.minimum,p.value-.8*p.error),min(p.maximum,p.value+.8*p.error))
            np.testing.assert_allclose(value(point,arrays),native(point),atol=1e-10,rtol=0.)
            np.testing.assert_allclose(evaluate_model_numpy(model,point),native(point),atol=1e-10,rtol=0.)
            np.testing.assert_allclose(gradient(point,arrays),old.value_and_grad(point)[1],atol=1e-10,rtol=1e-10)
            fd=finite_difference(native,point,np.full(len(point),2e-4))
            np.testing.assert_allclose(gradient(point,arrays),fd,atol=1e-8,rtol=1e-4)
            if i<2:np.testing.assert_allclose(hessian(point,arrays),old.hessian(point),atol=1e-10,rtol=1e-10)
        assert jax.make_jaxpr(fn)(point,arrays).consts==[]
        ref.setAnalyticBarlowBeeston(False)
    script='''import sys, numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import evaluate_model_numpy
m=TensorBundle.load(sys.argv[1])
assert np.isfinite(evaluate_model_numpy(m,m.arrays['scalars__parameter_initial']))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'model')],check=True,capture_output=True)


def test_profile_bounds_and_zero_error_are_invalid(template_path):
    with open_workspace(template_path) as workspace:
        workspace.var('prop_binch_bin0').setRange(-.001,.001)
        model=from_workspace(workspace,Path(template_path).parent/'card.txt')
    point=model.arrays['scalars__parameter_initial']
    assert np.isnan(evaluate_model_numpy(model,point))
    assert np.isnan(jax.jit(make_model_function(model))(point,jax.device_put(dict(model.arrays))))
    assert np.isnan(profile_solution(10.,0.,10.,.3,-7.,7.))
    for mean,error,obs,g in [(20.,2.,0.,0.),(1e6,.01,1e6+1,.2),(4.,2.,4.,0.)]:
        from combine_jax.bin_statistics import barlow_beeston_solution
        np.testing.assert_allclose(profile_solution(mean,error,obs,g,-1000.,1000.),barlow_beeston_solution(mean,error,obs,g),atol=1e-14)


def test_shared_padded_batches_have_flat_trace_and_fold_histograms(template_path):
    with open_workspace(template_path) as workspace:
        # Freeze the only morph: rates and bin corrections still float.
        workspace.var('shape').setConstant(True)
        old=old_import(workspace,analytic_barlow_beeston=False)
        model=from_workspace(workspace,Path(template_path).parent/'card.txt',analytic_barlow_beeston=False)
    components=unpack_model(model);s=components['scalars'];nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in old.nodes.items()}
    ch=old.channels[0]
    row=dict(name='ch',pdf=ch.pdf,observables=ch.observables,coordinates=dict(ch.data.coordinates),weights=ch.data.weights)
    sf=make_function(s);sv=np.asarray(sf(old.initial,jax.device_put(dict(s.arrays))))
    single=pack_templates(nodes,[row],s)
    assert len(single.metadata['folded_histograms'])==2
    assert len(single.metadata['schedule'])==1
    rows=[{**row,'name':f'ch{i}'} for i in range(9)]
    # Exercise padded rows, including an empty channel, without changing histogram geometry.
    rows[-1]={**rows[-1],'coordinates':{n:v[:0] for n,v in ch.data.coordinates.items()},'weights':ch.data.weights[:0]}
    many=pack_templates(nodes,rows,s)
    first=evaluate_templates(single,sv)[0][0];result=evaluate_templates(many,sv)[0]
    np.testing.assert_allclose(result[:8],first,atol=1e-12)
    assert len(many.metadata['histogram_names'])==2
    functions=[make_template_function(t) for t in (single,many)]
    traces=[jax.make_jaxpr(f)(sv,jax.device_put(dict(t.arrays))) for t,f in zip((single,many),functions,strict=True)]
    assert all(trace.consts==[] for trace in traces)
    assert len(traces[0].jaxpr.eqns)==len(traces[1].jaxpr.eqns)
    np.testing.assert_allclose(jax.jit(functions[1])(sv,jax.device_put(dict(many.arrays)))[0],result,atol=1e-12)
    bad=dict(many.arrays);bad['histogram_indices']=np.full_like(bad['histogram_indices'],100)
    with pytest.raises(ValueError,match='Invalid gather'):validate_templates(TensorBundle(many.metadata,bad))


def test_profile_declaration_and_composite_binding_corruption_rejected(template_path):
    with open_workspace(template_path) as workspace:
        old=old_import(workspace);model=from_workspace(workspace,Path(template_path).parent/'card.txt')
    parts=unpack_model(model);nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in old.nodes.items()}
    ch=old.channels[0];row=dict(name=ch.name,pdf=ch.pdf,observables=ch.observables,coordinates=dict(ch.data.coordinates),weights=ch.data.weights)
    declaration=dict(vars(old.profiles[0]));declaration['global_observed']=(1.,)
    with pytest.raises(UnsupportedModelError,match='global observation'):
        pack_templates(nodes,[row],parts['scalars'],profiles=[declaration])
    # Native profile data assignment uses the last row in a bin, not their sum.
    duplicate={**row,'coordinates':{n:np.r_[v,v[0]] for n,v in ch.data.coordinates.items()},'weights':np.r_[ch.data.weights,ch.data.weights[0]]}
    pack_templates(nodes,[duplicate],parts['scalars'],profiles=old.profiles)
    duplicate['weights'][-1]+=1.
    with pytest.raises(UnsupportedModelError,match='differs from data'):
        pack_templates(nodes,[duplicate],parts['scalars'],profiles=old.profiles)
    from combine_jax.tensor.model import compose_templates
    metadata={**parts['templates'].metadata,'profile_names':['wrong']}
    parts['templates']=TensorBundle(metadata,parts['templates'].arrays)
    with pytest.raises(ValueError,match='profile bindings differ'):compose_templates(**parts)
