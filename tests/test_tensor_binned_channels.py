"""Batched composite binned channels against CachingAddNLL and old AD."""
from array import array
from itertools import product
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters,walk,UnsupportedModelError
from combine_jax.likelihood import Channel,Likelihood
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.binned_channels import pack_binned_channels,make_binned_function,evaluate_binned_channels,validate_binned_channels
from test_tensor_likelihood_terms import scalar_export
from test_remaining_shapes import make_shape
from test_vertical_nd import make_multidimensional


def bind_model(root,shape,axes,coordinates,weights=None):
    rate=root.RooRealVar('test_rate','',12.,2.,30.)
    pdf=root.RooAddPdf('test_sum','',root.RooArgList(shape),root.RooArgList(rate))
    weight=root.RooRealVar('test_weight','',1.,0.,100.)
    columns=root.RooArgSet(*axes,weight);data=root.RooDataSet('test_data','',columns,root.RooFit.WeightVar(weight))
    names=tuple(str(v.GetName()) for v in axes);R=len(next(iter(coordinates.values())))
    for i in range(R):
        for v,n in zip(axes,names,strict=True):v.setVal(float(coordinates[n][i]))
        data.add(columns,float(weights[i]) if weights is not None else 0. if i==0 else .2+.3*i)
    registry=default_registry();graph=registry.export([pdf],names);info=parameters(pdf,names);raw=walk(pdf)
    nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in graph.items()};binding=bind_data(data,names)
    old=Likelihood(graph,registry,info,[Channel('test','test_sum',names,binding)],[])
    scalar=scalar_export(nodes,info,lambda n:float(raw[n].getVal()),observables=names)
    rows=[dict(name='test',pdf='test_sum',observables=names,coordinates=dict(binding.coordinates),weights=binding.weights)]
    binned=pack_binned_channels(nodes,rows,scalar)
    sf=make_function(scalar);bf=make_binned_function(binned);arrays=jax.device_put(dict(s=dict(scalar.arrays),b=dict(binned.arrays)))
    def function(theta,arrays):return bf(sf(theta,arrays['s']),arrays['b'])[0]
    reference=root.cacheutils.CachingAddNLL('test_reference','',pdf,data,True)
    reference.clearZeroPoint();reference.clearConstantZeroPoint()
    def native(theta):
        for p,v in zip(info,theta,strict=True):raw[p.name].setVal(float(v))
        return float(reference.getVal())
    return locals()


def check_model(model):
    f=model['function'];arrays=model['arrays'];old=model['old'];native=model['native']
    value=jax.jit(f);gradient=jax.jit(jax.grad(f));hessian=jax.jit(jax.hessian(f))
    rng=np.random.default_rng(765)
    points=[old.initial]+[np.array([rng.uniform(p.minimum+.1*(p.maximum-p.minimum),p.maximum-.1*(p.maximum-p.minimum)) for p in old.parameters]) for _ in range(4)]
    for i,theta in enumerate(points):
        np.testing.assert_allclose(value(theta,arrays),native(theta),atol=1e-9,rtol=0.)
        np.testing.assert_allclose(gradient(theta,arrays),old.value_and_grad(theta)[1],atol=1e-9,rtol=1e-10)
        scalars=np.asarray(model['sf'](theta,arrays['s']))
        np.testing.assert_allclose(evaluate_binned_channels(model['binned'],scalars),[native(theta)],atol=1e-9,rtol=0.)
        if i<2:
            np.testing.assert_allclose(gradient(theta,arrays),finite_difference(native,theta,np.full(len(theta),2e-4)),atol=2e-8,rtol=1e-4)
            np.testing.assert_allclose(hessian(theta,arrays),old.hessian(theta),atol=1e-9,rtol=1e-10)
    assert jax.make_jaxpr(f)(old.initial,arrays).consts==[]


@pytest.mark.parametrize('family',['step','hist2d','step_floor'])
def test_parametric_bin_channel_native_nll_ad_and_serialization(root,family,tmp_path):
    root=load_caching_nll();fixture=make_shape(root,'step' if family=='step_floor' else family)
    if family=='step_floor':
        for v in fixture['coefficients']:v.setRange(0.,2.);v.setVal(1.)
    model=bind_model(root,fixture['pdf'],fixture['axes'],fixture['coordinates']);check_model(model)
    model['scalar'].save(tmp_path/'scalar');model['binned'].save(tmp_path/'binned')
    script='''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalar');b=TensorBundle.load(p/'binned')
assert np.all(np.isfinite(evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)


@pytest.mark.parametrize('kind',['nset','cset','nested'])
def test_conditional_histogram_product_and_nested_fraction_mixture(root,kind):
    root=load_caching_nll();variables,a,b,conditional,edges,held=make_multidimensional(root,'FastVerticalInterpHistPdf2D2',True,False)
    x,y=variables;templates=root.TList();hists=[]
    for i,values in enumerate(([3.,5.,7.],[2.,6.,8.],[4.,4.,6.])):
        hist=root.TH1D(f'marginal_hist{i}','',3,array('d',edges[0]));hist.SetDirectory(0)
        for j,v in enumerate(values,1):hist.SetBinContent(j,v)
        templates.Add(hist);hists.append(hist)
    marginal=root.FastVerticalInterpHistPdf2('marginal','',x,templates,root.RooArgList(a),.7,1)
    uniform=root.RooUniform('uniform','',root.RooArgSet(x));fraction=root.RooRealVar('fraction','',.3,.1,.9)
    mixture=root.RooAddPdf('mixture','',root.RooArgList(marginal,uniform),root.RooArgList(fraction))
    parent=mixture if kind=='nested' else marginal
    selected=root.RooArgSet(x if kind=='cset' else y)
    shape=root.RooProdPdf('conditional_product','',root.RooArgSet(parent),root.RooFit.Conditional(root.RooArgSet(conditional),selected,kind=='cset'))
    coords=np.array(list(product((edges[0][1:-1]+edges[0][2:])/2,(edges[1][:-1]+edges[1][1:])/2)))
    model=bind_model(root,shape,variables,dict(zip(('ndx','ndy'),coords.T,strict=True)));check_model(model)


@pytest.mark.parametrize('algorithm,region',[(0,0.),(0,.7),(1,.7),(2,.7)])
def test_vertical_raw_parametric_integrals_and_floors(root,algorithm,region):
    root=load_caching_nll();fixture=make_shape(root,'hist2d');central=fixture['pdf'];x,y=fixture['axes']
    # With zero smoothing the origin is a kink; symmetric FD has no unique
    # derivative there. Check derivatives at a regular point instead.
    theta=root.RooRealVar('morph','',.2 if region==0 else 0.,-1.5,1.5);held=[];functions=root.RooArgList(central)
    for label,factor in [('up',1.3),('down',.8)]:
        coefs=[]
        for i,coefficient in enumerate(fixture['coefficients']):
            f=root.RooFormulaVar(f'{label}_{i}',f'{factor+i*.03}*@0',root.RooArgList(coefficient));held.append(f);coefs.append(f)
        pdf=root.RooParametricHist2D(label,'',x,y,root.RooArgList(*coefs),fixture['histogram']);held.append(pdf);functions.add(pdf)
    shape=root.VerticalInterpPdf('vertical','',functions,root.RooArgList(theta),region,algorithm)
    model=bind_model(root,shape,[x,y],fixture['coordinates']);check_model(model)


def test_shared_padding_empty_channels_masks_and_fixed_folding(root):
    root=load_caching_nll();fixture=make_shape(root,'step')
    model=bind_model(root,fixture['pdf'],fixture['axes'],fixture['coordinates'])
    rows=model['rows'];nodes=model['nodes'];s=model['scalar'];row=rows[0]
    copies=[{**row,'name':f'copy{i}'} for i in range(5)]
    copies.append({**row,'name':'empty','weights':np.empty(0),'coordinates':{n:np.empty(0) for n in row['observables']}})
    copies.append(dict(name='masked',pdf='missing',masked=True))
    batch=pack_binned_channels(nodes,copies,s)
    scalar_values=np.asarray(model['sf'](model['old'].initial,model['arrays']['s']))
    expected=model['native'](model['old'].initial)
    values=evaluate_binned_channels(batch,scalar_values)
    np.testing.assert_allclose(values[:5],expected,atol=1e-10);assert values[-1]==0.
    np.testing.assert_allclose(values[-2],12.,atol=1e-13)
    f=make_binned_function(batch);np.testing.assert_allclose(jax.jit(f)(scalar_values,jax.device_put(dict(batch.arrays))),values,atol=1e-10)
    # Same data shape replicated: operator count is independent of channel count.
    many=pack_binned_channels(nodes,copies[:5],s)
    traces=[jax.make_jaxpr(make_binned_function(b))(scalar_values,jax.device_put(dict(b.arrays))) for b in (model['binned'],many)]
    assert len(traces[0].jaxpr.eqns)==len(traces[1].jaxpr.eqns) and all(t.consts==[] for t in traces)
    for p in model['info']:model['raw'][p.name].setConstant(True)
    fixed_model=bind_model(root,fixture['pdf'],fixture['axes'],fixture['coordinates'])
    # The newly created outer rate floats; all shape queries fold.
    assert [b['operation'] for b in fixed_model['binned'].metadata['program']]==['mixture_yields']
    bad=dict(batch.arrays);key=batch.metadata['nll_program'][-1]['buffers']['densities'];bad[key]=np.full_like(bad[key],10**9)
    with pytest.raises(ValueError,match='Invalid binned index'):validate_binned_channels(TensorBundle(batch.metadata,bad))


def test_overlapping_or_unsupported_product_fails_by_name(root):
    root=load_caching_nll();fixture=make_shape(root,'step');shape=fixture['pdf'];x=fixture['x']
    uniform=root.RooUniform('uniform','',root.RooArgSet(x));product_pdf=root.RooProdPdf('bad_product','',root.RooArgList(shape,uniform))
    with pytest.raises(UnsupportedModelError,match='bad_product.*overlapping'):
        bind_model(root,product_pdf,[x],fixture['coordinates'])
    mean=root.RooRealVar('poisson_mean','',2.,1.,3.)
    poisson=root.RooPoisson('unsupported_poisson','',x,mean)
    with pytest.raises(UnsupportedModelError,match='unsupported_poisson.*unsupported binned density class'):
        bind_model(root,poisson,[x],fixture['coordinates'])


@pytest.mark.parametrize('target',['fast','program','nll','cross'])
def test_scatter_unique_destination_guarantees_are_validated(root,target):
    root=load_caching_nll();axes,a,b,shape,edges,held=make_multidimensional(root,'FastVerticalInterpHistPdf2D2',False,False)
    points=np.array(list(product(*[(e[:-1]+e[1:])/2 for e in edges])))
    model=bind_model(root,shape,axes,dict(zip(('ndx','ndy'),points.T,strict=True)))
    bundle=pack_binned_channels(model['nodes'],[{**model['rows'][0],'name':f'copy{i}'} for i in range(2)],model['scalar'])
    key=('fast_outputs' if target=='fast' else bundle.metadata['program' if target in ('program','cross') else 'nll_program'][0]['buffers']['outputs'])
    bad=dict(bundle.arrays);bad[key]=bad[key].copy();flat=bad[key].reshape(-1);assert len(flat)>1
    if target=='cross':flat[0]=bundle.arrays['fast_outputs'][0]
    else:flat[1]=flat[0]
    with pytest.raises(ValueError,match='Repeated binned outputs'):
        make_binned_function(TensorBundle(bundle.metadata,bad))


def test_complete_workspace_card_import_and_roundtrip(tmp_path):
    import importlib.util
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy
    from combine_jax.likelihood import from_workspace as old_import
    source=Path(__file__).resolve().parents[1]/'scripts/build_binned_fixture.py'
    spec=importlib.util.spec_from_file_location('build_binned_fixture',source)
    builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
    path=builder.build_binned_fixture(tmp_path/'fixture')
    with open_workspace(path) as workspace:
        old=old_import(workspace);model=from_workspace(workspace,path.parent/'card.txt')
    model.save(tmp_path/'model');model=TensorBundle.load(tmp_path/'model')
    assert model.metadata['provenance']['all_card_channels_imported'] and model.metadata['kind']=='binned_likelihood'
    f=make_model_function(model);arrays=jax.device_put(dict(model.arrays))
    np.testing.assert_allclose(jax.jit(f)(old.initial,arrays),old.value(old.initial),atol=1e-11)
    np.testing.assert_allclose(evaluate_model_numpy(model,old.initial),old.value(old.initial),atol=1e-11)
    np.testing.assert_allclose(jax.jit(jax.hessian(f))(old.initial,arrays),old.hessian(old.initial),atol=1e-10)
    assert jax.make_jaxpr(f)(old.initial,arrays).consts==[]


@pytest.mark.parametrize('selected',[0,1])
def test_frozen_binned_multipdf_and_native_correction(root,selected):
    root=load_caching_nll();fixture=make_shape(root,'step');x=fixture['x']
    uniform=root.RooUniform('alternative_uniform','',root.RooArgSet(x))
    index=root.RooCategory('pdf_index','');index.defineType('step',0);index.defineType('uniform',1);index.setIndex(selected)
    choice=root.RooMultiPdf('binned_choice','',index,root.RooArgList(fixture['pdf'],uniform))
    model=bind_model(root,choice,[x],fixture['coordinates']);check_model(model)
    if selected==0:assert choice.getCorrection()>0
