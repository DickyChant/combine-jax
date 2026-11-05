"""Native numerical shape queries: exact normalization, AD and neutral transport."""
from array import array
import copy
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.binned_channels import (
    pack_binned_channels,make_binned_function,evaluate_binned_channels,validate_binned_channels)
from combine_jax.validation import finite_difference
from test_tensor_binned_channels import bind_model


def fixture(root,kind):
    x=root.RooRealVar('numeric_x','',125.,110.,160.)
    variables=[];held=[]
    if kind in ('breit_wigner','photon'):
        a=root.RooRealVar('numeric_a','',-.3,-1.2,-.05);variables.append(a)
        expression='2.5*exp(@0*@1/100.0)/(pow(@0-91.2,2)+pow(2.5/2,2))'
        if kind=='photon':
            f=root.RooRealVar('numeric_f','',.2,.01,.5);variables.append(f)
            expression='2.5*exp(@1*@0/100)/(pow(@0-91,2)+pow(2.5/2.0,2))+1000*@2*@2*exp(@1*@0)/pow(@0,2)'
        shape=root.RooGenericPdf('numeric_pdf',expression,root.RooArgList(x,*variables))
    elif kind in ('landau','fixed_landau'):
        mean=root.RooRealVar('numeric_mean','',120.,119.,121.);width=root.RooRealVar('numeric_width','',5.,4.,7.)
        if kind=='fixed_landau':mean.setConstant(True);width.setConstant(True)
        else:variables.extend([mean,width])
        held.extend([mean,width]);shape=root.RooGenericPdf('numeric_pdf','TMath::Landau(@0,@1,@2)',root.RooArgList(x,mean,width))
    elif kind=='fixed_spline':
        knots=array('d',[110.,118.,125.,139.,145.,153.,160.]);contents=array('d',[1.2,.8,1.,.7,.8,.5,.3])
        spline=root.RooSpline1D('numeric_spline','',x,len(knots),knots,contents,'CSPLINE')
        held.extend([knots,contents,spline]);shape=root.RooGenericPdf('numeric_pdf','@0',root.RooArgList(spline))
    else:
        a=root.RooRealVar('numeric_a','',-.03,-.06,.01)
        b=root.RooRealVar('numeric_b','',1e-4,-2e-4,2e-4)
        c=root.RooRealVar('numeric_c','',2.,1.3,2.8)
        variables.extend([a,b,c]);args=[x,*variables]
        if kind=='modz_bernstein':
            coef=[root.RooRealVar('numeric_c'+str(i),'',.01,-.02,.02) for i in range(3)]
            variables.extend(coef);args.append(root.RooArgList(*coef))
        shape=root.RooModZPdf('numeric_pdf','',*args)
    model=bind_model(root,shape,[x],{'numeric_x':np.linspace(111.,159.,9)})
    return locals()


@pytest.mark.parametrize('kind',['breit_wigner','photon','landau','fixed_landau','fixed_spline','modz','modz_bernstein'])
def test_numeric_shapes_native_nll_gradient_hessian(root,kind):
    root=load_caching_nll();f=fixture(root,kind);model=f['model'];old=model['old'];arrays=model['arrays']
    value=jax.jit(model['function']);grad=jax.jit(jax.grad(model['function']));hess=jax.jit(jax.hessian(model['function']))
    rng=np.random.default_rng(957)
    points=[old.initial]+[np.array([rng.uniform(p.minimum+.1*(p.maximum-p.minimum),p.maximum-.1*(p.maximum-p.minimum))
        for p in old.parameters]) for _ in range(25)]
    steps=np.array([1e-8 if p.name=='numeric_b' else 1e-5 for p in old.parameters])
    for i,theta in enumerate(points):
        native=model['native'](theta)
        np.testing.assert_allclose(value(theta,arrays),native,atol=1e-9,rtol=0.)
        np.testing.assert_allclose(grad(theta,arrays),old.value_and_grad(theta)[1],atol=2e-8,rtol=1e-10)
        scalars=np.asarray(model['sf'](theta,arrays['s']))
        np.testing.assert_allclose(evaluate_binned_channels(model['binned'],scalars),[native],atol=1e-9,rtol=0.)
        if i<2:
            np.testing.assert_allclose(grad(theta,arrays),finite_difference(model['native'],theta,steps),atol=2e-8,rtol=1e-4)
            np.testing.assert_allclose(hess(theta,arrays),old.hessian(theta),atol=2e-8,rtol=1e-10)
    assert jax.make_jaxpr(model['function'])(old.initial,arrays).consts==[]
    if kind.startswith('fixed'):
        assert not any(q['operation'].startswith(('numeric_','gridexpr_')) for q in model['binned'].metadata['program'])
    else:
        for entry in model['binned'].metadata['program']:
            if entry['operation'].endswith('_integral'):
                assert model['binned'].arrays[entry['buffers']['outputs']].shape[1]==1


def test_numeric_shared_batches_structural_gradient_hvp_transport(root,tmp_path):
    root=load_caching_nll();f=fixture(root,'breit_wigner');m=f['model']
    many=pack_binned_channels(m['nodes'],[{**m['rows'][0],'name':'copy'+str(i)} for i in range(3)],m['scalar'])
    scalar=np.asarray(m['sf'](m['old'].initial,m['arrays']['s']));buffers=jax.device_put(dict(many.arrays))
    traces=[jax.make_jaxpr(make_binned_function(bundle))(scalar,jax.device_put(dict(bundle.arrays)))
            for bundle in (m['binned'],many)]
    assert len(traces[0].jaxpr.eqns)==len(traces[1].jaxpr.eqns) and all(t.consts==[] for t in traces)
    functions=[]
    for policy in ('ad','structural','structural_sequential'):
        kernel=make_binned_function(many,gradient=policy)
        def function(theta,kernel=kernel):return jnp.sum(kernel(m['sf'](theta,m['arrays']['s']),buffers))
        functions.append(function)
    theta=m['old'].initial;direction=np.linspace(.2,.7,len(theta))
    reference=jax.jit(jax.value_and_grad(functions[0]))(theta)
    hvp=jax.jit(lambda t:jax.jvp(jax.grad(functions[0]),(t,),(jnp.asarray(direction),))[1])(theta)
    for fun in functions[1:]:
        result=jax.jit(jax.value_and_grad(fun))(theta)
        np.testing.assert_allclose(result[0],reference[0],atol=1e-10,rtol=0.)
        np.testing.assert_allclose(result[1],reference[1],atol=2e-8,rtol=1e-10)
        np.testing.assert_allclose(jax.jit(lambda t:jax.jvp(jax.grad(fun),(t,),(jnp.asarray(direction),))[1])(theta),hvp,atol=2e-8,rtol=1e-10)
    m['scalar'].save(tmp_path/'scalar');many.save(tmp_path/'binned')
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
    metadata=copy.deepcopy(many.metadata)
    entry=next(e for e in metadata['program'] if e['operation']=='numeric_generic_integral')
    entry['spec']['expression']=['argument',999]
    # Match both schedule representations so the numeric-expression validator is exercised.
    metadata['schedule'][2]['schedule']=metadata['program']
    with pytest.raises(ValueError,match='Invalid numerical formula index'):
        validate_binned_channels(TensorBundle(metadata,many.arrays))


def test_modz_mixed_bernstein_degrees_are_distinct_structures(root):
    root=load_caching_nll();f=fixture(root,'modz_bernstein');x=f['x'];p=f['variables']
    simple=root.RooModZPdf('numeric_simple','',x,*p[:3])
    fraction=root.RooRealVar('numeric_fraction','',.4,.2,.7)
    mixture=root.RooAddPdf('numeric_mixture','',root.RooArgList(simple,f['shape']),root.RooArgList(fraction))
    m=bind_model(root,mixture,[x],{'numeric_x':np.linspace(111.,159.,9)})
    entries=[e for e in m['binned'].metadata['program'] if e['operation']=='numeric_modz_integral']
    assert {e['spec']['parameter_count'] for e in entries}=={5,8}
    theta=m['old'].initial
    np.testing.assert_allclose(jax.jit(m['function'])(theta,m['arrays']),m['native'](theta),atol=1e-9,rtol=0.)
    np.testing.assert_allclose(jax.jit(jax.grad(m['function']))(theta,m['arrays']),m['old'].value_and_grad(theta)[1],atol=2e-8,rtol=1e-10)
