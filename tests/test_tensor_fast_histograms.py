"""Native bin values, integrals and AD for neutral fast-histogram batches."""
from itertools import product
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import parameters,walk,UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.fast_histograms import (pack_fast_histograms,pack_fast_queries,
    make_fast_function,make_query_function,evaluate_fast_histograms,evaluate_fast_queries,
    validate_fast_histograms,validate_fast_queries)
from test_tensor_likelihood_terms import scalar_export
from test_vertical import make_fast_vertical
from test_vertical_nd import make_multidimensional


def prepare(pdf,variables,coordinates,requests=None):
    axes=tuple(str(v.GetName()) for v in variables);registry=default_registry()
    graph=registry.export([pdf],axes);info=parameters(pdf,axes);raw=walk(pdf)
    nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in graph.items()}
    scalar=scalar_export(nodes,info,lambda n:float(raw[n].getVal()),observables=axes)
    name=str(pdf.GetName());hist=pack_fast_histograms(nodes,[name],scalar)
    requests=requests or [dict(name='points',pdf=name,mode='density',coordinates=coordinates,axes=axes),
                         dict(name='integral',pdf=name,mode='integral',axes=axes)]
    queries=pack_fast_queries(hist,requests)
    sf=make_function(scalar);hf=make_fast_function(hist);qf=make_query_function(queries)
    arrays=jax.device_put(dict(s=dict(scalar.arrays),h=dict(hist.arrays),q=dict(queries.arrays)))
    def evaluate(theta,arrays):return qf(hf(sf(theta,arrays['s']),arrays['h']),arrays['q'])
    def oracle(theta):
        outputs=[]
        for r in requests:
            context=Evaluation(graph,registry,tuple(p.name for p in info),theta,r.get('coordinates',{}))
            outputs.append(jnp.atleast_1d(getattr(context,r['mode'])(r['pdf'],r['axes'])))
        return jnp.concatenate(outputs)
    return scalar,hist,queries,info,raw,evaluate,oracle,arrays


@pytest.mark.parametrize('old,logarithmic,active,region,freeze',[
    (False,False,False,.7,False),(False,True,False,.7,False),
    (False,False,True,.7,True),(False,True,True,.7,True),
    (True,False,False,0.,False),(True,True,False,0.,False)])
def test_native_one_dimensional_bins_tails_restricted_integrals_and_ad(root,old,logarithmic,active,region,freeze):
    x,a,b,pdf,held=make_fast_vertical(root,old=old,logarithmic=logarithmic,active=active,region=region)
    if freeze:b.setVal(-.31);b.setConstant(True)
    # Default ranges need not span the stored grid; density remains selfNormalized.
    x.setRange(.1,6.)
    coords=np.array([.1,.25,.5,.500001,1.25,2.,3.,4.,5.5,6.])
    s,h,q,info,raw,f,oracle,arrays=prepare(pdf,[x],{'vx':coords})
    value=jax.jit(f);grad=jax.jit(jax.jacrev(f));hess=jax.jit(jax.hessian(lambda t,arrays:jnp.sum(f(t,arrays))))
    for a_value in (-1.3,0.,.7,1.2):
        theta=np.array([a_value if p.name=='va' else -.31 for p in info])
        for p,v in zip(info,theta,strict=True):raw[p.name].setVal(float(v))
        reference=[]
        for c in coords:x.setVal(float(c));reference.append(float(pdf.getVal(root.RooArgSet(x))))
        edges=h.arrays[h.metadata['histograms'][0]['edges'][0]];integral=0.
        for lo,hi in zip(edges[:-1],edges[1:],strict=True):
            lo,hi=max(lo,x.getMin()),min(hi,x.getMax())
            if hi>lo:x.setVal(float((lo+hi)/2));integral+=float(pdf.getVal())*(hi-lo)
        reference.append(integral)
        np.testing.assert_allclose(value(theta,arrays),reference,atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(evaluate_fast_queries(q,evaluate_fast_histograms(h,np.asarray(make_function(s)(theta,arrays['s'])))),reference,atol=1e-13,rtol=1e-13)
        np.testing.assert_allclose(grad(theta,arrays),jax.jacrev(oracle)(theta),atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(hess(theta,arrays),jax.hessian(lambda t:jnp.sum(oracle(t)))(theta),atol=1e-12,rtol=1e-12)
    assert jax.make_jaxpr(f)(theta,arrays).consts==[]


@pytest.mark.parametrize('cls,conditional,logarithmic',list(product(
    ('FastVerticalInterpHistPdf2D','FastVerticalInterpHistPdf2D2','FastVerticalInterpHistPdf3D'),(False,True),(False,True))))
def test_native_nd_all_partial_integrals_conditioning_and_ad(root,cls,conditional,logarithmic):
    variables,a,b,pdf,edges,held=make_multidimensional(root,cls,conditional,logarithmic)
    axes=tuple(str(v.GetName()) for v in variables)
    coordinates=np.array(list(product(*(np.sort(np.r_[e,(e[:-1]+e[1:])/2]) for e in edges))))
    bound={n:coordinates[:,i] for i,n in enumerate(axes)}
    requests=[dict(name='data',pdf='ndpdf',mode='density',axes=axes,coordinates=bound)]
    fixed={n:float((e[1]+e[2])/2) for n,e in zip(axes,edges,strict=True)}
    for mask in range(1,2**len(axes)):
        requests.append(dict(name=f'integral{mask}',pdf='ndpdf',mode='integral',
            axes=tuple(n for i,n in enumerate(axes) if mask&(1<<i)),coordinates=fixed))
    s,h,q,info,raw,f,oracle,arrays=prepare(pdf,variables,bound,requests)
    score=lambda t,arrays:jnp.sum(f(t,arrays))
    value=jax.jit(f);grad=jax.jit(jax.grad(score));hess=jax.jit(jax.hessian(score))
    for theta in [np.array([.2,-.31]),np.array([-1.1,.4])]:
        a.setVal(float(theta[0]));b.setVal(float(theta[1]));ref=[]
        for row in coordinates:
            for v,c in zip(variables,row,strict=True):v.setVal(float(c))
            ref.append(float(pdf.getVal(root.RooArgSet(*variables))))
        shape=tuple(len(e)-1 for e in edges);grid=np.zeros(shape)
        for index in np.ndindex(shape):
            for v,e,i in zip(variables,edges,index,strict=True):v.setVal(float((e[i]+e[i+1])/2))
            grid[index]=pdf.getVal()
        for request in requests[1:]:
            total=0.
            for index in np.ndindex(shape):
                if any(index[i]!=1 for i,n in enumerate(axes) if n not in request['axes']):continue
                total+=grid[index]*np.prod([edges[i][index[i]+1]-edges[i][index[i]] for i,n in enumerate(axes) if n in request['axes']])
            ref.append(total)
        np.testing.assert_allclose(value(theta,arrays),ref,atol=3e-13,rtol=1e-13)
        np.testing.assert_allclose(grad(theta,arrays),jax.grad(lambda t:jnp.sum(oracle(t)))(theta),atol=1e-11,rtol=1e-12)
        np.testing.assert_allclose(hess(theta,arrays),jax.hessian(lambda t:jnp.sum(oracle(t)))(theta),atol=1e-11,rtol=1e-12)
    assert jax.make_jaxpr(f)(theta,arrays).consts==[]


def test_frozen_histograms_queries_and_neutral_transport(root,tmp_path):
    x,a,b,pdf,held=make_fast_vertical(root,logarithmic=True,active=True)
    a.setVal(.4);b.setVal(-.2);a.setConstant(True);b.setConstant(True)
    s,h,q,info,raw,f,oracle,arrays=prepare(pdf,[x],{'vx':np.array([.25,.5,4.,7.])})
    assert not h.metadata['schedule'] and h.metadata['folded_histograms']==['vertical']
    assert q.metadata['folded_queries']==['points','integral']
    assert q.arrays['gather_indices'].size==q.arrays['integral_indices'].size==0
    for name,bundle in [('scalars',s),('histograms',h),('queries',q)]:bundle.save(tmp_path/name)
    script='''import sys, numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.fast_histograms import evaluate_fast_histograms,evaluate_fast_queries
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalars');h=TensorBundle.load(p/'histograms');q=TensorBundle.load(p/'queries')
assert np.all(np.isfinite(evaluate_fast_queries(q,evaluate_fast_histograms(h,evaluate_numpy(s,s.arrays['parameter_initial'])))))
assert not any(n in sys.modules for n in ('ROOT','jax','torch'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)
    np.testing.assert_allclose(jax.jit(f)(np.empty(0),arrays),oracle(np.empty(0)),atol=1e-13)


def test_same_structure_histograms_batch_with_flat_trace_and_reject_bad_maps(root):
    x,a,b,pdf,held=make_fast_vertical(root)
    s,h,q,info,raw,f,oracle,arrays=prepare(pdf,[x],{'vx':np.array([.2])})
    graph=default_registry().export([pdf],['vx']);nodes={n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in graph.items()}
    for i in range(8):nodes[f'copy{i}']=nodes['vertical']
    many=pack_fast_histograms(nodes,['vertical']+[f'copy{i}' for i in range(8)],s)
    sv=np.asarray(make_function(s)(np.array([.3,-.2]),arrays['s']))
    single_fn=make_fast_function(h);many_fn=make_fast_function(many)
    single_trace=jax.make_jaxpr(single_fn)(sv,jax.device_put(dict(h.arrays)))
    many_trace=jax.make_jaxpr(many_fn)(sv,jax.device_put(dict(many.arrays)))
    assert len(single_trace.jaxpr.eqns)==len(many_trace.jaxpr.eqns) and single_trace.consts==many_trace.consts==[]
    np.testing.assert_allclose(np.asarray(jax.jit(many_fn)(sv,jax.device_put(dict(many.arrays)))).reshape(9,4),
        np.broadcast_to(evaluate_fast_histograms(h,sv),(9,4)),atol=1e-13)
    bad=dict(many.arrays);key=many.metadata['schedule'][0]['buffers']['outputs'];bad[key]=np.zeros_like(bad[key])
    with pytest.raises(ValueError,match='coverage'):validate_fast_histograms(TensorBundle(many.metadata,bad))
    bad=dict(q.arrays);bad['gather_indices']=np.full_like(bad['gather_indices'],100000)
    with pytest.raises(ValueError,match='query gather'):validate_fast_queries(TensorBundle(q.metadata,bad))
    requests=[dict(name='points',pdf='vertical',coordinates={'vx':np.array([.2,.8])},axes=('vx',)),
              dict(name='integral',pdf='vertical',mode='integral',axes=('vx',))]
    duplicate=pack_fast_queries(h,requests)
    for target in ('gather_outputs','integral_outputs'):
        bad=dict(duplicate.arrays);bad[target]=bad[target].copy()
        bad[target][-1]=bad['gather_outputs'][0]
        with pytest.raises(ValueError,match='Overlapping query outputs'):
            validate_fast_queries(TensorBundle(duplicate.metadata,bad))
    with pytest.raises(UnsupportedModelError,match='normalization axes'):
        pack_fast_queries(h,[dict(name='bad',pdf='vertical',axes=('foreign',),coordinates={'vx':.3})])
    with pytest.raises(UnsupportedModelError,match='missing independent scalar'):
        bad_nodes={**nodes,'vertical':(nodes['vertical'][0],nodes['vertical'][1],{**nodes['vertical'][2],'coefs':('vx','vb')})}
        pack_fast_histograms(bad_nodes,['vertical'],s)
