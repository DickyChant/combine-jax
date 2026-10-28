"""Batched constraints/counting terms preserve native conventions and complete AD."""
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import Parameter, UnsupportedModelError, walk
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.tensor.scalar import SCALAR_CLASSES, dependency_flags, import_scalars
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.constraints import pack_constraints, make_constraint_function, evaluate_constraints, validate_constraints
from combine_jax.tensor.counting import pack_counting_channels, make_counting_function, evaluate_counting, make_counting_nll
from combine_jax.tensor.bundle import TensorBundle


def scalar_export(nodes,parameters,reader,observables=(),profiles=()):
    flags = dependency_flags(nodes,parameters,observables=observables,profiles=profiles)
    roots = sorted(n for n in nodes if nodes[n][0] in SCALAR_CLASSES and not any(flags[n][1:]))
    return import_scalars(nodes,parameters,roots,fixed_value=reader,observables=observables,profiles=profiles)


def constraint_fixture(pdf,variables,parameter_names,axes):
    registry = default_registry()
    graph = registry.export([pdf])
    nodes = {n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in graph.items()}
    raw = walk(pdf)
    parameters = tuple(Parameter(n,float(variables[n].getVal()),float(variables[n].getMin()),
                                 float(variables[n].getMax()),1.,'nuisance') for n in parameter_names)
    scalars = scalar_export(nodes,parameters,lambda n:float(raw[n].getVal()))
    name = str(pdf.GetName())
    constraints = pack_constraints(nodes,[name,name],{name:axes},scalars)
    scalar_function = make_function(scalars)
    cf = make_constraint_function(constraints)
    sa,ca = jax.device_put(dict(scalars.arrays)),jax.device_put(dict(constraints.arrays))
    def value(theta):
        return cf(scalar_function(theta,sa),jnp.empty((0,)),ca)[0]
    def oracle(theta):
        context = Evaluation(graph,registry,parameter_names,theta)
        return -context.log_value(name)+(jnp.log(context.integral(name,axes)) if axes else 0.)
    return scalars,constraints,value,oracle


@pytest.mark.parametrize('fast,axis',[(False,'none'),(True,'none'),(False,'x'),(False,'mean')])
def test_gaussian_normalization_values_gradients_hessians(root,fast,axis):
    x = root.RooRealVar('gx','',.3,-2.,4.)
    mean = root.RooRealVar('gm','',.1,-3.,3.)
    sigma = root.RooRealVar('gs','',1.2,.1,3.)
    if fast: sigma.setConstant(True)
    cls = root.SimpleGaussianConstraint if fast else root.RooGaussian
    pdf = cls('gc','',x,mean,sigma)
    variables = dict(gx=x,gm=mean,gs=sigma)
    axes = () if axis == 'none' else ('gx',) if axis == 'x' else ('gm',)
    names = ('gx','gm') if fast else tuple(variables)
    s,c,f,oracle = constraint_fixture(pdf,variables,names,axes)
    value,grad,hess = jax.jit(f),jax.jit(jax.grad(f)),jax.jit(jax.hessian(f))
    nset = root.RooArgSet(*(variables[n] for n in axes))
    for theta in [np.array([.3,.1,1.2]),np.array([-1.,.6,.7]),np.array([2.8,-1.3,2.])]:
        if fast: theta = theta[:2]
        for n,v in zip(names,theta,strict=True): variables[n].setVal(float(v))
        native = -float(pdf.getLogValFast() if fast else pdf.getLogVal(nset))
        np.testing.assert_allclose(value(theta),native,rtol=1e-13,atol=1e-12)
        np.testing.assert_allclose(grad(theta),jax.grad(oracle)(theta),rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(hess(theta),jax.hessian(oracle)(theta),rtol=2e-12,atol=1e-12)
    assert len(c.metadata['names']) == 1  # Duplicate shared constraints counted once.


@pytest.mark.parametrize('rounding',[False,True])
@pytest.mark.parametrize('axis',['none','observed','mean'])
def test_poisson_normalizations_and_zero_count(root,rounding,axis):
    x = root.RooRealVar('px','',2.5,0.,40.)
    mean = root.RooRealVar('pm','',3.,.2,28.)
    if axis != 'observed': x.setConstant(True)
    pdf = root.RooPoisson('pc','',x,mean,not rounding)
    variables = dict(px=x,pm=mean)
    names = ('px','pm') if axis == 'observed' else ('pm',)
    axes = () if axis == 'none' else ('px',) if axis == 'observed' else ('pm',)
    s,c,f,oracle = constraint_fixture(pdf,variables,names,axes)
    nset = root.RooArgSet(*(variables[n] for n in axes))
    value,grad,hess = jax.jit(f),jax.jit(jax.grad(f)),jax.jit(jax.hessian(f))
    for mu in [.2,.7,3.,20.]:
        mean.setVal(mu)
        theta = np.array([2.5,mu]) if axis == 'observed' else np.array([mu])
        np.testing.assert_allclose(value(theta),-pdf.getLogVal(nset),rtol=2e-12,atol=1e-12)
        np.testing.assert_allclose(grad(theta),jax.grad(oracle)(theta),rtol=2e-12,atol=1e-12)
        np.testing.assert_allclose(hess(theta),jax.hessian(oracle)(theta),rtol=2e-12,atol=1e-12)
    if axis == 'mean': assert c.metadata['folded_normalizations'] == ['pc']


@pytest.mark.parametrize('observed',[0.,2.5,999999.,1e6,2e6])
def test_fast_poisson_preserves_native_large_count_branch(root,observed):
    x = root.RooConstVar('fx','',observed)
    mean = root.RooRealVar('fm','',max(1.,observed),0.,max(20.,2*observed))
    pdf = root.SimplePoissonConstraint('fc','',x,mean,True)
    s,c,f,oracle = constraint_fixture(pdf,{'fx':x,'fm':mean},('fm',),())
    value,grad,hess = jax.jit(f),jax.jit(jax.grad(f)),jax.jit(jax.hessian(f))
    for mu in [max(1.,observed*.9),max(2.,observed),max(3.,observed*1.1)]:
        mean.setVal(mu);theta=np.array([mu])
        np.testing.assert_allclose(value(theta),-pdf.getLogValFast(),rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(grad(theta),jax.grad(oracle)(theta),rtol=1e-12,atol=1e-11)
        np.testing.assert_allclose(hess(theta),jax.hessian(oracle)(theta),rtol=1e-12,atol=1e-11)


def test_fixed_constraint_and_normalization_fold_and_profiles_stay_dynamic(root):
    p = Parameter('r',.3,-2.,2.,1.,'poi')
    nodes = {'r':('RooRealVar',(),dict(value=.3,constant=False)),
        'profile':('RooRealVar',(),dict(value=0.,constant=False)),
        'zero':('RooConstVar',(),dict(value=0.)), 'one':('RooConstVar',(),dict(value=1.)),
        'two':('RooConstVar',(),dict(value=2.)),
        'fixed':('SimpleGaussianConstraint',('two','zero','one'),{}),
        'prof':('SimpleGaussianConstraint',('profile','zero','one'),{})}
    s = scalar_export(nodes,[p],lambda n:nodes[n][2]['value'],profiles=['profile'])
    c = pack_constraints(nodes,['fixed','prof'],{},s,profile_names=['profile'])
    assert c.metadata['folded_constraints'] == ['fixed']
    sf,cf = make_function(s),make_constraint_function(c)
    sa,ca = jax.device_put(dict(s.arrays)),jax.device_put(dict(c.arrays))
    def f(theta): return jnp.sum(cf(sf(theta,sa),theta**2,ca))
    theta = np.array([.3])
    np.testing.assert_allclose(f(theta),2.+.5*.3**4)
    np.testing.assert_allclose(jax.grad(f)(theta),[2*.3**3])
    np.testing.assert_allclose(jax.hessian(f)(theta),[[6*.3**2]])
    np.testing.assert_allclose(evaluate_constraints(c,np.array(sf(theta,sa)),theta**2),[2.,.5*.3**4])
    with pytest.raises(ValueError,match='shape mismatch'): cf(sf(theta,sa),jnp.empty((0,)),ca)


def test_empty_poisson_at_zero_mean_is_finite(root):
    x = root.RooConstVar('zx','',0.)
    mean = root.RooRealVar('zm','',0.,0.,10.)
    pdf = root.RooPoisson('zc','',x,mean,True)
    _,_,f,_ = constraint_fixture(pdf,dict(zx=x,zm=mean),('zm',),())
    point=np.array([0.])
    assert f(point)==0.
    np.testing.assert_array_equal(jax.grad(f)(point),[1.])
    np.testing.assert_array_equal(jax.hessian(f)(point),[[0.]])


def test_complete_counting_nll_matches_native_and_oracle(counting_path):
    from combine_jax.likelihood import from_workspace
    from combine_jax.root_io import open_workspace,load_caching_nll
    from combine_jax.validation import finite_difference
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        old = from_workspace(workspace)
        nodes = {n:(v.class_name,v.inputs,dict(v.attrs)) for n,v in old.nodes.items()}
        raw=walk(workspace.obj('ModelConfig').GetPdf())
        obs={n for ch in old.channels for n in ch.observables}
        s=scalar_export(nodes,old.parameters,lambda n:float(raw[n].getVal()),observables=obs)
        c=pack_constraints(nodes,old.constraints,old.constraint_axes,s)
        rows=[dict(name=ch.name,pdf=ch.pdf,observables=ch.observables,weights=ch.data.weights,masked=ch.masked) for ch in old.channels]
        ch=pack_counting_channels(nodes,rows,s)
        arrays=jax.device_put(dict(scalars=dict(s.arrays),constraints=dict(c.arrays),channels=dict(ch.arrays),offset=np.array(old.offset)))
        fn=make_counting_nll(s,c,ch)
        value,gradient,hessian=jax.jit(fn),jax.jit(jax.grad(fn)),jax.jit(jax.hessian(fn))
        native=root.cacheutils.CachingSimNLL(workspace.pdf('model_s'),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        native.clearZeroPoint();native.clearConstantZeroPoint()
        def reference(theta):
            for n,v in zip(old.parameter_names,theta,strict=True):workspace.var(n).setVal(float(v))
            return float(native.getVal())
        points=[old.initial,*np.random.default_rng(1725).uniform([-1.8,-1.8,.1],[1.8,1.8,2.5],(25,3))]
        for i,theta in enumerate(points):
            np.testing.assert_allclose(value(theta,arrays),reference(theta),atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(gradient(theta,arrays),old.value_and_grad(theta)[1],atol=1e-11,rtol=1e-11)
            np.testing.assert_allclose(gradient(theta,arrays),finite_difference(reference,theta,np.full(3,2e-4)),atol=1e-8,rtol=1e-6)
            if i<3:np.testing.assert_allclose(hessian(theta,arrays),old.hessian(theta),atol=1e-11,rtol=1e-11)
        traced=jax.make_jaxpr(fn)(old.initial,arrays)
        assert traced.consts==[]
        shifted={**arrays,'offset':jnp.array(13.)}
        np.testing.assert_allclose(value(old.initial,shifted)-value(old.initial,arrays),13.,atol=1e-12)


def test_counting_padding_volume_empty_data_masks_and_fixed_channel():
    p=Parameter('r',1.,0.,5.,1.,'poi')
    nodes={'r':('RooRealVar',(),dict(value=1.)), 'two':('RooConstVar',(),dict(value=2.)),
        'x':('RooRealVar',(),dict(value=0.,minimum=-2.,maximum=3.)),
        'u':('RooUniform',('x',),{}),
        'a':('RooAddPdf',('u','u','r','two'),dict(extended=True,pdfs=('u','u'),coefs=('r','two'))),
        'b':('RooAddPdf',('u','r'),dict(extended=True,pdfs=('u',),coefs=('r',))),
        'f':('RooAddPdf',('u','two'),dict(extended=True,pdfs=('u',),coefs=('two',)))}
    s=scalar_export(nodes,[p],lambda n:nodes[n][2]['value'],observables=['x'])
    rows=[dict(name='a',pdf='a',observables=('x',),weights=[0.,1.5,2.5]),
          dict(name='b',pdf='b',observables=('x',),weights=[]),
          dict(name='fixed',pdf='f',observables=('x',),weights=[1.]),
          dict(name='masked',pdf='unsupported',observables=(),weights=[100.],masked=True)]
    c=pack_counting_channels(nodes,rows,s)
    assert c.metadata['folded_channels']==['fixed','masked']
    sf,cf=make_function(s),make_counting_function(c)
    sa,ca=jax.device_put(dict(s.arrays)),jax.device_put(dict(c.arrays))
    theta=jnp.array([1.4]);sv=sf(theta,sa)
    expected=np.array([4*np.log(5)+3.4-4-4*np.log(3.4/4),1.4,np.log(5)+1-np.log(2),0.])
    np.testing.assert_allclose(cf(sv,ca),expected,atol=1e-13)
    np.testing.assert_allclose(evaluate_counting(c,np.asarray(sv)),expected,atol=1e-13)
    with pytest.raises(UnsupportedModelError,match='requires an extended'):
        pack_counting_channels(nodes,[dict(name='bad',pdf='u',observables=('x',),weights=[1.])],s)


def test_complete_workspace_card_artifact_roundtrips_without_frameworks(counting_path,tmp_path):
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path(__file__).parent/'cards/counting.txt')
    path=tmp_path/'model';model.save(path)
    loaded=TensorBundle.load(path)
    assert loaded.structure_key==model.structure_key
    assert loaded.metadata['provenance']['all_card_channels_imported']
    function=jax.jit(make_model_function(loaded))
    point=loaded.arrays['scalars__parameter_initial']
    np.testing.assert_allclose(function(point,jax.device_put(dict(loaded.arrays))),evaluate_model_numpy(loaded,point),atol=1e-12)
    script='''import sys, numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import evaluate_model_numpy
m=TensorBundle.load(sys.argv[1])
assert np.isfinite(evaluate_model_numpy(m,m.arrays['scalars__parameter_initial']))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(path)],check=True,capture_output=True)


def test_missing_bindings_bad_normalization_and_duplicate_outputs_rejected():
    p=Parameter('x',1.,0.,10.,1.,'nuisance')
    nodes={'x':('RooRealVar',(),dict(value=1.,minimum=0.,maximum=10.)),
           'one':('RooConstVar',(),dict(value=1.)),
           'c':('RooPoisson',('x','one'),dict(no_rounding=True))}
    s=scalar_export(nodes,[p],lambda n:nodes[n][2]['value'])
    with pytest.raises(UnsupportedModelError,match='must be a direct RooRealVar'):
        pack_constraints(nodes,['c'],{'c':('one',)},s)
    with pytest.raises(UnsupportedModelError,match='unsupported constraint normalization'):
        pack_constraints(nodes,['c'],{'c':('x','one')},s)
    altered={**nodes,'c':('RooPoisson',('x','missing'),dict(no_rounding=True))}
    with pytest.raises(UnsupportedModelError,match='bindings missing'):
        pack_constraints(altered,['c'],{},s)
    c=pack_constraints(nodes,['c'],{},s)
    arrays=dict(c.arrays);arrays['output_0']=np.array([1],dtype=np.int32)
    with pytest.raises(ValueError,match='exactly once'):
        validate_constraints(TensorBundle(c.metadata,arrays))
