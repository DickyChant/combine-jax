"""Scalar batching must preserve values, sharing, derivatives and import boundaries."""
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import Parameter, UnsupportedModelError
from combine_jax.registry import Evaluation, Node
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import dependency_flags, import_scalars, evaluate_numpy, validate_program
from combine_jax.translators import default_registry


def fixture():
    parameters = (Parameter('theta', .2, -7., 7., 1., 'nuisance'),
                  Parameter('fixed', 2., 2., 2., 0., 'nuisance'))
    nodes = {
        'theta': ('RooRealVar', (), dict(value=.2, constant=False)),
        'fixed': ('RooRealVar', (), dict(value=2., constant=False)),
        'low': ('RooConstVar', (), dict(value=.8)),
        'high': ('RooConstVar', (), dict(value=1.3)),
        'power': ('AsymPow', ('low','high','theta'), {}),
        'fraction': ('RooRecursiveFraction', ('theta','theta'), {}),
        'formula': ('RooFormulaVar', ('theta','fixed'), dict(ast=(
            'binary', '+', ('call','exp', (('argument',0),)),
            ('binary','*',('argument',1),('constant',3.))))),
        'normalization': ('ProcessNormalization', ('theta','theta','formula'),
            dict(nominal=17., symmetric=('theta',), log_kappa=(.1,),
                 asymmetric=('theta',), log_asymmetric=((-0.2,.3),), other=('formula',))),
        'product': ('RooProduct', ('normalization','power','fraction'), {}),
        'sum': ('RooAddition', ('product','formula','power'), {}),
    }
    reads = []
    def native_fixed(name):
        reads.append(name)
        return nodes[name][2]['value']
    bundle = import_scalars(nodes, parameters, ['sum','normalization','formula','power','fraction','theta','fixed'], fixed_value=native_fixed)
    return nodes, parameters, bundle, reads


def test_values_gradients_hessians_match_old_registry():
    nodes, parameters, bundle, reads = fixture()
    registry_nodes = {n:Node(n,c,i,a) for n,(c,i,a) in nodes.items()}
    names = [p.name for p in parameters]
    def oracle(theta):
        context = Evaluation(registry_nodes, default_registry(), names, theta)
        return jnp.stack([context.value(n) for n in bundle.metadata['output_names']])
    arrays = jax.device_put(dict(bundle.arrays))
    function = make_function(bundle)
    value = jax.jit(function)
    gradient = jax.jit(jax.jacfwd(function, argnums=0))
    hessian = jax.jit(jax.jacfwd(jax.jacrev(function, argnums=0), argnums=0))
    refg, refh = jax.jacfwd(oracle), jax.jacfwd(jax.jacrev(oracle))
    for t in (-1.2, -.5, -.1, 0., .2, .5, 1.4):
        theta = np.array([t,2.])
        np.testing.assert_allclose(evaluate_numpy(bundle,theta), oracle(theta), rtol=3e-14, atol=1e-12)
        np.testing.assert_allclose(value(theta,arrays), oracle(theta), rtol=3e-14, atol=1e-12)
        np.testing.assert_allclose(gradient(theta,arrays)[:,0], refg(theta)[:,0], rtol=3e-13, atol=1e-11)
        np.testing.assert_allclose(hessian(theta,arrays)[:,0,0], refh(theta)[:,0,0], rtol=3e-12, atol=1e-10)
        np.testing.assert_array_equal(gradient(theta,arrays)[:,1],0.)
    assert set(reads) == {'fixed','low','high'}


def test_constant_fold_requires_no_observable_or_profile_dependency():
    nodes, pars, _, _ = fixture()
    nodes.update(obs=('RooRealVar',(),dict(value=3.,constant=False)),
                 profiled=('RooRealVar',(),dict(value=0.,constant=False)),
                 with_obs=('RooProduct',('fixed','obs'),{}),
                 with_profile=('RooProduct',('fixed','profiled'),{}),
                 with_float=('RooProduct',('fixed','theta'),{}))
    flags = dependency_flags(nodes,pars,observables=['obs'],profiles=['profiled'])
    assert flags['fixed'] == (False,False,False)
    assert flags['with_obs'] == (False,True,False)
    assert flags['with_profile'] == (False,False,True)
    assert flags['with_float'] == (True,False,False)
    for root in ('with_obs','with_profile'):
        with pytest.raises(UnsupportedModelError,match='cannot bind'):
            import_scalars(nodes,pars,[root],fixed_value=lambda n:pytest.fail('Unsafe fold'),
                           observables=['obs'],profiles=['profiled'])


def test_fixed_reader_folds_whole_mass_spline_without_dynamic_lowering():
    nodes = {'MX':('RooRealVar',(),dict(value=2000.,constant=True)),
             'spline':('RooSpline1D',('MX',),dict(algorithm='CSPLINE')),
             'formula':('RooFormulaVar',('spline',),dict(ast=('argument',0)))}
    read = []
    def fixed(n):
        read.append(n)
        return 17.5
    b = import_scalars(nodes,[],['formula'],fixed_value=fixed)
    np.testing.assert_array_equal(evaluate_numpy(b,[]), [17.5])
    assert read == ['formula'] and b.metadata['schedule'] == []


def test_program_is_framework_neutral_and_roundtrips(tmp_path):
    _, _, bundle, _ = fixture()
    directory = tmp_path/'bundle'
    bundle.save(directory)
    loaded = TensorBundle.load(directory)
    assert loaded.structure_key == bundle.structure_key
    np.testing.assert_array_equal(evaluate_numpy(loaded,[.2,2.]), evaluate_numpy(bundle,[.2,2.]))
    command = '''import sys
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
b = TensorBundle.load(sys.argv[1])
assert evaluate_numpy(b, [.2,2.]).shape == (7,)
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',command,str(directory)],check=True,capture_output=True)
    with pytest.raises(FileExistsError): bundle.save(directory)
    with (directory/'arrays.npz').open('ab') as stream: stream.write(b'corrupted')
    with pytest.raises(ValueError,match='checksum'): TensorBundle.load(directory)


def test_buffers_are_runtime_arguments_not_closed_over_constants():
    _,_,bundle,_ = fixture()
    function = make_function(bundle)
    arrays = jax.device_put(dict(bundle.arrays))
    traced = jax.make_jaxpr(function)(jnp.array([.2,2.]),arrays)
    assert traced.consts == []
    other = dict(bundle.arrays)
    other['initial_values'] = other['initial_values'].copy()
    slot = int(bundle.arrays['output_slots'][-1])  # Folded fixed leaf, reused in formula.
    other['initial_values'][slot] = 3.
    changed = TensorBundle(bundle.metadata,other)
    assert changed.structure_key == bundle.structure_key
    assert not np.array_equal(function(jnp.array([.2,2.]),arrays),
                              function(jnp.array([.2,2.]),jax.device_put(dict(changed.arrays))))


def test_shared_dependency_and_same_operation_batching():
    p = Parameter('x',.2,-5,5,1,'nuisance')
    nodes = {'x':('RooRealVar',(),dict(value=.2))}
    for i in range(120):
        nodes[f'c{i}'] = ('RooConstVar',(),dict(value=float(i+1)))
        nodes[f'f{i}'] = ('RooProduct',('x',f'c{i}'),{})
    nodes['alias'] = nodes['f0']
    b = import_scalars(nodes,[p],[*[f'f{i}' for i in range(120)],'alias'],
                       fixed_value=lambda n:nodes[n][2]['value'])
    assert b.metadata['batches'] == 1
    assert b.metadata['dynamic_instructions'] == 120
    assert b.arrays['output_slots'][0] == b.arrays['output_slots'][-1]
    np.testing.assert_allclose(evaluate_numpy(b,[.3])[:120], .3*np.arange(1,121))


def test_conditional_constant_branch_short_circuits_and_dynamic_rejects():
    p = Parameter('x',.2,-5,5,1,'nuisance')
    nodes = {'x':('RooRealVar',(),dict(value=.2)),
      'f':('RooFormulaVar',('x',),dict(ast=('conditional',('constant',1.),
          ('argument',0),('call','log',(('constant',-1.),)))))}
    b = import_scalars(nodes,[p],['f'],fixed_value=lambda _:pytest.fail('Unexpected fold'))
    np.testing.assert_array_equal(evaluate_numpy(b,[.2]), [.2])
    nodes['f'][2]['ast'] = ('conditional',('argument',0),('constant',1.),('constant',2.))
    with pytest.raises(UnsupportedModelError,match='Dynamic conditional'):
        import_scalars(nodes,[p],['f'],fixed_value=lambda _:0.)


def test_fixed_parameter_changes_and_malformed_program_are_rejected():
    _,_,bundle,_ = fixture()
    with pytest.raises(ValueError,match='Fixed-bound'):
        evaluate_numpy(bundle,[.2,3.])
    arrays = jax.device_put(dict(bundle.arrays))
    assert np.all(np.isnan(make_function(bundle)(jnp.array([.2,3.]),arrays)))
    altered = dict(bundle.arrays)
    name = bundle.metadata['schedule'][0]['inputs']
    altered[name] = altered[name].copy()
    altered[name][0,0] = len(altered['initial_values'])
    with pytest.raises(ValueError,match='Invalid slot'):
        validate_program(TensorBundle(bundle.metadata,altered))


def test_unknown_fixed_class_cannot_hide_behind_constant_folding():
    with pytest.raises(UnsupportedModelError,match='UnknownPdf'):
        import_scalars({'bad':('UnknownPdf',(),{})},[],['bad'],fixed_value=lambda n:1.)
    nodes = {'bad':('UnknownPdf',(),{}),
             'wrapper':('RooFormulaVar',('bad',),dict(ast=('argument',0)))}
    with pytest.raises(UnsupportedModelError,match='UnknownPdf'):
        import_scalars(nodes,[],['wrapper'],fixed_value=lambda n:pytest.fail('Unsupported dependency hidden'))


def test_workspace_plus_card_import_and_native_scalar_values(counting_path):
    from combine_jax.root_io import open_workspace
    from combine_jax.tensor.root_scalar_import import from_workspace
    card = Path(__file__).parent/'cards/counting.txt'
    with open_workspace(counting_path) as workspace:
        bundle = from_workspace(workspace,card)
        assert len(bundle.metadata['provenance']['channels']) == 2
        function = jax.jit(make_function(bundle))
        arrays = jax.device_put(dict(bundle.arrays))
        names = bundle.metadata['parameter_names']
        for theta in np.random.default_rng(2026).uniform([-.9,-.9,.2],[.9,.9,2.],(26,3)):
            for n,x in zip(names,theta,strict=True): workspace.var(n).setVal(float(x))
            native = np.array([workspace.arg(n).getVal() for n in bundle.metadata['output_names']])
            np.testing.assert_allclose(function(theta,arrays),native,rtol=2e-13,atol=1e-12)
