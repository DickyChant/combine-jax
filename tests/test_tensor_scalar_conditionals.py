"""Mask inactive scalar arithmetic before AD; retain native selected branches."""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.formula import evaluate_formula, parse_formula
from combine_jax.graph import Parameter
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import evaluate_numpy, import_scalars, validate_program
from combine_jax.validation import finite_difference


EXPRESSIONS = [
    '@0<0 ? sqrt(-@0)+log(-@0) : (@0>0 ? sqrt(@0)+log(@0) : 0)',
    '@0==0 ? 3 : sin(@0)/@0',
    '@0>0 ? pow(@0,1.3) : (@0<0 ? pow(-@0,1.3) : 0)',
]


def bundle(expression, copies=1):
    nodes={};parameters=[];roots=[]
    for i in range(copies):
        x='x'+str(i);name='f'+str(i)
        nodes[x]=('RooRealVar',(),dict(value=.3))
        nodes[name]=('RooFormulaVar',(x,),dict(ast=parse_formula(expression,(x,))))
        parameters.append(Parameter(x,.3,-3.,3.,1.,'nuisance'));roots.append(name)
    return import_scalars(nodes,parameters,roots,fixed_value=lambda _:pytest.fail('Unexpected fold'))


@pytest.mark.parametrize('expression',EXPRESSIONS)
def test_masked_scalar_branches_match_root_and_oracle_derivatives(root,expression):
    x=root.RooRealVar('branch_x','',.3,-3.,3.)
    pdf=root.RooFormulaVar('branch_f',expression,root.RooArgList(x))
    b=bundle(expression);a=jax.device_put(dict(b.arrays));f=make_function(b)
    value=jax.jit(f);gradient=jax.jit(jax.jacrev(f));hessian=jax.jit(jax.jacfwd(jax.jacrev(f)))
    tree=parse_formula(expression,('branch_x',));oracle=lambda t:evaluate_formula(tree,[t[0]])
    refg=jax.jit(jax.grad(oracle));refh=jax.jit(jax.hessian(oracle))
    for theta in [np.array([v]) for v in [0.,-2.,-.5,.5,2.,*np.random.default_rng(42).uniform(-2.,2.,25)]]:
        x.setVal(float(theta[0]));expected=pdf.getVal()
        with np.errstate(all='raise'):
            np.testing.assert_allclose(evaluate_numpy(b,theta),[expected],rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(value(theta,a),[expected],rtol=2e-14,atol=2e-14)
        np.testing.assert_allclose(gradient(theta,a)[0],refg(theta),rtol=2e-13,atol=2e-13)
        np.testing.assert_allclose(hessian(theta,a)[0],refh(theta),rtol=2e-12,atol=2e-12)
        if abs(theta[0])>.01:
            def native(t):x.setVal(float(t[0]));return pdf.getVal()
            np.testing.assert_allclose(gradient(theta,a)[0],finite_difference(native,theta,[2e-4]),rtol=1e-7,atol=1e-9)


def test_conditional_transport_and_batch_graph_size(tmp_path):
    expression=EXPRESSIONS[0]
    one=bundle(expression);many=bundle(expression,40)
    assert one.metadata['batches']==many.metadata['batches']
    traces=[]
    for b in (one,many):
        trace=jax.make_jaxpr(make_function(b))(b.arrays['parameter_initial'],jax.device_put(dict(b.arrays)))
        assert not trace.consts;traces.append(len(trace.jaxpr.eqns))
    assert traces[0]==traces[1]
    many.save(tmp_path/'model')
    script='''import sys,numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
b=TensorBundle.load(sys.argv[1]);t=np.zeros(40);assert np.isfinite(evaluate_numpy(b,t)).all()
assert not any(k in sys.modules for k in ('ROOT','jax','torch'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'model')],check=True,capture_output=True)
    loaded=TensorBundle.load(tmp_path/'model');validate_program(loaded)


def test_active_invalid_branch_is_not_hidden():
    b=bundle('@0>=0 ? @0*2 : log(-1)');f=jax.jit(make_function(b));a=jax.device_put(dict(b.arrays))
    np.testing.assert_allclose(f(np.array([.3]),a),[.6])
    np.testing.assert_allclose(jax.jit(jax.jacrev(f))(np.array([.3]),a),[[2.]])
    assert np.isnan(f(np.array([-.3]),a)[0])
