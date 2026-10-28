"""Conditional shape batching, exact coordinate compression and inactive AD branches."""
from itertools import product
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
from test_tensor_binned_channels import bind_model,check_model
from test_products import make_product


@pytest.mark.parametrize('mode,nested',list(product(('nset','cset'),(False,True))))
def test_conditional_product_native_nll_and_ad(root,mode,nested):
    root=load_caching_nll();fixture=make_product(root,mode,nested)
    coords=np.array(list(product((-1.8,-.3,.4,2.6),(-2.7,.6,1.2,3.6))))
    model=bind_model(root,fixture['pdf'],[fixture['x'],fixture['y']],dict(zip(('px','py'),coords.T,strict=True)))
    check_model(model)
    integral=[q for q in model['binned'].metadata['analytic_grid_queries'] if ':gy:integral:' in q['name']]
    assert integral and all(q['grid_rows']==4 and q['data_rows']==16 for q in integral)


def conditional_fixture(root,family,*,fixed=False,nested=False):
    x=root.RooRealVar('gx','',2.,.5,5.);y=root.RooRealVar('gy','',.3,-1.,1.);z=root.RooRealVar('gz','',0.,-1.,1.)
    theta=root.RooRealVar('grid_theta','',.03,-.2,.2)
    if fixed:theta.setConstant(True)
    mean=root.RooFormulaVar('grid_mean','2+0.1*@0+@1',root.RooArgList(y,theta))
    expression=('2+0.1*(@0>@1 ? log(@0-@1) : (@0<@1 ? sqrt(@1-@0) : 0))' if nested
        else '1+0.1*(@0>@1 ? sqrt(@0-@1) : sqrt(@1-@0))')
    width=root.RooFormulaVar('grid_width',expression,root.RooArgList(y,theta))
    slope=root.RooFormulaVar('grid_slope','-0.3+0.05*@0+0.02*@1',root.RooArgList(y,theta))
    constants=[root.RooConstVar(f'grid_c{i}','',v) for i,v in enumerate((1.,1.2,2.,1.7,2.5))]
    if family=='gaussian':shape=root.RooGaussian('grid_shape','',x,mean,width)
    elif family=='doublecb':shape=root.RooDoubleCB('grid_shape','',x,mean,width,*constants[1:])
    elif family=='erfexp':shape=root.RooErfExpPdf('grid_shape','',x,slope,mean,width)
    elif family=='exponential':shape=root.RooExponential('grid_shape','',x,slope)
    elif family=='power':shape=root.RooPower('grid_shape','',x,slope)
    elif family=='bernstein':shape=root.RooBernstein('grid_shape','',x,root.RooArgList(constants[0],width,constants[1]))
    else:raise ValueError(family)
    marginal=root.RooUniform('grid_marginal','',root.RooArgSet(y,z))
    pdf=root.RooProdPdf('grid_product','',root.RooArgSet(marginal),root.RooFit.Conditional(root.RooArgSet(shape),root.RooArgSet(x)))
    points=np.array(list(product((.7,2.,4.5),(-.8,-.4,.4,.8),(-.6,.2,.6))))
    coordinates=dict(zip(('gx','gy','gz'),points.T,strict=True))
    return locals()


@pytest.mark.parametrize('family',['gaussian','doublecb','erfexp','exponential','power','bernstein'])
def test_conditional_analytic_shapes_and_safe_piecewise_gradients(root,family):
    root=load_caching_nll();f=conditional_fixture(root,family)
    model=bind_model(root,f['pdf'],[f['x'],f['y'],f['z']],f['coordinates']);check_model(model)
    queries=model['binned'].metadata['analytic_grid_queries']
    assert {q['distinct_grid_rows'] for q in queries}=={4,12}
    assert {q['grid_rows'] for q in queries}=={4,16} and all(q['data_rows']==64 for q in queries)
    layout=model['binned'].metadata['data_layouts'][0]
    assert (layout['original_rows'],layout['positive_rows'],layout['packed_rows'])==(36,35,64)
    assert all('gz' not in q['coordinate_axes'] for q in queries)


def test_nested_inactive_branches_and_fixed_coordinate_folding(root):
    root=load_caching_nll();f=conditional_fixture(root,'gaussian',nested=True)
    model=bind_model(root,f['pdf'],[f['x'],f['y'],f['z']],f['coordinates']);check_model(model)
    f=conditional_fixture(root,'erfexp',fixed=True)
    model=bind_model(root,f['pdf'],[f['x'],f['y'],f['z']],f['coordinates']);check_model(model)
    assert all(q['folded'] for q in model['binned'].metadata['analytic_grid_queries'])
    assert not any(q['operation'].startswith(('gridexpr_','gridanalytic_')) for q in model['binned'].metadata['program'])


def test_grid_transport_shared_trace_and_corruption(root,tmp_path):
    root=load_caching_nll();f=conditional_fixture(root,'gaussian')
    model=bind_model(root,f['pdf'],[f['x'],f['y'],f['z']],f['coordinates'])
    many=pack_binned_channels(model['nodes'],[{**model['rows'][0],'name':f'copy{i}'} for i in range(3)],model['scalar'])
    values=np.asarray(model['sf'](model['old'].initial,model['arrays']['s']))
    traces=[jax.make_jaxpr(make_binned_function(b))(values,jax.device_put(dict(b.arrays))) for b in (model['binned'],many)]
    assert len(traces[0].jaxpr.eqns)==len(traces[1].jaxpr.eqns) and all(t.consts==[] for t in traces)
    np.testing.assert_allclose(jax.jit(make_binned_function(many))(values,jax.device_put(dict(many.arrays))),model['native'](model['old'].initial),atol=1e-10)
    model['scalar'].save(tmp_path/'scalar');many.save(tmp_path/'channels')
    script='''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalar');b=TensorBundle.load(p/'channels')
assert np.all(np.isfinite(evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)
    bad=dict(many.arrays);entry=next(q for q in many.metadata['program'] if q['operation']=='gridexpr_mul')
    key=entry['buffers']['operand_mask'];bad[key]=np.zeros_like(bad[key])
    with pytest.raises(ValueError,match='Invalid coordinate expression arity'):validate_binned_channels(TensorBundle(many.metadata,bad))


def test_integration_variable_dependence_is_rejected(root):
    root=load_caching_nll();f=conditional_fixture(root,'gaussian')
    width=root.RooFormulaVar('bad_width','1+@0',root.RooArgList(f['x']))
    shape=root.RooGaussian('bad_integrated_shape','',f['x'],f['mean'],width)
    pdf=root.RooProdPdf('bad_integrated_product','',root.RooArgSet(f['marginal']),root.RooFit.Conditional(root.RooArgSet(shape),root.RooArgSet(f['x'])))
    with pytest.raises(UnsupportedModelError,match='bad_integrated_shape.*depend on integration/data variable gx'):
        bind_model(root,pdf,[f['x'],f['y'],f['z']],f['coordinates'])
