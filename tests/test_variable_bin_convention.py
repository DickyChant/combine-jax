"""Native-v10 golden regression on the standalone synthetic 3D card."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import jax
import numpy as np
import pytest

from combine_jax.data import bind_channels
from combine_jax.methods.workspace import prepare_expected_workspace
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info


@pytest.fixture(scope='module')
def density_fixture(tmp_path_factory):
    directory=tmp_path_factory.mktemp('variable_bin')
    source=directory/'source'
    producer=Path('scripts/reproduce_variable_bin_volume.py')
    reference=json.loads(Path('reports/variable_bin_reproducer_v10.json').read_text())
    assert reference['script_sha256']==hashlib.sha256(producer.read_bytes()).hexdigest()
    run=subprocess.run([sys.executable,str(producer),'--pip-stack','--output-dir',str(source)],
                       capture_output=True,text=True,timeout=90)
    assert run.returncode==0,run.stdout+run.stderr
    model,preparation=prepare_expected_workspace(source/'workspace.root',source/'card.txt',directory/'prepared')
    return model,preparation,reference,source


def test_corrected_3d_card_v10_nll_gradient_and_asimov(density_fixture):
    model,preparation,reference,_=density_fixture
    assert model.metadata['provenance']['template_convention']=='v10_density'
    assert model_parameter_info(model).metadata['names']==reference['parameter_names']
    assert all(a.dtype!=np.dtype('float32') for a in model.arrays.values())
    fn=jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
    arrays=jax.device_put(dict(model.arrays))
    for case in reference['cases']:
        value,gradient=fn(np.array(case['point']),arrays)
        np.testing.assert_allclose(value,case['nll'],atol=1e-6,rtol=0.)
        np.testing.assert_allclose(gradient,case['gradient'],atol=1e-8,rtol=1e-4)
    with open_workspace(preparation['prepared_workspace']) as ws:
        rows=bind_channels(ws.data('combine_jax_asimov_bonly'),'CMS_channel',{'ch':('x','y','z')})['ch']
        np.testing.assert_array_equal(rows.weights,[r['weight'] for r in reference['asimov']])
        np.testing.assert_array_equal(np.column_stack([rows.coordinates[n] for n in ('x','y','z')]),
                                      [r['coordinates'] for r in reference['asimov']])


def test_density_integral_and_explicit_native_oracle(density_fixture):
    _,_,reference,source=density_fixture
    from combine_jax.registry import Evaluation
    from combine_jax.translators import default_registry
    with open_workspace(source/'shapes.root') as ws:
        pdf=ws.pdf('background');coords=np.array([r['coordinates'] for r in reference['asimov']])
        data={n:coords[:,i] for i,n in enumerate(('x','y','z'))}
        correct=default_registry(template_convention='v10_density')
        native=default_registry()
        c=Evaluation(correct.export([pdf],('x','y','z')),correct,('shape',),np.array([0.]),data)
        n=Evaluation(native.export([pdf],('x','y','z')),native,('shape',),np.array([0.]),data)
        masses=np.array([b['template_probability'] for b in reference['bins']])
        volumes=np.array([b['volume'] for b in reference['bins']])
        np.testing.assert_allclose(np.asarray(c.density('background',('x','y','z')))*volumes,masses,rtol=8e-8)
        np.testing.assert_allclose(c.integral('background',('x','y','z')),1.,rtol=0.,atol=2e-15)
        # The original oracle deliberately continues to reproduce installed v11.
        measured=json.loads((source/'result.json').read_text())
        np.testing.assert_allclose(np.asarray(n.density('background',('x','y','z')))*volumes,
                                   [b['morpher_probability'] for b in measured['bins']],atol=1e-15,rtol=0.)
        assert max(abs(np.asarray(n.density('background',('x','y','z')))*volumes-masses))>.1


def test_uncertified_conditional_variable_bin_mode_fails_before_fit(density_fixture,tmp_path):
    _,_,_,source=density_fixture
    with pytest.raises(NotImplementedError,match='v10 background-fit validation'):
        prepare_expected_workspace(source/'workspace.root',source/'card.txt',tmp_path/'conditional',run='expected')


@pytest.mark.parametrize('dimension,conditional',[(2,False),(2,True),(3,False),(3,True)])
def test_density_convention_nd_normalization(root,dimension,conditional):
    """Independent bin-mass integral including conditional x slices."""
    from array import array
    from itertools import product
    from combine_jax.registry import Evaluation
    from combine_jax.translators import default_registry
    edges=[np.array([0.,1.,4.]),np.array([0.,2.,3.,8.]),np.array([0.,1.,5.])][:dimension]
    axes=[root.RooRealVar('volume_'+n,'',.5,float(e[0]),float(e[-1])) for n,e in zip('xyz',edges)]
    for v,e in zip(axes,edges):v.setBinning(root.RooBinning(len(e)-1,array('d',e)))
    shape=tuple(len(e)-1 for e in edges)
    h=getattr(root,f'TH{dimension}D')('density_integral_hist','',
        *(arg for e in edges for arg in (len(e)-1,array('d',e))))
    h.SetDirectory(0)
    weights=np.arange(1,np.prod(shape)+1,dtype=float).reshape(shape)
    for cell in np.ndindex(shape):h.SetBinContent(*(i+1 for i in cell),weights[cell])
    d=root.RooDataHist('density_integral_data','',root.RooArgList(*axes),h)
    p=root.RooHistPdf('density_integral_input','',root.RooArgSet(*axes),d)
    m=getattr(root,f'FastVerticalInterpHistPdf{dimension}D')('density_integral_morph','',
        *axes,conditional,root.RooArgList(p),root.RooArgList(),1.,-1)
    names=tuple(v.GetName() for v in axes)
    centers=np.array(list(product(*((e[:-1]+e[1:])/2 for e in edges))))
    reg=default_registry(template_convention='v10_density')
    context=Evaluation(reg.export([m],names),reg,(),np.empty(0),
        {n:centers[:,i] for i,n in enumerate(names)})
    norm_axes=tuple(range(1 if conditional else 0,dimension))
    volumes=np.ones(shape)
    for i in norm_axes:
        broad=[1]*dimension;broad[i]=shape[i]
        volumes*=np.diff(edges[i]).reshape(broad)
    expected=weights/np.sum(weights,axis=norm_axes,keepdims=True)
    actual=np.asarray(context.density(m.GetName(),names)).reshape(shape)*volumes
    np.testing.assert_allclose(actual,expected,rtol=8e-8,atol=0.)
    np.testing.assert_allclose(np.sum(actual,axis=norm_axes),1.,rtol=0.,atol=5e-15)
