import subprocess
import sys

import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.graph import UnsupportedModelError
from test_cbshape import make_cbshape
from test_tensor_binned_channels import bind_model, check_model


@pytest.mark.parametrize('mode',['positive','negative','conditional','conditional_negative'])
def test_cbshape_tensor_native_nll_derivatives_and_transport(root,tmp_path,mode):
    root=load_caching_nll();fixture=make_cbshape(root,negative_alpha='negative' in mode)
    pdf=fixture['pdf'];axes=[fixture['x']];coordinates={'cbx':np.linspace(-8.,9.,9)}
    if mode.startswith('conditional'):
        y=root.RooRealVar('cby','',.1,-1.,1.);axes.append(y)
        mean=root.RooFormulaVar('cb_conditional_mean','@0+0.2*@1',root.RooArgList(fixture['variables'][0],y))
        width=root.RooFormulaVar('cb_conditional_width','@0*(1+0.1*@1)',root.RooArgList(fixture['variables'][1],y))
        child=root.RooCBShape('cb_conditional','',fixture['x'],mean,width,*fixture['variables'][2:])
        marginal=root.RooUniform('cby_pdf','',root.RooArgSet(y))
        source=root.RooProdPdf('cb_conditional_source','',root.RooArgSet(marginal),
            root.RooFit.Conditional(root.RooArgSet(child),root.RooArgSet(fixture['x'])))
        pdf=root.SimpleProdPdf('cb_product','',source)
        coordinates['cby']=np.repeat([-.7,0.,.8],3)
    else:
        # Generic product caching uses the scalar ROOT evaluator. The distinct
        # direct VectorizedCBShape path is checked/rejected separately below.
        source=root.RooProdPdf('cb_source','',root.RooArgList(pdf))
        pdf=root.SimpleProdPdf('cb_product','',source)
    workspace=root.RooWorkspace('w');getattr(workspace,'import')(pdf)
    path=tmp_path/'cb.root';workspace.writeToFile(str(path));handle=root.TFile.Open(str(path));loaded=handle.Get('w')
    model=bind_model(root,loaded.pdf(pdf.GetName()),[loaded.var(x.GetName()) for x in axes],coordinates)
    check_model(model)
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
    handle.Close()


@pytest.mark.parametrize('negative',[False,True])
def test_direct_cbshape_cached_route_is_explicitly_unsupported(root,negative):
    fixture=make_cbshape(root,negative_alpha=negative)
    with pytest.raises(UnsupportedModelError,match='VectorizedCBShape.*not yet translated'):
        bind_model(root,fixture['pdf'],[fixture['x']],{'cbx':np.linspace(-8.,9.,9)})
