import subprocess
import sys

import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from test_landau import make_landau
from test_tensor_binned_channels import bind_model,check_model


@pytest.mark.parametrize('conditional',[False,True])
def test_landau_tensor_native_nll_derivatives_and_transport(root,tmp_path,conditional):
    root=load_caching_nll();f=make_landau(root,(-3.,20.));pdf=f['pdf'];axes=[f['x']]
    coordinates={'landau_x':np.linspace(-1.5,18.,9)}
    if conditional:
        y=root.RooRealVar('landau_y','',.1,-1.,1.);axes.append(y)
        mean=root.RooFormulaVar('landau_conditional_mean','@0+0.2*@1',root.RooArgList(f['variables'][0],y))
        child=root.RooLandau('landau_conditional','',f['x'],mean,f['variables'][1])
        marginal=root.RooUniform('landau_y_pdf','',root.RooArgSet(y))
        pdf=root.RooProdPdf('landau_product','',root.RooArgSet(marginal),
            root.RooFit.Conditional(root.RooArgSet(child),root.RooArgSet(f['x'])))
        coordinates['landau_y']=np.repeat([-.7,0.,.8],3)
    model=bind_model(root,pdf,axes,coordinates);check_model(model)
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
