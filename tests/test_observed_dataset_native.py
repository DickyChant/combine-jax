"""Observed-style data: native-v10 NLLs and derivative references, no analysis input."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest

from combine_jax.testing.observed_datasets import workspace
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.model import from_workspace,make_model_function


@pytest.mark.parametrize('kind',['unweighted_unbinned','weighted_binned','empty_bins','mixed'])
def test_observed_data_native_values_gradient_hessian(kind):
    root=load_caching_nll()
    reference=json.loads((Path(__file__).parent/'data/observed_dataset_v10.json').read_text())
    case=next(x for x in reference['cases'] if x['fixture']['kind']==kind)
    w,description=workspace(root,kind)
    assert description==case['fixture']
    model=from_workspace(w,data='data_obs',analysis='synthetic',dataset_kind='observed')
    names=model.metadata['parameter_names']
    assert set(names)=={'mu','theta'}
    assert model.metadata['provenance']['dataset_selection']['observed_data_accessed']
    arrays={k:v for k,v in model.arrays.items() if not k.startswith('parameter_info__')}
    f=make_model_function(model,binned_gradient='structural')
    vg=jax.jit(jax.value_and_grad(f));hessian=jax.jit(jax.hessian(f))
    order=[case['parameter_names'].index(n) for n in names]
    errors=[]
    for point in case['points']:
        x=np.asarray(point['point'])[order]
        value,gradient=vg(x,arrays);h=hessian(x,arrays)
        np.testing.assert_allclose(value,point['nll'],rtol=0,atol=1e-6)
        np.testing.assert_allclose(gradient,np.asarray(point['gradient'])[order],rtol=0,atol=1e-7)
        np.testing.assert_allclose(h,np.asarray(point['hessian'])[np.ix_(order,order)],rtol=0,atol=1e-6)
        errors.append(dict(nll=abs(float(value)-point['nll']),gradient=float(np.max(abs(np.asarray(gradient)-np.asarray(point['gradient'])[order]))),
            hessian=float(np.max(abs(np.asarray(h)-np.asarray(point['hessian'])[np.ix_(order,order)])))))
    print('OBSERVED_NATIVE_ERRORS',kind,json.dumps(errors),flush=True)
