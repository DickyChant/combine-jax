"""Regression cases where objective roundoff must not replace gradient gates."""
import json
from pathlib import Path

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.fitting import prepare_fit
from combine_jax.root_io import open_workspace
from test_tensor_compilation import poisson_model
from test_discrete_masks import multipdf_path


def test_poisson_fit_at_strict_gradient_target_with_roundoff():
    model=poisson_model(8);arrays=dict(model.arrays);arrays['offset']=arrays['offset']+.25
    model=TensorBundle(model.metadata,arrays)
    summaries=[]
    for multiplier in (0.,8.):
        fn,c,a=prepare_fit(model,gradient_tolerance=1e-8,armijo_roundoff_multiplier=multiplier)
        result=jax.device_get(fn(c,a));norm=float(np.max(np.abs(result['projected_gradient'])))
        summaries.append(dict(multiplier=multiplier,status=int(result['status']),projected_gradient_max=norm,
            roundoff_accepts=int(result['roundoff_accepts']),maximum_roundoff_allowance=float(result['maximum_roundoff_allowance'])))
        if multiplier==0.:
            assert result['status']==2 and norm>1e-8
        else:
            assert result['status']==0 and norm<=1e-8
            assert result['roundoff_accepts']>0 and result['maximum_roundoff_allowance']<1e-13
            np.testing.assert_allclose(result['values'],[10.],atol=2e-8,rtol=0.)
    print('LINE_SEARCH_POISSON_JSON='+json.dumps(summaries,allow_nan=False))


def test_bernstein_profile_preserves_bound_and_converges(multipdf_path):
    r=json.loads(Path('reports/tensor_analytic_reported_error_fits.json').read_bytes())['cases'][-1]
    failed=r['profiles'][-1];assert not failed['passed'] and failed['device_status']==2
    with open_workspace(multipdf_path) as workspace:
        workspace.cat('pdf_index').setIndex(1)
        for n,e in [('multi_slope',.005),('multi_c1',.1),('multi_c2',.1)]:workspace.var(n).setError(e)
        model=from_workspace(workspace,multipdf_path.parent/'card.txt')
    fn,c,a=prepare_fit(model,initial=r['device']['values'],fixed={'r':failed['r']},gradient_tolerance=1e-7)
    result=jax.device_get(fn(c,a));norm=float(np.max(np.abs(result['projected_gradient'])))
    # Reduced-space curvature can now reach this minimum using strict Armijo
    # decreases alone. The Poisson case above still exercises the allowance.
    assert result['status']==0 and norm<=1e-7
    assert result['maximum_roundoff_allowance']<2e-12
    np.testing.assert_allclose(result['values'],failed['native_values'],atol=1e-4,rtol=1e-4)
    assert abs(float(result['value'])-failed['native_nll'])<1e-6
    i=model.metadata['parameter_names'].index('multi_c1')
    assert result['values'][i]==.2 and result['reported_error_status'][i]==2
    print('LINE_SEARCH_PROFILE_JSON='+json.dumps(dict(status=int(result['status']),projected_gradient_max=norm,
        roundoff_accepts=int(result['roundoff_accepts']),maximum_roundoff_allowance=float(result['maximum_roundoff_allowance']),
        nll_absolute_error=abs(float(result['value'])-failed['native_nll'])),allow_nan=False))
