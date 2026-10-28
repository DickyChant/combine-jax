"""Compare bounded-error reporting with the installed native ROOT function."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.reported_errors import minuit_reported_errors


@pytest.fixture(scope='module')
def native_error():
    from combine_jax.root_io import load_root
    root=load_root();assert root.gSystem.Load('libMinuit2')>=0
    assert root.gInterpreter.Declare(r'''
    #include "Minuit2/MnUserTransformation.h"
    #include <cmath>
    double tensor_reported_error_reference(double value,double sigma,double lower,double upper) {
      ROOT::Minuit2::MnUserTransformation t;
      t.Add("x",value,sigma);
      if (std::isfinite(lower) && std::isfinite(upper)) t.SetLimits(0,lower,upper);
      else if (std::isfinite(lower)) t.SetLowerLimit(0,lower);
      else if (std::isfinite(upper)) t.SetUpperLimit(0,upper);
      double q=t.Ext2int(0,value);
      return t.Int2extError(0,q,sigma/std::abs(t.DInt2Ext(0,q)));
    }
    ''')
    return root.tensor_reported_error_reference


@pytest.mark.parametrize('kind',['unbounded','lower','upper','double'])
def test_native_transformation_runtime_arrays_and_large_steps(native_error,kind):
    rng=np.random.default_rng(4420);n=96
    lo=rng.uniform(-8.,-2.,n) if kind in ('lower','double') else np.full(n,-np.inf)
    hi=rng.uniform(2.,8.,n) if kind in ('upper','double') else np.full(n,np.inf)
    if kind=='double':x=lo+(hi-lo)*rng.uniform(1e-4,1-1e-4,n)
    elif kind=='lower':x=lo+10**rng.uniform(-4.,2.,n)
    elif kind=='upper':x=hi-10**rng.uniform(-4.,2.,n)
    else:x=rng.uniform(-10.,10.,n)
    sigma=10**rng.uniform(-3.,1.,n)
    reference=np.array([native_error(*args) for args in zip(x,sigma,lo,hi,strict=True)])
    fn=lambda *args:minuit_reported_errors(*args,xp=jnp)
    for error,status in (minuit_reported_errors(x,sigma,lo,hi),jax.device_get(jax.jit(fn)(x,sigma,lo,hi))):
        np.testing.assert_array_equal(status,0)
        np.testing.assert_allclose(error,reference,rtol=2e-10,atol=3e-13)
    assert not jax.make_jaxpr(fn)(x,sigma,lo,hi).consts


def test_fixed_invalid_and_numerically_unresolved_boundaries():
    x=np.array([0.,0.,1e-12,1.,.5,np.nan]);sigma=np.array([0.,1.,1.,-1.,np.nan,1.])
    lo=np.array([0.,0.,0.,-np.inf,0.,0.]);hi=np.array([0.,1.,np.inf,np.inf,1.,1.])
    for xp in (np,jnp):
        error,status=minuit_reported_errors(x,sigma,lo,hi,xp=xp)
        np.testing.assert_array_equal(status,[1,2,2,3,3,3]);assert error[0]==0. and np.isnan(error[1:]).all()
    for bad in (x.astype(np.float32),x[None,:]):
        with pytest.raises(ValueError):minuit_reported_errors(bad,sigma,lo,hi)


def test_saved_native_fits_keep_covariance_errors_and_reproduce_reported_errors(native_error):
    report=json.loads(Path('reports/tensor_analytic_fits.json').read_bytes());summaries=[]
    for case in report['cases']:
        lo=np.array([p['minimum'] for p in case['parameters']]);hi=np.array([p['maximum'] for p in case['parameters']])
        native=case['native'];device=case['device'];sigma=np.array(device['errors'])
        covariance=np.array(device['covariance']);np.testing.assert_allclose(sigma,np.sqrt(np.diag(covariance)),atol=1e-14)
        expected=np.array(native['errors']);nc=np.sqrt(np.diag(native['covariance']))
        reconstruction,status=minuit_reported_errors(np.array(native['values']),nc,lo,hi)
        np.testing.assert_array_equal(status,0);np.testing.assert_allclose(reconstruction,expected,rtol=2e-12,atol=1e-13)
        actual,status=jax.device_get(jax.jit(lambda *a:minuit_reported_errors(*a,xp=jnp))(np.array(device['values']),sigma,lo,hi))
        np.testing.assert_array_equal(status,0);np.testing.assert_allclose(actual,expected,rtol=.01,atol=0.)
        direct=np.array([native_error(*args) for args in zip(device['values'],sigma,lo,hi,strict=True)])
        np.testing.assert_allclose(actual,direct,rtol=2e-12,atol=1e-13)
        summaries.append(dict(label=case['label'],reported_errors=actual.tolist(),native_reported_errors=expected.tolist(),
            curvature_errors=sigma.tolist(),max_reported_error_relative_difference=float(np.max(np.abs(actual-expected)/expected)),
            max_native_reconstruction_relative_difference=float(np.max(np.abs(reconstruction-expected)/expected))))
    print('REPORTED_ERROR_JSON='+json.dumps(summaries,allow_nan=False))
