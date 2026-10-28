"""Fresh-process executable-cache probe; numerical results, not speed claims."""
import argparse
import json
from pathlib import Path
import sys

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.compilation import configure_model_cache
from combine_jax.tensor.model import make_model_function,evaluate_model_numpy
from combine_jax.tensor.fitting import prepare_fit


def main():
    parser=argparse.ArgumentParser();parser.add_argument('model');parser.add_argument('cache');parser.add_argument('scale',type=float)
    args=parser.parse_args();assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    model=TensorBundle.load(args.model)
    arrays=dict(model.arrays)
    arrays['channels__observed']=arrays['channels__observed']*args.scale
    arrays['offset']=arrays['offset']+.25*args.scale
    model=TensorBundle(model.metadata,arrays)
    config=configure_model_cache(model,args.cache)
    point=model.arrays['scalars__parameter_initial']
    nll=jax.jit(make_model_function(model))
    value=float(nll(point,jax.device_put(dict(model.arrays))))
    expected=float(evaluate_model_numpy(model,point))
    assert abs(value-expected)<1e-12
    tolerance=1e-7  # Same optimizer target as the accepted native-card gates.
    fit,c,a=prepare_fit(model,gradient_tolerance=tolerance)
    result=jax.device_get(fit(c,a))
    assert result['status']==0 and result['covariance_status']==0
    observed=arrays['channels__observed'];optimum=np.mean(observed)
    np.testing.assert_allclose(result['values'],[optimum],atol=2*tolerance*optimum/len(observed),rtol=0.)
    np.testing.assert_allclose(result['errors'],[np.sqrt(optimum/len(observed))],atol=1e-7,rtol=0.)
    assert 'ROOT' not in sys.modules
    print(json.dumps(dict(status='passed',cache=config,scale=args.scale,nll=value,numpy_nll=expected,
        values=result['values'].tolist(),errors=result['errors'].tolist(),channels=len(observed),
        gradient_tolerance=tolerance,projected_gradient_max=float(np.max(np.abs(result['projected_gradient']))),
        cache_entries=sorted(p.name for p in Path(config['directory']).glob('*-cache'))),allow_nan=False),flush=True)


if __name__=='__main__':main()
