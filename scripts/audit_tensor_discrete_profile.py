"""Independently recompute profile stationarity, branch winners and native gates."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from tensor_benchmark_common import save_json, sha
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = json.loads(args.input.read_text())
    assert report['status']=='passed' and report['float64']
    assert report['script_sha256']==sha('scripts/validate_tensor_discrete_profile.py')
    assert report['workspace_sha256']==sha(report['workspace'])
    assert all(sha(p)==digest for p,digest in report['sources'].items())
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    rows=[];native_rows=[];device_rows=[]
    for branch in report['branches']:
        model=TensorBundle.load(branch['model'])
        assert branch['model_sha256']=={n:sha(Path(branch['model'])/n) for n in ('model.json','arrays.npz')}
        info=model_parameter_info(model)
        arrays=jax.device_put({n:a for n,a in model.arrays.items() if not n.startswith('parameter_info__')})
        function=jax.jit(jax.value_and_grad(make_model_function(model)))
        checks=[];native_values=[];device_values=[]
        assert len(branch['points'])==len(report['grid'])
        for point,poi in zip(branch['points'],report['grid'],strict=True):
            assert point['poi']==poi and point['passed'] and point['native_valid'] and point['device_status']==0
            theta=np.array(point['device_values'])
            assert np.all(theta>=info.arrays['lower']) and np.all(theta<=info.arrays['upper'])
            assert theta[branch['parameters'].index('r')]==poi
            np.testing.assert_allclose(theta,point['native_values'],atol=1e-4,rtol=1e-4)
            value,gradient=map(np.asarray,function(theta,arrays))
            coordinates=fit_coordinates(info,fixed={'r':poi})
            free=coordinates['free_indices'];scale=coordinates['scale']
            z=(theta[free]-coordinates['reference'][free])/scale
            scaled=gradient[free]*scale
            eps=32*np.finfo(float).eps*(1+np.abs(z))
            blocked=((z<=coordinates['lower']+eps)&(scaled>0))|((z>=coordinates['upper']-eps)&(scaled<0))
            projected=np.where(blocked,0.,scaled)
            norm=float(np.max(np.abs(projected),initial=0.))
            error=abs(float(value)-point['native_nll'])
            assert norm<=1e-7 and error<1e-6 and abs(float(value)-point['native_at_device_nll'])<1e-6
            checks.append(dict(poi=poi,projected_gradient_max=norm,native_nll_absolute_error=error,
                               native_parameter_max_absolute_difference=float(np.max(np.abs(theta-point['native_values'])))))
            native_values.append(point['native_nll']);device_values.append(float(value))
        rows.append(dict(branch=branch['label'],checks=checks));native_rows.append(native_values);device_rows.append(device_values)
        jax.clear_caches()
    native=np.array(native_rows);device=np.array(device_rows)
    assert native.shape[0]==2
    winner=np.argmin(native,axis=0)
    np.testing.assert_array_equal(winner,np.argmin(device,axis=0))
    np.testing.assert_array_equal(winner,report['envelope']['native_branch'])
    np.testing.assert_allclose(device.min(axis=0),report['envelope']['device_nll'],atol=1e-10,rtol=0)
    assert len(set(winner))==2,'This audit requires an observed branch switch'
    switches=[dict(left=report['grid'][i-1],right=report['grid'][i],
                   from_branch=int(winner[i-1]),to_branch=int(winner[i]))
              for i in range(1,len(winner)) if winner[i]!=winner[i-1]]
    result=dict(status='passed',script_sha256=sha(__file__),input=str(args.input),input_sha256=sha(args.input),
        branches=rows,grid_points=len(winner),branch_switch_intervals=switches,
        maximum_branch_nll_absolute_error=float(np.max(np.abs(device-native))),
        maximum_envelope_nll_absolute_error=float(np.max(np.abs(device.min(axis=0)-native.min(axis=0)))),
        gradient_tolerance=1e-7,
        scope='Independent NLL/gradient recomputation at every fitted branch point; native parameters, envelope and a change of winning branch agree. Switch location is bracketed by grid points; confidence endpoints and HHVV scan remain unvalidated.')
    save_json(args.output,result);print('passed',switches,result['maximum_envelope_nll_absolute_error'],flush=True)


if __name__=='__main__':
    main()
