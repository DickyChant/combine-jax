"""Validate the device scan API against the saved native discrete-profile grid."""
import argparse
import json
from pathlib import Path

import jax
import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from tensor_benchmark_provenance import verified_source_record
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.profiling import prepare_profile_scan
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    source=Path('reports/tensor_discrete_profile_switch.json')
    native=json.loads(source.read_text());assert native['status']=='passed'
    source_evidence=verified_source_record(native,'scripts/validate_tensor_discrete_profile.py','02e2c76')
    grid=np.array(native['grid'])
    report=dict(provenance(),status='checking',script_sha256=sha(__file__),float64=True,
        jax_version=jax.__version__,native_reference=str(source),native_reference_sha256=sha(source),
        native_source_evidence=source_evidence,grid=grid,branches=[],
        scope='One device scan per branch, checked against native Combine/iminuit profile minima on a common 12-point grid. Independent gradient recomputation verifies stationarity. No GPU, confidence-endpoint or performance acceptance.')
    monitor=Resources(False,args.output)
    def save():
        report['resources']=monitor.snapshot();save_json(args.output,report)
    save()
    try:
        for branch in native['branches']:
            model=TensorBundle.load(branch['model']);info=model_parameter_info(model)
            assert branch['model_sha256']=={n:sha(Path(branch['model'])/n) for n in ('model.json','arrays.npz')}
            function,c,arrays=prepare_profile_scan(model,'r',grid,gradient_tolerance=1e-7)
            result=jax.device_get(function(c,arrays))
            numeric=args.output.with_name(args.output.stem+'_'+branch['label']+'.npz')
            assert not numeric.exists();np.savez_compressed(numeric,grid=grid,parameter_names=np.array(info.metadata['names']),**result)
            value_gradient=jax.jit(jax.value_and_grad(make_model_function(model)))
            row=dict(label=branch['label'],model=branch['model'],model_sha256=branch['model_sha256'],
                arrays=str(numeric),arrays_sha256=sha(numeric),points=[])
            report['branches'].append(row);save()
            for i,point in enumerate(branch['points']):
                assert result['status'][i]==0 and point['native_valid'] and point['passed']
                values=result['values'][i]
                assert values[info.metadata['names'].index('r')]==grid[i]
                np.testing.assert_allclose(values,point['native_values'],atol=1e-4,rtol=1e-4)
                nll,g=map(np.asarray,value_gradient(values,arrays))
                coordinates=fit_coordinates(info,fixed={'r':grid[i]})
                free=coordinates['free_indices'];scaled=g[free]*coordinates['scale']
                z=(values[free]-coordinates['reference'][free])/coordinates['scale']
                eps=32*np.finfo(float).eps*(1+np.abs(z))
                blocked=((z<=coordinates['lower']+eps)&(scaled>0))|((z>=coordinates['upper']-eps)&(scaled<0))
                projected=np.where(blocked,0.,scaled)
                gradient_max=float(np.max(np.abs(projected),initial=0.))
                error=abs(float(nll)-point['native_nll'])
                assert gradient_max<=1e-7 and error<1e-6 and abs(float(nll)-result['value'][i])<1e-10
                np.testing.assert_allclose(projected,result['projected_gradient'][i],atol=2e-10,rtol=1e-5)
                row['points'].append(dict(poi=float(grid[i]),nll=float(nll),native_nll=point['native_nll'],
                    nll_absolute_error=error,recomputed_projected_gradient_max=gradient_max,
                    iterations=int(result['iteration'][i]),status=int(result['status'][i])))
            save();jax.clear_caches();print('passed',branch['label'],max(p['nll_absolute_error'] for p in row['points']),flush=True)
        calculated=np.array([[p['nll'] for p in b['points']] for b in report['branches']])
        reference=np.array(native['envelope']['native_nll'])
        np.testing.assert_array_equal(np.argmin(calculated,axis=0),native['envelope']['native_branch'])
        report.update(status='passed',maximum_envelope_nll_absolute_error=float(np.max(np.abs(calculated.min(axis=0)-reference))),
                      selected_branch=np.argmin(calculated,axis=0))
        assert report['maximum_envelope_nll_absolute_error']<1e-6
        assert all(sha(p)==digest for p,digest in report['sources'].items());save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        monitor.close()


if __name__=='__main__':
    main()
