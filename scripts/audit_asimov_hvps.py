"""Audit selected nominal Asimov HVPs against an explicit saved AD Hessian.

Native NLL/gradient gates are replayed independently. Products certify only the
supplied directions, never the full matrix, a native covariance, or a fit.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from audit_asimov_hessian import checked_capture, native_reference, nominal_errors
from compare_local_cpu_hessian import TOLERANCE
from native_fit_model import load_native_fit_model
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import benchmark_dataset
from tensor_benchmark_provenance import verified_source_record
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates


def checked_hvps(path, directions_path, info, hashes, structure, revision=None):
    report=json.loads(Path(path).read_text())
    assert report['status']=='measured_pending_validation' and report['stage']=='complete'
    assert report['operation']=='hvp' and report['float64'] and not report['root_imported']
    assert report['backend'] in ('cpu','gpu') and report['mode']=='jit' and report['batch_size']==1
    assert report['model_sha256']==hashes and report['structure_key']==structure
    source=verified_source_record(report,'scripts/benchmark_tensor.py',revision)
    assert report['points_source']=='model initial' and report['points_source_sha256'] is None
    assert sha(directions_path)==report['directions_source_sha256']
    assert sha(report['arrays'])==report['arrays_sha256']
    with np.load(directions_path,allow_pickle=False) as saved:
        directions=saved['directions']
        np.testing.assert_array_equal(saved['parameter_names'],info.metadata['names'])
    with np.load(report['arrays'],allow_pickle=False) as saved:
        actual={key:saved[key] for key in ('parameter_names','points','values','gradients','directions','hvps')}
    n=len(info.metadata['names']);d=report['directions']
    assert n==report['parameters'] and report['points']==1 and d>0
    np.testing.assert_array_equal(actual['parameter_names'],info.metadata['names'])
    np.testing.assert_array_equal(actual['points'],info.arrays['initial'][None,:])
    np.testing.assert_array_equal(actual['directions'],directions)
    for key,shape in (('points',(1,n)),('values',(d,)),('gradients',(d,n)),('directions',(d,n)),('hvps',(d,n))):
        assert actual[key].shape==shape and actual[key].dtype==np.float64 and np.isfinite(actual[key]).all()
    fixed=info.arrays['lower']==info.arrays['upper']
    np.testing.assert_array_equal(actual['gradients'][:,fixed],0.)
    np.testing.assert_array_equal(actual['hvps'][:,fixed],0.)
    fixed_only=np.all(actual['directions'][:,~fixed]==0.,axis=1)
    np.testing.assert_array_equal(actual['hvps'][fixed_only],0.)
    return report,actual,source


def product_errors(actual, reference, info):
    coordinates=fit_coordinates(info);free=coordinates['free_indices'];scales=coordinates['scale']
    # With ||v/scale||_1 <= 1, the scaled product error is bounded by the
    # maximum entrywise scaled Hessian error. Keep the existing 1e-6 threshold.
    norms=np.sum(np.abs(actual['directions'][:,free]/scales),axis=1)
    assert np.all(norms<=1.+1e-12), 'Directions exceed the scaled L1 validation budget'
    expected=actual['directions']@reference['hessians'][0].T
    difference=np.abs(actual['hvps']-expected)
    scaled=difference[:,free]*scales
    worst=np.unravel_index(np.argmax(scaled),scaled.shape)
    return dict(passed=bool(scaled.max(initial=0.)<TOLERANCE),
        compared_products=len(expected),compared_free_entries=int(scaled.size),
        maximum_absolute_error=float(difference.max(initial=0.)),
        maximum_scaled_absolute_error=float(scaled.max(initial=0.)),scaled_direction_l1=norms.tolist(),
        per_direction_maximum_scaled_absolute_error=scaled.max(axis=1).tolist(),
        worst_scaled_entry=dict(direction=int(worst[0]),parameter=info.metadata['names'][int(free[worst[1]])]),
        scope='Selected products only; no full-matrix agreement claim.')


def compare(*,manifest,benchmark,directions,hessian,native_gate,source_revision=None,hessian_revision=None):
    case=json.loads(Path(manifest).read_text())['hhvv']
    if case.get('dataset')!='asimov_prefit':
        raise ValueError('HVP audit requires nominal pre-fit Asimov input')
    model,loader=load_native_fit_model(case['model']);benchmark_dataset(case,model,label='hhvv')
    info=model_parameter_info(model)
    assert all(sha(path)==checksum for path,checksum in case['hashes'].items())
    hashes={n:sha(Path(case['model'])/n) for n in ('model.json','arrays.npz')}
    structure=json.loads((Path(case['model'])/'model.json').read_text())['structure_key']
    references,value,native_evidence=native_reference(native_gate,case,info,hashes)
    report,actual,source=checked_hvps(benchmark,directions,info,hashes,structure,source_revision)
    baseline,reference,baseline_source=checked_capture(hessian,info,hashes,structure,hessian_revision)
    native=[nominal_errors(dict(values=actual['values'][i:i+1],gradients=actual['gradients'][i:i+1]),references,value)
            for i in range(len(actual['hvps']))]
    native_baseline=nominal_errors(reference,references,value)
    errors=product_errors(actual,reference,info)
    passed=errors['passed'] and native_baseline['passed'] and all(r['passed'] for r in native)
    return dict(status='passed_selected_products_and_nominal_native_gates' if passed else 'failed',
        script_sha256=sha(__file__),dataset='asimov_prefit',dataset_identity=model.metadata['provenance']['dataset']['identity'],
        workspace_sha256=sha(case['workspace']),model_sha256=hashes,native_model_loader=loader,
        artifacts={str(p):sha(p) for p in (manifest,benchmark,directions,hessian,native_gate)},
        arrays_sha256=dict(benchmark=report['arrays_sha256'],hessian=baseline['arrays_sha256']),
        source_evidence=dict(benchmark=source,hessian=baseline_source),native_reference=native_evidence,
        nominal_native_comparison=dict(benchmark_directions=native,hessian=native_baseline),
        selected_products=errors,scaled_absolute_tolerance=TOLERANCE,scale_policy=fit_coordinates(info)['scale_policy'],
        exact_fixed_zeros=True,covariance_certified=False,
        measurement={k:report.get(k) for k in ('host','backend','jax_version','jaxlib_version','devices','environment',
            'compiler_memory','timings','resources','jax_allocator')},
        scope='Nominal-only selected HVP comparison to a saved exact AD Hessian, plus unchanged native NLL/gradient gates for every capture. No full matrix, fit, native curvature certificate or covariance claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','benchmark','directions','hessian','native-gate','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--source-revision');parser.add_argument('--hessian-revision')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Preserve prior audit evidence')
    result=compare(**{k:v for k,v in vars(args).items() if k!='output'})
    save_json(args.output,result)
    print(result['status'],result['selected_products'])
    raise SystemExit(0 if result['status']=='passed_selected_products_and_nominal_native_gates' else 1)


if __name__=='__main__':main()
