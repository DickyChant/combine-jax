"""Validate every saved device profile point against native CLI profile fits."""
import argparse
import json
from pathlib import Path

import numpy as np

from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.tensor.profile_point import configured_profile_model
from native_fit_model import load_native_fit_model
from tensor_benchmark_common import save_json,sha
from tensor_benchmark_dataset import benchmark_dataset
from tensor_benchmark_provenance import verified_source_record


STOPPING_RULE=dict(type='Minuit2',algorithm='Migrad',strategy=1,tolerance=.1,
                   error_definition=.5,raw_nll_edm_target=1e-4)


def compare_points(model,tensor,arrays,natives,checks,*,allow_uncertified_curvature=False):
    """Numerical/state gates; file and producer identities are checked by main."""
    info=model_parameter_info(model);names=info.metadata['names'];grid=np.asarray(tensor['grid'])
    parameter=tensor['parameter'];index=names.index(parameter)
    assert info.arrays['lower'][index]<info.arrays['upper'][index]
    assert grid.ndim==1 and len(grid)>0 and np.isfinite(grid).all()
    assert len(set(grid.tolist()))==len(grid), 'A profile grid must contain distinct physical points'
    count=len(grid);assert len(natives)==len(checks)==count
    assert tensor['status']=='measured_pending_validation' and tensor['float64'] and not tensor['root_imported']
    assert tensor['fit_state']==info.metadata['fit_policy']
    options=tensor['fit_options']
    assert options['initial']=='imported nominal' and options['stopping_criterion']=='edm' and options['edm_tolerance']==1e-4
    if allow_uncertified_curvature:
        assert options['edm_validation']=='estimated' and options['uncertainty_note']
    else:assert options['edm_validation']=='exact_hessian'
    np.testing.assert_array_equal(arrays['parameter_names'],names)
    np.testing.assert_array_equal(arrays['grid'],grid)
    assert arrays['values'].shape==(count,len(names)) and arrays['value'].shape==(count,)
    assert np.isfinite(arrays['values']).all() and np.isfinite(arrays['value']).all()
    np.testing.assert_array_equal(arrays['values'][:,index],grid)
    np.testing.assert_array_equal(arrays['status'],tensor['point_statuses'])
    assert tensor['all_converged'] and np.all(arrays['status']==0)
    edm=arrays['corrected_estimated_edm'];assert np.all(np.isfinite(edm)&(edm>=0)&(edm<1e-4))
    if not allow_uncertified_curvature:
        exact=arrays['hessian_edm']
        assert np.all(arrays['edm_curvature_validated']) and np.all(np.isfinite(exact)&(exact>=0)&(exact<1e-4))
    result=[]
    for i,(value,native,check) in enumerate(zip(grid,natives,checks,strict=True)):
        profile=configured_profile_model(model,{parameter:float(value)})
        expected=model_parameter_info(profile)
        assert native['status']=='measured_fit_pending_agreement' and native['fit_scope']=='migrad'
        assert native['root_version']=='6.40.00' and native['combine_version']=='11.0.0'
        assert native['stopping_rule']==STOPPING_RULE
        fit=native['fit'];assert fit['status']==0 and fit['edm_target_met'] and 0<=fit['edm']<1e-4
        assert fit['roofit_evaluation_errors']==0
        assert native['host']==tensor['host'] and native['dataset']==tensor['dataset']==check['dataset']
        assert native['profile_point']==profile.metadata['profile_point']==check['profile_point']
        assert native['preflight']['status']=='passed'
        assert native['fit_state']==native['preflight']['initial_state']==check['fit_state']==expected.metadata['fit_policy']
        np.testing.assert_array_equal(native['initial_parameters'],names)
        np.testing.assert_array_equal(native['initial_values'],expected.arrays['initial'])
        np.testing.assert_array_equal(fit['parameters'],names)
        assert check['profile_index']==i and check['status']=='passed_point_only'
        assert check['optimizer_status']==0
        np.testing.assert_array_equal(check['point'],arrays['values'][i])
        assert check['saved_tensor_nll']==arrays['value'][i]
        assert all(np.isfinite(e) and 0<=e<1e-6 for e in check['errors'].values())
        values=arrays['values'][i];reference=np.asarray(fit['values']);sigma=np.asarray(fit['errors'])
        assert reference.shape==sigma.shape==(len(names),)
        assert np.isfinite(reference).all() and np.isfinite(sigma).all()
        free=expected.arrays['lower']<expected.arrays['upper']
        assert np.all(sigma[free]>0)
        for point in (values,reference):
            assert np.all(point>=expected.arrays['lower']) and np.all(point<=expected.arrays['upper'])
            np.testing.assert_array_equal(point[~free],expected.arrays['initial'][~free])
        delta=abs(values-reference);limits=np.maximum(.01*abs(reference),.01*sigma)
        scaled=np.zeros(len(names));scaled[free]=delta[free]/sigma[free]
        difference=float(arrays['value'][i]-fit['canonical_nll'])
        parameter_passed=bool(np.all(delta[free]<=limits[free]))
        passed=parameter_passed and abs(difference)<=2e-4
        poi=[dict(name=name,tensor=float(values[j]),native=float(reference[j]),delta_over_sigma=float(scaled[j]))
             for j,name in enumerate(names) if name in ('r','r_hhvv') and free[j]]
        result.append(dict(index=i,fixed_value=float(value),passed=passed,
            same_point_nll_error=check['errors']['canonical_jax_absolute_error'],
            minimum_nll_difference=difference,minimum_nll_budget=2e-4,parameter_gate_passed=parameter_passed,
            max_parameter_delta_over_sigma=float(np.max(scaled,initial=0.)),
            failed_parameters=[name for j,name in enumerate(names) if free[j] and delta[j]>limits[j]],
            native_edm=float(fit['edm']),tensor_edm=float(edm[i]),poi=poi))
    return result


def add_baseline_shifts(rows,plan,tensor,arrays,natives):
    """Use one accepted native baseline for both profile curves and POI shifts."""
    assert tensor['host']==plan['host'] and tensor['dataset']==plan['dataset']
    assert tensor['model_sha256']==plan['model_sha256']
    assert tensor['parameter']==plan['parameter']
    np.testing.assert_array_equal(tensor['grid'],plan['grid'])
    np.testing.assert_array_equal(arrays['parameter_names'],plan['parameter_names'])
    index=plan['poi_index'];baseline=plan['baseline_values'][index];sigma=plan['baseline_errors'][index]
    assert np.isfinite(sigma) and sigma>0
    for i,(row,native) in enumerate(zip(rows,natives,strict=True)):
        assert native['profile_plan']=={**tensor['profile_plan'],'index':i}
        tensor_shift=float(arrays['values'][i,index]-baseline)
        native_shift=float(native['fit']['values'][index]-baseline)
        error=abs(tensor_shift-native_shift);limit=max(.01*abs(native_shift),.01*sigma)
        row.update(poi_shift=dict(name=plan['poi'],tensor=tensor_shift,native=native_shift,
            absolute_error=error,tolerance=limit,passed=error<=limit),
            tensor_delta_nll=float(arrays['value'][i]-plan['baseline_canonical_nll']),
            native_delta_nll=float(native['fit']['canonical_nll']-plan['baseline_canonical_nll']))
        row['passed']=row['passed'] and error<=limit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--tensor',type=Path,required=True)
    parser.add_argument('--native',type=Path,nargs='+',required=True)
    parser.add_argument('--native-point',type=Path,nargs='+',required=True)
    parser.add_argument('--source-revision')
    parser.add_argument('--allow-uncertified-curvature',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    case=json.loads(args.manifest.read_text())[args.case]
    model,_=load_native_fit_model(case['model'])
    dataset=benchmark_dataset(case,model,label=args.case)
    tensor=json.loads(args.tensor.read_text())
    assert tensor['dataset']==dataset
    assert tensor['model_sha256']=={name:sha(Path(case['model'])/name) for name in ('model.json','arrays.npz')}
    assert sha(tensor['arrays'])==tensor['arrays_sha256']
    natives=[json.loads(p.read_text()) for p in args.native]
    checks=[json.loads(p.read_text()) for p in args.native_point]
    evidence={'tensor':verified_source_record(tensor,'scripts/benchmark_tensor_profile.py',args.source_revision),
              'native':[],'point':[]}
    for native,check in zip(natives,checks,strict=True):
        assert native['manifest_sha256']==sha(args.manifest)==check['manifest_sha256']
        assert native['input_workspace_sha256']==sha(case['workspace'])==check['workspace_sha256']
        assert check['fit_sha256']==sha(args.tensor) and check['arrays_sha256']==tensor['arrays_sha256']
        evidence['native'].append(verified_source_record(native,'scripts/benchmark_combine_fit.py',args.source_revision))
        evidence['point'].append(verified_source_record(check,'scripts/check_tensor_fit_candidate.py',args.source_revision))
    with np.load(tensor['arrays'],allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
    rows=compare_points(model,tensor,arrays,natives,checks,allow_uncertified_curvature=args.allow_uncertified_curvature)
    plan=None
    if tensor.get('profile_plan'):
        from tensor_profile_plan import load_plan
        binding=tensor['profile_plan'];assert sha(binding['path'])==binding['sha256']
        plan=load_plan(binding['path']);assert plan['inputs']['case']==args.case
        assert plan['inputs']['manifest']==str(args.manifest)
        add_baseline_shifts(rows,plan,tensor,arrays,natives)
    else:assert all(native.get('profile_plan') is None for native in natives)
    report=dict(status='passed_all_profile_minima_no_speed_claim' if all(r['passed'] for r in rows) else 'failed_profile_agreement',
        script_sha256=sha(__file__),dataset=dataset,case=args.case,parameter=tensor['parameter'],grid=tensor['grid'],points=rows,
        artifacts={str(p):sha(p) for p in (args.manifest,args.tensor,*args.native,*args.native_point)},source_evidence=evidence,
        max_minimum_nll_difference=max(abs(r['minimum_nll_difference']) for r in rows),
        max_parameter_delta_over_sigma=max(r['max_parameter_delta_over_sigma'] for r in rows),
        covariance_certified=False,uncertified_curvature_allowed=args.allow_uncertified_curvature,
        baseline_linked_poi_shifts=plan is not None,
        scope='Every physical-grid minimum checked with the same nominal nuisance start and accepted EDM rule. Same-point NLL <1e-6; minimum NLL budget 2e-4; coordinates within max(1% native value,0.01 native sigma). With a shared plan, POI shifts use the same accepted native baseline and max(1% native shift,0.01 baseline sigma). No confidence endpoints, covariance or scan speedup are certified.')
    if plan:
        report['profile_plan']=tensor['profile_plan']
        report['artifacts'][tensor['profile_plan']['path']]=tensor['profile_plan']['sha256']
    save_json(args.output,report);print(report['status'],flush=True)
    assert all(r['passed'] for r in rows)


if __name__=='__main__':main()
