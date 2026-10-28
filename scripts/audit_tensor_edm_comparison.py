"""Compare saved EDM fits without conflating same-point and minimum agreement."""
import argparse
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case',required=True)
    parser.add_argument('--manifest',type=Path,default=Path('build/tensor_benchmark_inputs/manifest.json'))
    parser.add_argument('--tensor',type=Path,required=True)
    parser.add_argument('--native',type=Path,required=True)
    parser.add_argument('--native-point',type=Path,required=True)
    parser.add_argument('--source-revision')
    parser.add_argument('--allow-uncertified-curvature',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); assert not args.output.exists()
    result=compare(args.tensor,args.native,args.native_point,manifest=args.manifest,case=args.case,
        source_revision=args.source_revision,allow_uncertified_curvature=args.allow_uncertified_curvature)
    save_json(args.output,result)
    print(args.case,result['status'],result['minimum_nll_difference'],result['max_parameter_delta_over_sigma'])
    assert result['status']=='passed_same_minimum_gates_no_speed_claim'


def compare(tensor,native,native_point,*,manifest,case,source_revision=None,allow_uncertified_curvature=False):
    args=SimpleNamespace(tensor=Path(tensor),native=Path(native),native_point=Path(native_point),
        manifest=Path(manifest),case=case,source_revision=source_revision,
        allow_uncertified_curvature=allow_uncertified_curvature)
    from native_fit_model import load_native_fit_model
    from combine_jax.tensor.parameter_info import model_parameter_info
    from tensor_benchmark_dataset import benchmark_dataset
    manifest=args.manifest
    case=json.loads(manifest.read_text())[args.case]
    model,_=load_native_fit_model(case['model']);info=model_parameter_info(model)
    dataset_name=benchmark_dataset(case,model,label=args.case)
    tensor,native,point=(json.loads(p.read_text()) for p in (args.tensor,args.native,args.native_point))
    assert tensor['status']=='measured_pending_validation' and tensor['operation']=='fit'
    assert native['status']=='measured_fit_pending_agreement' and native['fit_scope']=='migrad'
    assert point['status']=='passed_point_only' and point['fit_sha256']==sha(args.tensor)
    source={name:verified_source_record(r,script,args.source_revision) for name,r,script in (
        ('tensor',tensor,'scripts/benchmark_tensor.py'),('native',native,'scripts/benchmark_combine_fit.py'),
        ('point',point,'scripts/check_tensor_fit_candidate.py'))}
    assert tensor['host']==native['host']
    assert tensor['float64'] and native['root_version']=='6.40.00' and native['combine_version']=='11.0.0'
    assert native['stopping_rule']==dict(type='Minuit2',algorithm='Migrad',strategy=1,tolerance=.1,error_definition=.5,raw_nll_edm_target=1e-4)
    assert tensor['fit_options']['stopping_criterion']=='edm' and tensor['fit_options']['edm_tolerance']==1e-4
    assert native['fit']['status']==0 and native['fit']['edm_target_met']
    assert np.isfinite(native['fit']['edm']) and 0<=native['fit']['edm']<1e-4
    assert native['fit']['roofit_evaluation_errors']==0
    assert sha(tensor['arrays'])==tensor['arrays_sha256']
    assert point['arrays_sha256']==tensor['arrays_sha256']
    assert all(np.isfinite(error) and 0<=error<1e-6 for error in point['errors'].values())
    assert native['manifest_sha256']==sha(manifest)==point['manifest_sha256']
    assert info.metadata.get('fit_policy'), 'A native fit-policy import is required'
    assert native['preflight']['status']=='passed'
    assert native['preflight']['initial_state']==native['fit_state']==point['fit_state']==info.metadata['fit_policy']
    assert tensor['model_sha256']=={name:sha(Path(case['model'])/name) for name in ('model.json','arrays.npz')}
    assert native['input_workspace_sha256']==sha(case['workspace'])==point['workspace_sha256']
    np.testing.assert_array_equal(native['initial_values'],info.arrays['initial'])
    assert tensor['fit_options']['initial']=='imported nominal' and native['dataset']==dataset_name==point['dataset']
    with np.load(tensor['arrays'],allow_pickle=False) as saved:
        names=saved['parameter_names'];values=saved['fit_values'];sigma=np.array(native['fit']['errors'])
        np.testing.assert_array_equal(names,native['initial_parameters'])
        np.testing.assert_array_equal(names,native['fit']['parameters'])
        np.testing.assert_array_equal(values,point['point'])
        assert saved['fit_status']==0 and 0<=saved['fit_corrected_estimated_edm']<1e-4
        if args.allow_uncertified_curvature:
            assert tensor['fit_options']['edm_validation']=='estimated' and tensor['fit_options']['uncertainty_note']
        else:
            assert saved['fit_edm_curvature_validated'] and 0<=saved['fit_hessian_edm']<1e-4
        free=info.arrays['lower']<info.arrays['upper']
        assert np.all(sigma[free]>0) and np.isfinite(sigma).all()
        reference=np.array(native['fit']['values'])
        assert np.isfinite(reference).all() and np.isfinite(values).all()
        for fitted in (reference,values):
            assert np.all(fitted>=info.arrays['lower']) and np.all(fitted<=info.arrays['upper'])
            np.testing.assert_array_equal(fitted[~free],info.arrays['initial'][~free])
        delta=values-reference
        scaled=np.zeros(len(names));scaled[free]=abs(delta[free])/sigma[free]
        tolerance=np.maximum(.01*abs(reference),.01*sigma)
        passed=np.all(abs(delta[free])<=tolerance[free])
        nll_delta=float(saved['fit_value']-native['fit']['canonical_nll'])
        # This is the sum of the two declared stopping budgets. It is distinct
        # from the unchanged 1e-6 same-point NLL implementation gate.
        nll_budget=2e-4
        minimum_passed=bool(passed and abs(nll_delta)<=nll_budget)
        order=np.argsort(scaled)[::-1]
        rows=[dict(name=str(names[i]),tensor=float(values[i]),native=float(reference[i]),
                   native_sigma=float(sigma[i]),delta_over_sigma=float(scaled[i]),absolute_delta=float(abs(delta[i])))
              for i in order if free[i]]
        result=dict(status=('passed_same_minimum_gates_no_speed_claim' if minimum_passed else 'failed_minimum_agreement'),script_sha256=sha(__file__),case=args.case,
            artifacts={str(p):sha(p) for p in (args.tensor,args.native,args.native_point)},source_evidence=source,
            dataset=dataset_name,same_host=tensor['host'],same_start=True,same_workspace_dataset=True,same_fit_state=True,
            same_point_nll_error=point['errors']['canonical_jax_absolute_error'],minimum_nll_difference=nll_delta,
            minimum_nll_budget=nll_budget,parameter_gate='max(1% of native value, 0.01 native sigma)',
            parameter_gate_passed=bool(passed),minimum_nll_gate_passed=abs(nll_delta)<=nll_budget,
            max_parameter_delta_over_sigma=float(max(scaled)),poi=[r for r in rows if r['name'] in ('r','r_hhvv')],
            leading_parameters=rows[:20],tensor_edm=float(saved['fit_corrected_estimated_edm']),
            tensor_exact_hessian_edm=float(saved['fit_hessian_edm']),native_edm=native['fit']['edm'],
            tensor_edm_refinements=int(saved['fit_edm_refinements']),
            curvature_edm_validated=bool(saved['fit_edm_curvature_validated']),
            covariance_certified=False,uncertified_curvature_allowed=args.allow_uncertified_curvature,
            tensor_evaluation_counts=tensor['evaluation_counts'],native_evaluation_counts=native['evaluation_counts'],
            scope='Numerical agreement only; timing eligibility is audited separately. Native call instrumentation, when present, supplies numerical-gradient invocation counts. This does not certify HHVV covariance at accepted clipping surfaces.')
    return result


if __name__=='__main__': main()
