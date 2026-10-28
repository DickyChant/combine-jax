"""Apply unchanged nominal/random native gradient gates to tensor captures."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_gpu_results import verified_refinement
from gpu_gradient_evidence import random_validation


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',default='reports/tensor_hhvv_gradients_capture.json')
    parser.add_argument('--output',default='reports/tensor_hhvv_gradients.json')
    args=parser.parse_args();path=Path(args.capture);output=Path(args.output);assert not output.exists()
    capture=json.loads(path.read_bytes());assert capture['status']=='captured' and capture['captured_points']==26
    assert capture['float64'] and capture['closed_over_arrays']==0
    assert sha('scripts/capture_tensor_hhvv_gradients.py')==capture['script_sha256']
    for p,digest in capture['tensor_sources'].items():assert sha(p)==digest
    for key in ('values_report','import_report','old_checkpoint','old_comparison','arrays'):
        assert sha(capture[key])==capture[key+'_sha256']
    imp=json.loads(Path(capture['import_report']).read_bytes())
    for p,digest in imp['artifacts'].items():assert sha(p)==digest
    values=json.loads(Path(capture['values_report']).read_bytes())
    old=json.loads(Path(capture['old_checkpoint']).read_bytes());assert sha(old['arrays'])==capture['old_arrays_sha256']==old['arrays_sha256']
    with np.load(capture['arrays'],allow_pickle=False) as a,np.load(old['arrays'],allow_pickle=False) as b:
        for key in ('parameter_names','point','validation_points','validation_labels'):np.testing.assert_array_equal(a[key],b[key])
        gradients=a['validation_gradients'];points=a['validation_points'];names=a['parameter_names']
        assert gradients.shape==points.shape==(26,2656) and np.isfinite(gradients).all()
        np.testing.assert_array_equal(a['gradient'],gradients[0]);np.testing.assert_array_equal(a['point'],points[0])
        assert a['vg_value']==a['validation_vg_values'][0] and a['value']==a['validation_values'][0]
        fixed=np.isin(names,capture['fixed_parameters']);assert np.count_nonzero(fixed)==4
        np.testing.assert_array_equal(gradients[:,fixed],np.zeros((26,4)))
        absolute=np.abs(gradients[:,~fixed]-b['validation_gradients'][:,~fixed]);relative=absolute/np.maximum(np.abs(b['validation_gradients'][:,~fixed]),1e-8)
        assert np.max(relative)<1e-4
        np.testing.assert_array_equal(np.max(absolute,axis=1),[q['old_ad_max_absolute_error'] for q in capture['cases']])
        np.testing.assert_array_equal(np.max(relative,axis=1),[q['old_ad_max_relative_error'] for q in capture['cases']])
        assert sha(values['native_report'])==values['native_report_sha256'];native=json.loads(Path(values['native_report']).read_bytes())
        assert native['source']==capture['source'] and native['parameter_names']==names.tolist()
        np.testing.assert_array_equal(points,[q['theta'] for q in native['cases']])
        expected=np.array([q['combine_nll'] for q in native['cases']])
        nll_errors={k:float(np.max(np.abs(a[k]-expected))) for k in ('validation_values','validation_vg_values')}
        assert all(v<1e-6 for v in nll_errors.values())
    baseline=Path('reports/gpu_hhvv_cpu_scoped_gradient_all_points_checkpoint_combine_random_gradients_refined.json')
    reference=json.loads(baseline.read_bytes());assert reference['status']=='passed'
    commands=[
        [sys.executable,'scripts/reconcile_gpu_gradient_references.py',str(path),'--directions-reference','reports/gpu_combine_directions_refined.json'],
        [sys.executable,'scripts/compare_gpu_random_gradients.py',str(path),'--reference',reference['reference'],
            '--stable-reference',reference['stable_reference'],'--refined-full-nll-reference',reference['refinements']['full_nll_reference'],
            '--refined-intensity-reference',reference['refinements']['intensity_reference']]]
    for command in commands:subprocess.run(command,check=True)
    nominal=verified_refinement(path,capture);random=random_validation(path,capture,sha(path),refined=True)
    assert nominal['status']==nominal['directions_status']==random['status']=='passed'
    random_report=json.loads(Path(random['validation']).read_bytes())
    assert random['status_counts']==dict(passed=66300,fixed_bounds=100)
    maximum=max(p['max_relative_error'] for p in random_report['points']);assert maximum<1e-4
    old_comparison=json.loads(Path(capture['old_comparison']).read_bytes())
    previous=old_comparison['combine_validation']['cpu']
    for group,keys in [('nominal',('reference_sha256','directions_reference_sha256')),
                       ('random',('reference_sha256','stable_reference_sha256','refinement_documents_sha256'))]:
        result=nominal if group=='nominal' else random
        for key in keys:assert result[key]==previous[group][key]
    report=dict(status='passed',script_sha256=sha(__file__),capture=str(path),capture_sha256=sha(path),arrays_sha256=capture['arrays_sha256'],
        values_report=capture['values_report'],values_report_sha256=capture['values_report_sha256'],source=capture['source'],
        nominal=nominal,random=random,native_random_max_relative_error=maximum,
        native_random_max_absolute_error=max(p['max_absolute_error'] for p in random_report['points']),
        nll_max_absolute_errors=nll_errors,old_ad_max_absolute_error=float(np.max(absolute)),old_ad_max_relative_error=float(np.max(relative)),
        excluded_fixed_parameters=capture['fixed_parameters'],timings=capture['timings'],peak_rss_mib=capture['peak_rss_mib'],
        runtime={k:capture[k] for k in ('jax_version','python','backend','devices','float64','hostname','cpu_model','cpu_affinity','environment')},
        checker_sources={str(p):sha(p) for p in map(Path,('scripts/reconcile_gpu_gradient_references.py','scripts/compare_gpu_random_gradients.py',
            'scripts/gpu_random_refinements.py','scripts/fd_reference.py','scripts/compare_gpu_results.py','scripts/gpu_gradient_evidence.py'))},
        timing_notes='The constant-free jaxpr check precedes the reported lowering timer. Compile seconds cover XLA compilation only; the full cold-start preparation cost is not represented by lowering+compile alone. Warm calls synchronize device results without copying them to NumPy.',
        policy='Identical frozen native reference hashes and unchanged comparison scripts as the accepted old-backend CPU run. Relative tolerance 1e-4; all selected estimates must pass. Four fixed-bound coordinates have exact zero tensor derivatives.',
        scope='Full HHVV tensor NLL and gradient acceptance at nominal plus 25 random points. No HHVV Hessian, fit, covariance or profile acceptance; local timings are not a same-node comparison with lxplus or GPU.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('passed',nominal['max_relative_error'],maximum,report['old_ad_max_relative_error'],flush=True)


if __name__=='__main__':main()
