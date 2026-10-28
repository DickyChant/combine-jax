"""Rederive full tensor curvature comparisons and preserve native uncertainty."""
import hashlib
import json
from pathlib import Path

import numpy as np

from compare_local_cpu_hessian import load_arrays,matrix_errors,reassemble_local
from compare_gpu_cached_curvature import compare as native_compare


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_hhvv_hessian.json');output=Path('reports/tensor_hhvv_hessian_audit.json');assert not output.exists()
    r=json.loads(path.read_bytes());assert r['status']=='validated_old_ad' and r['stage']=='complete'
    assert r['script_sha256']==sha('scripts/validate_tensor_hhvv_hessian.py')
    for p,digest in r['tensor_sources'].items():assert sha(p)==digest
    for key in ('import_report','gradient_gate','reference_report','prior_reference_audit','tests_log','arrays'):
        assert sha(r[key])==r[key+'_sha256']
    assert '12 passed, 2 warnings' in Path(r['tests_log']).read_text()
    reference=json.loads(Path(r['reference_report']).read_bytes());oracle=load_arrays(reference)
    assert reference['arrays_sha256']==r['reference_arrays_sha256']
    assembly=reassemble_local(reference,oracle);assert assembly==r['reference_reassembly']
    gate=json.loads(Path(r['gradient_gate']).read_bytes());assert gate['status']=='passed'
    assert sha(gate['capture'])==gate['capture_sha256'];capture=json.loads(Path(gate['capture']).read_bytes())
    assert sha(capture['arrays'])==capture['arrays_sha256']
    with np.load(r['arrays'],allow_pickle=False) as a,np.load(capture['arrays'],allow_pickle=False) as c:
        for key in ('parameter_names','point'):np.testing.assert_array_equal(a[key],c[key]);np.testing.assert_array_equal(a[key],oracle[key])
        np.testing.assert_array_equal(a['parameter_scales'],oracle['parameter_scales'])
        h=a['hessian'];assert h.shape==(2656,2656) and np.isfinite(h).all()
        fixed=np.isin(a['parameter_names'],r['excluded_fixed_parameters']);assert np.count_nonzero(fixed)==4;free=~fixed
        np.testing.assert_array_equal(h[fixed,:],0.);np.testing.assert_array_equal(h[:,fixed],0.)
        base={k:a[k][free] for k in ('point','parameter_names','parameter_scales')}
        comparison=matrix_errors(dict(base,hessian=oracle['hessian'][np.ix_(free,free)]),dict(base,hessian=h[np.ix_(free,free)]))
        assert comparison==r['comparison'] and comparison['passed']
        absolute=np.abs(a['gradient']-c['gradient']);relative=absolute/np.maximum(np.abs(c['gradient']),1e-8)
        assert dict(max_absolute_error=float(absolute.max()),max_relative_error=float(relative.max()))==r['gradient_comparison']
        assert abs(float(a['value'])-float(c['vg_value']))==r['nll_absolute_error']<1e-6 and np.max(relative)<1e-4
    # Adapt the completed artifact to the unchanged native Hessian consumer's
    # protocol, preserving the exact raw producer report and its checksum.
    view=Path('reports/tensor_hhvv_hessian_reference_view.json');assert not view.exists()
    view.write_text(json.dumps(dict(source=r['source'],arrays=r['arrays'],arrays_sha256=r['arrays_sha256'],completed_operations=['hessian'],
        producer=str(path),producer_sha256=sha(path),producer_status=r['status']),indent=2)+'\n')
    prior_path=Path('reports/gpu_hhvv_recovered_hessian_reference_checked_curvature.json');prior=json.loads(prior_path.read_bytes())
    ref=Path(prior['reference_audit']['reference']);assert sha(ref)==prior['reference_audit']['reference_sha256']
    native=native_compare(view,ref);native_path=Path('reports/tensor_hhvv_hessian_native_projections.json');assert not native_path.exists()
    native_path.write_text(json.dumps(native,indent=2,allow_nan=False)+'\n')
    assert native['reference_audit']==prior['reference_audit']
    assert native['unresolved_directions']==prior['unresolved_directions'] and not native['failed_directions']
    overlap_path=Path('reports/tensor_hhvv_hessian_overlap.json')
    overlap=json.loads(overlap_path.read_bytes());assert overlap['hessian_report']==str(path)
    report=dict(status='passed_old_ad_with_native_reference_uncertainty',script_sha256=sha(__file__),producer=str(path),producer_sha256=sha(path),
        arrays_sha256=r['arrays_sha256'],reference_reassembly=assembly,free_hessian_comparison=comparison,
        excluded_fixed_parameters=r['excluded_fixed_parameters'],native_projections=str(native_path),native_projections_sha256=sha(native_path),
        native_status=native['status'],native_passed_directions=[q['label'] for q in native['cases'] if q['status']=='passed'],
        native_unresolved_directions=native['unresolved_directions'],prior_native_comparison=str(prior_path),prior_native_comparison_sha256=sha(prior_path),
        checker_sources={p:sha(p) for p in ('scripts/compare_local_cpu_hessian.py','scripts/compare_gpu_cached_curvature.py','scripts/compare_gpu_curvature.py')},
        timings=r['timings'],compiler_memory=r['compiler_memory'],peak_rss_mib=r['peak_rss_mib'],
        execution_timing_scope=overlap['timing_scope'],execution_isolated_benchmark=False,
        overlap_record=str(overlap_path),overlap_record_sha256=sha(overlap_path),
        scope='Every free nominal Hessian entry agrees with the full old CPU AD oracle; four fixed-bound rows/columns are exactly zero. The unchanged native projection gate retains its unresolved references. No minimum, covariance or warm repeated-Hessian acceptance.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['status'],comparison,native['status'],flush=True)


if __name__=='__main__':main()
