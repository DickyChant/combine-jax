"""Audit distinct curvature/reported errors and keep the open profile failure."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from combine_jax.tensor.reported_errors import minuit_reported_errors,SOURCE_URL


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=Path('reports/tensor_analytic_reported_error_fits.json');output=Path('reports/tensor_reported_errors_audit.json');assert not output.exists()
    r=json.loads(source.read_bytes());assert r['stage']=='complete' and r['status']=='failed_gates'
    for p,h in r['sources'].items():assert sha(p)==h
    original=Path('reports/tensor_analytic_fits.json');old=json.loads(original.read_bytes())
    log=Path('reports/tensor_reported_errors_fit_tests.log');text=log.read_text();assert '14 passed, 2 warnings' in text
    saved=json.loads(re.findall(r'REPORTED_ERROR_JSON=(\[[^\n]*\])',text)[0]);assert len(saved)==3
    summaries=[]
    for c,previous in zip(r['cases'],old['cases'],strict=True):
        assert c['label']==previous['label']
        d,n=c['device'],c['native'];assert d['status']==0 and d['covariance_status']==0 and n['valid'] and n['accurate_covariance']
        # Reporting is additional output; the accepted optimizer and raw covariance are unchanged.
        for key in ('value','values','errors','covariance','hessian'):
            np.testing.assert_array_equal(d[key],previous['device'][key])
        sigma=np.sqrt(np.diag(d['covariance']));np.testing.assert_array_equal(sigma,d['errors'])
        lo=np.array([p['minimum'] for p in c['parameters']]);hi=np.array([p['maximum'] for p in c['parameters']])
        reported,status=minuit_reported_errors(np.array(d['values']),sigma,lo,hi)
        np.testing.assert_array_equal(status,0);np.testing.assert_array_equal(d['reported_error_status'],0)
        np.testing.assert_allclose(reported,d['reported_errors'],atol=2e-13,rtol=2e-12)
        relative=np.abs(np.array(d['reported_errors'])-n['errors'])/n['errors'];assert np.max(relative)<.01
        np.testing.assert_array_equal(relative,c['error_relative_difference'])
        np.testing.assert_allclose(d['covariance'],n['covariance'],atol=1e-5,rtol=.01)
        assert c['nll_absolute_error']<1e-6 and np.max(c['parameter_error_over_tolerance'])<1.
        failures=[p for p in c['profiles'] if not p['passed']]
        assert c['failed_gates']==(['profile'] if failures else [])
        if failures:
            assert c['label']=='multipdf_bern' and len(failures)==1
            assert failures[0]['device_status']==2 and failures[0]['native_valid']
        summaries.append(dict(label=c['label'],baseline_fit_and_reported_errors_pass=True,
            max_reported_error_relative_difference=float(np.max(relative)),raw_covariance_unchanged=True,
            profiles_passed=sum(p['passed'] for p in c['profiles']),failed_profiles=failures))
    probe=Path('reports/tensor_profile_line_probe.json');p=json.loads(probe.read_bytes());assert p['status']==2
    library=Path('.venv/lib/python3.13/site-packages/ROOT/lib/libMinuit2.so');assert library.is_file()
    report=dict(status='passed_error_reporting_with_open_profile_failure',producer=str(source),producer_sha256=sha(source),
        prior_raw_error_comparison=str(original),prior_raw_error_comparison_sha256=sha(original),script_sha256=sha(__file__),
        sources={p:sha(p) for p in ('src/combine_jax/tensor/reported_errors.py','src/combine_jax/tensor/fitting.py','tests/test_tensor_reported_errors.py','tests/test_tensor_fitting.py')},
        root_version=r['root_version'],native_library=str(library),native_library_sha256=sha(library),source_documentation=SOURCE_URL,
        test_log=str(log),test_log_sha256=sha(log),tests=14,native_transformation_test_points=384,
        saved_fit_transform_comparisons=saved,cases=summaries,open_profile_probe=str(probe),open_profile_probe_sha256=sha(probe),
        scope='Raw covariance and its diagonal errors remain byte-identical. Separate reported_errors match the native bounded reporting convention. All three analytic baseline fit/error gates pass; one Bernstein profile still fails the unchanged optimizer convergence requirement.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(report['status'],flush=True)


if __name__=='__main__':main()
