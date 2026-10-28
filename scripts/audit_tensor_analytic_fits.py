"""Audit analytic-card results, retaining the failed bounded-error comparison."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_analytic_fits.json');output=Path('reports/tensor_analytic_fits_audit.json');assert not output.exists()
    r=json.loads(path.read_bytes());assert r['status']=='failed_gates' and r['stage']=='complete'
    for p,h in r['sources'].items():assert sha(p)==h
    assert [c['label'] for c in r['cases']]==['parametric','multipdf_exp','multipdf_bern']
    summaries=[]
    for c in r['cases']:
        assert sha(c['workspace'])==c['workspace_sha256']
        for n,h in c['model_sha256'].items():assert sha(Path(c['model'])/n)==h
        d,n=c['device'],c['native'];assert n['valid'] and n['accurate_covariance']
        assert d['status']==0 and d['covariance_status']==0 and d['inverse_residual']<1e-6
        assert abs(d['value']-n['nll'])==c['nll_absolute_error']<1e-6
        dv,nv,de,ne=map(np.asarray,(d['values'],n['values'],d['errors'],n['errors']))
        ratios=np.abs(dv-nv)/np.maximum(.01*np.abs(nv),.01*ne)
        np.testing.assert_array_equal(ratios,c['parameter_error_over_tolerance']);assert np.all(ratios<1.)
        errors=np.abs(de-ne)/ne;np.testing.assert_array_equal(errors,c['error_relative_difference'])
        h=np.array(d['hessian']);cov=np.array(d['covariance']);native_cov=np.array(n['covariance'])
        assert np.linalg.eigvalsh(h).min()>0
        np.testing.assert_allclose(h@cov,np.eye(len(dv)),atol=1e-8,rtol=0.)
        np.testing.assert_allclose(np.sqrt(np.diag(cov)),de,atol=1e-14,rtol=1e-14)
        expected=[] if np.all(errors<.01) else ['uncertainties'];assert c['failed_gates']==expected
        assert c['status']==('passed' if not expected else 'failed')
        if expected:
            assert c['label']=='multipdf_bern' and n['has_parameters_at_limit'] and not c['profiles']
        else:
            assert len(c['profiles'])==2
            for p in c['profiles']:
                assert p['passed'] and p['native_valid'] and p['device_status']==0
                assert abs(p['device_nll']-p['native_nll'])==p['nll_absolute_error']<1e-6
                np.testing.assert_allclose(p['device_values'],p['native_values'],atol=1e-4,rtol=1e-4)
        summaries.append(dict(label=c['label'],status=c['status'],failed_gates=expected,nll_absolute_error=c['nll_absolute_error'],
            max_parameter_error_over_tolerance=float(np.max(ratios)),max_reported_error_relative_difference=float(np.max(errors)),
            max_native_covariance_diagonal_error_relative_difference=float(np.max(np.abs(de-np.sqrt(np.diag(native_cov)))/np.sqrt(np.diag(native_cov)))),
            native_parameters_at_limit=n['has_parameters_at_limit'],old_hessian_max_absolute_error=c['old_hessian_max_absolute_error'],
            max_profile_nll_absolute_error=max((p['nll_absolute_error'] for p in c['profiles']),default=None)))
    old={}
    for p in Path('src/combine_jax').glob('*.py'):
        previous=subprocess.run(['git','show',f'dca96c2:{p}'],check=True,capture_output=True).stdout
        assert hashlib.sha256(previous).hexdigest()==sha(p);old[str(p)]=sha(p)
    output.write_text(json.dumps(dict(status='audited_with_failed_uncertainty_gate',producer=str(path),producer_sha256=sha(path),
        script_sha256=sha(__file__),old_backend_unchanged_since='dca96c2',old_backend_sources=old,cases=summaries,
        scope='Parametric and fixed exponential MultiPdf branches pass. The Bernstein branch passes values, minima and old-AD curvature, but fails reported native Minuit errors near bounds. No substitution of covariance diagonal errors; no discrete-envelope or performance acceptance.'),indent=2,allow_nan=False)+'\n')
    print('audited: 2 passed cards/branches; Bernstein reported-error gate remains failed',flush=True)


if __name__=='__main__':main()
