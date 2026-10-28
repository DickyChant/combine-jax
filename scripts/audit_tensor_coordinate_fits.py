"""Recompute gradients after coordinate-bound correction using unchanged native gates."""
import hashlib
import json
from pathlib import Path
import re
import subprocess

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info,fit_coordinates
from combine_jax.tensor.reported_errors import minuit_reported_errors


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=Path('reports/tensor_coordinate_bounds_analytic_fits.json');output=Path('reports/tensor_coordinate_bounds_fits_audit.json');assert not output.exists()
    r=json.loads(source.read_bytes());assert r['status']=='passed' and r['stage']=='complete'
    for p,h in r['sources'].items():assert sha(p)==h
    log=Path('reports/tensor_coordinate_bounds_tests.log');text=log.read_text();assert '30 passed, 2 warnings' in text
    regression=json.loads(re.findall(r'LINE_SEARCH_POISSON_JSON=(\[[^\n]*\])',text)[0])
    assert regression[0]['status']==2 and regression[0]['projected_gradient_max']>1e-8
    assert regression[1]['status']==0 and regression[1]['projected_gradient_max']<=1e-8 and regression[1]['roundoff_accepts']>0
    fitlog=Path('reports/tensor_coordinate_bounds_tests.log');fittext=fitlog.read_text();assert '30 passed, 2 warnings' in fittext
    cache=json.loads(re.findall(r'CACHE_VALIDATION_JSON=(\{[^\n]*\})',fittext)[0]);assert cache['status']=='passed'
    assert all(len(c['cache_hit_lines'])==2 for c in cache['cases'] if c['scale']==1.4)
    summaries=[]
    for c in r['cases']:
        assert c['status']=='passed' and not c['failed_gates']
        model=TensorBundle.load(c['model']);info=model_parameter_info(model)
        for n,h in c['model_sha256'].items():assert sha(Path(c['model'])/n)==h
        arrays=jax.device_put(dict(model.arrays));value_gradient=jax.jit(jax.value_and_grad(make_model_function(model)))
        checks=[]
        records=[dict(label='minimum',values=c['device']['values'],reference=c['native']['nll'],
            gradient=c['device']['projected_gradient'],coordinates=fit_coordinates(info),source=c['device'])]
        assert len(c['profiles'])==2
        for p in c['profiles']:
            assert p['passed'] and p['device_status']==0 and p['native_valid']
            records.append(dict(label='profile_r_'+str(p['r']),values=p['device_values'],reference=p['native_nll'],
                gradient=p['projected_gradient'],coordinates=fit_coordinates(info,initial=c['device']['values'],fixed={'r':p['r']}),source=p))
        for record in records:
            values=np.array(record['values']);nll,g=map(np.asarray,value_gradient(values,arrays));coordinates=record['coordinates']
            free=coordinates['free_indices'];scaled=g[free]*coordinates['scale']
            z=(values[free]-coordinates['reference'][free])/coordinates['scale'];eps=32*np.finfo(np.float64).eps*(1+np.abs(z))
            blocked=((z<=coordinates['lower']+eps)&(scaled>0))|((z>=coordinates['upper']-eps)&(scaled<0))
            pg=np.where(blocked,0.,scaled);norm=float(np.max(np.abs(pg),initial=0.))
            np.testing.assert_allclose(pg,record['gradient'],atol=2e-10,rtol=1e-5)
            assert norm<=1e-7 and abs(float(nll)-record['reference'])<1e-6
            assert record['source']['maximum_roundoff_allowance']<2e-12
            checks.append(dict(label=record['label'],recomputed_projected_gradient_max=norm,
                native_nll_absolute_error=abs(float(nll)-record['reference']),roundoff_accepts=record['source']['roundoff_accepts'],
                maximum_roundoff_allowance=record['source']['maximum_roundoff_allowance']))
        d,n=c['device'],c['native'];assert d['covariance_status']==0 and n['accurate_covariance'] and n['valid']
        h,cov=map(np.array,(d['hessian'],d['covariance']));assert np.linalg.eigvalsh(h).min()>0
        np.testing.assert_allclose(h@cov,np.eye(len(h)),atol=1e-8,rtol=0.)
        np.testing.assert_allclose(cov,n['covariance'],atol=1e-5,rtol=.01)
        sigma=np.sqrt(np.diag(cov));np.testing.assert_array_equal(sigma,d['errors'])
        reported,status=minuit_reported_errors(np.array(d['values']),sigma,info.arrays['lower'],info.arrays['upper'])
        np.testing.assert_array_equal(status,0);np.testing.assert_allclose(reported,d['reported_errors'],rtol=2e-12,atol=2e-13)
        relative=np.abs(reported-n['errors'])/n['errors'];assert np.max(relative)<.01
        tolerance=np.maximum(.01*np.abs(n['values']),.01*np.array(n['errors']))
        assert np.all(np.abs(np.array(d['values'])-n['values'])<tolerance)
        summaries.append(dict(label=c['label'],status='passed',checks=checks,max_reported_error_relative_difference=float(np.max(relative))))
        jax.clear_caches()
    old={}
    for p in Path('src/combine_jax').glob('*.py'):
        previous=subprocess.run(['git','show',f'3217d9a:{p}'],check=True,capture_output=True).stdout
        assert hashlib.sha256(previous).hexdigest()==sha(p);old[str(p)]=sha(p)
    report=dict(status='passed',producer=str(source),producer_sha256=sha(source),script_sha256=sha(__file__),
        gradient_tolerance=1e-7,strict_poisson_gradient_tolerance=1e-8,armijo_roundoff_multiplier=8.,
        gate_source='scripts/audit_tensor_roundoff_fits.py',gate_source_sha256=sha('scripts/audit_tensor_roundoff_fits.py'),
        tests=30,test_logs={str(p):sha(p) for p in (log,fitlog)},poisson_regression=regression,cases=summaries,
        old_backend_unchanged_since='3217d9a',old_backend_sources=old,
        scope='Recomputed NLL/gradients validate every analytic minimum/profile at unchanged thresholds. All five baseline card/branch fit and reported-error gates pass; all ten checked profile points pass. Raw covariance and separate Minuit reporting remain distinct. No discrete-envelope, HHVV fit or performance acceptance.')
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('passed: independent gradients at 3 minima + 6 profile points; 30 tests',flush=True)


if __name__=='__main__':main()
