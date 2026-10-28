"""Recompute saved template gates without ROOT/JAX or their evaluation kernels."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_templates.json');report=json.loads(path.read_bytes())
    destination=Path('reports/tensor_templates_audit.json');assert not destination.exists()
    assert report['status']=='passed' and report['float64'] and report['backend']=='cpu'
    assert sha('scripts/validate_tensor_templates.py')==report['script_sha256']
    for p,h in report['tensor_sources'].items():assert sha(p)==h
    for p,h in report['artifacts'].items():assert sha(p)==h
    h=report['hhvv'];assert h['closed_over_arrays']==0 and report['automcstats']['closed_over_arrays']==0
    for stem in ('prior_native_terms_report','native_channel_report','validation_arrays'):
        assert sha(h[stem])==h[stem+'_sha256']
    previous=json.loads(Path(h['prior_native_terms_report']).read_bytes())
    assert sha(previous['validation_arrays'])==previous['validation_arrays_sha256']
    references=np.load(previous['validation_arrays'],allow_pickle=False)
    arrays=np.load(h['validation_arrays'],allow_pickle=False)
    native=json.loads(Path(h['native_channel_report']).read_bytes())
    assert native['source']==h['source'] and h['template_channels']+h['counting_channels']+h['remaining_channels']==459
    np.testing.assert_array_equal(arrays['points'],references['points'])
    np.testing.assert_array_equal(arrays['points'],np.array([c['theta'] for c in native['cases']]))
    for name,reference in [('profiles','profile_values'),('constraints','native_constraints'),('counting','native_channels')]:
        np.testing.assert_array_equal(arrays['native_'+name],references[reference])
    expected=np.array([[{ch['name']:ch['combine'] for ch in c['channels']}[n] for n in arrays['channel_names']] for c in native['cases']])
    np.testing.assert_array_equal(arrays['native_channels'],expected)
    results={}
    for name,count in [('channels',182),('profiles',74),('constraints',2124),('counting',178)]:
        for backend in ('jax','numpy'):
            actual=arrays[backend+'_'+name];ref=arrays['native_'+name]
            assert actual.shape==ref.shape==(26,count) and np.all(np.isfinite(actual)) and np.all(np.isfinite(ref))
            delta=np.abs(actual-ref);limit=1e-10 if name=='profiles' else 1e-6
            assert np.all(delta<limit)
            field=name if backend=='jax' else backend+'_'+name
            np.testing.assert_array_equal(np.max(delta,axis=1),[c[field]['maximum_absolute_error'] for c in h['cases']])
            results[backend+'_'+name]=dict(shape=list(actual.shape),max_absolute_error=float(np.max(delta)),tolerance=limit)
    a=report['automcstats'];assert sha(a['manifest'])==a['manifest_sha256']
    manifest=json.loads(Path(a['manifest']).read_bytes());reference=manifest['models']['automcstats']
    assert report['combine_version']==manifest['combine_version'] and report['root_version']==manifest['root_version']
    assert a['parameter_names']==reference['reference']['parameters'] and a['source']==reference['source']
    assert len(a['cases'])==len(reference['reference']['cases'])==26
    nll=[];gradient=[];hessian=[]
    for c,r in zip(a['cases'],reference['reference']['cases'],strict=True):
        assert c['native_nll']==r['combine_nll'] and c['root_evaluation_errors']==0
        d=abs(c['nll']-r['combine_nll']);assert d==c['nll_absolute_error'] and d<1e-6;nll.append(d)
        g=np.array(c['gradient']);fd=np.array(r['combine_fd_gradient'])
        relative=np.abs(g-fd)/np.maximum(1e-8,np.abs(fd))
        np.testing.assert_array_equal(relative,c['gradient_relative_error'])
        assert np.all(relative<1e-4);gradient.append(float(np.max(relative)))
        if 'hessian' in c:
            d=float(np.max(np.abs(np.array(c['hessian'])-np.array(r['fd_hessian']))))
            assert d==c['fd_hessian_max_error'] and d<1e-6;hessian.append(d)
    assert len(hessian)==3
    baseline='6746693'
    files=subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'src/combine_jax'],text=True).splitlines()
    oracle={}
    for p in files:
        if p.startswith('src/combine_jax/tensor/'):continue
        original=subprocess.check_output(['git','show',baseline+':'+p])
        assert Path(p).read_bytes()==original
        oracle[p]=sha(p)
    testlog=Path('reports/tensor_templates_tests.log')
    assert '44 passed, 2 warnings' in testlog.read_text()
    result=dict(status='passed',report=str(path),report_sha256=sha(path),script_sha256=sha(__file__),hhvv=results,
        automcstats=dict(points=26,max_nll_absolute_error=max(nll),max_gradient_relative_error=max(gradient),
            hessian_points=3,max_fd_hessian_error=max(hessian)),
        old_backend_baseline=baseline,old_backend_unchanged_files=oracle,
        focused_tests=dict(log=str(testlog),sha256=sha(testlog),passed=44,warnings=2),
        scope='Saved native matrix/point/source alignment, gates, runtime buffer trace and unchanged old backend. No additional full HHVV or performance claim.')
    destination.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='old_backend_unchanged_files'},indent=2))


if __name__=='__main__':main()
