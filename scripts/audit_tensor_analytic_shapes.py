"""Recompute analytic-channel gates from saved native and tensor evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_analytic_shapes.json');r=json.loads(path.read_bytes())
    output=Path('reports/tensor_analytic_shapes_audit.json');assert not output.exists()
    assert r['status']=='passed' and r['float64'] and r['backend']=='cpu'
    for p,key in [('scripts/validate_tensor_analytic_shapes.py','script_sha256'),('scripts/validate_analytic_native_reference.py','native_reference_worker_sha256')]:assert sha(p)==r[key]
    for p,h in {**r['tensor_sources'],**r['artifacts']}.items():assert sha(p)==h
    h=r['hhvv'];c=r['parametric_card'];assert h['closed_over_arrays']==c['closed_over_arrays']==0
    for field in ('validation_arrays','prior_report','export'):assert sha(h[field])==h[field+'_sha256']
    for p,digest in h['input_artifacts'].items():assert sha(p)==digest
    prior=json.loads(Path(h['prior_report']).read_bytes())['hhvv']
    for field in ('native_report','term_report'):assert sha(prior[field])==prior[field+'_sha256']
    native=json.loads(Path(prior['native_report']).read_bytes());terms=json.loads(Path(prior['term_report']).read_bytes())
    assert sha(terms['validation_arrays'])==terms['validation_arrays_sha256']
    refs=np.load(terms['validation_arrays'],allow_pickle=False);a=np.load(h['validation_arrays'],allow_pickle=False)
    np.testing.assert_array_equal(a['points'],refs['points']);np.testing.assert_array_equal(a['points'],[case['theta'] for case in native['cases']])
    assert a['native_channels'].shape==(26,74) and len(set(a['subtotal_channels']))==434
    assert a['channel_names'].tolist()==h['channels'] and a['subtotal_channels'].tolist()==h['subtotal_channels']
    assert set(prior['channels'])<=set(h['channels'])
    assert len(set(h['channels'])-set(prior['channels']))==24 and all(n.startswith('WW_lvqq') for n in set(h['channels'])-set(prior['channels']))
    from combine_jax.tensor.bundle import TensorBundle
    model=TensorBundle.load(Path(h['validation_arrays']).parent/'hhvv_subset')
    assert model.metadata['provenance']['all_card_channels_imported'] is False and model.metadata['channel_names']==h['subtotal_channels']
    assert model.structure_key==h['structure_key'] and model.metadata['parameter_names']==native['parameter_names']
    for i,case in enumerate(native['cases']):
        ref={q['name']:q['combine'] for q in case['channels']}
        np.testing.assert_array_equal(a['native_channels'][i],[ref[n] for n in h['channels']])
        assert a['native_subtotal'][i]==sum(ref[n] for n in h['subtotal_channels'])+np.sum(refs['native_constraints'][i])+model.arrays['offset']
    maxima={}
    for kind in ('channels','subtotal'):
        for backend in ('jax','numpy'):
            actual=a[backend+'_'+kind];ref=a['native_'+kind];delta=np.abs(actual-ref)
            assert actual.shape==ref.shape and np.all(np.isfinite(actual)) and np.all(delta<1e-6)
            values=np.max(delta,axis=1) if kind=='channels' else delta;field=kind if backend=='jax' else backend+'_'+kind
            np.testing.assert_array_equal(values,[q[field]['max_absolute_error'] for q in h['cases']]);maxima[backend+'_'+kind]=float(np.max(delta))
    all_names={q['name'] for q in native['cases'][0]['channels']};remaining=sorted(all_names-set(h['subtotal_channels']))
    assert remaining==h['remaining_channels'] and len(remaining)==25
    for field in ('native_reference','workspace','card'):assert sha(c[field])==c[field+'_sha256']
    reference=json.loads(Path(c['native_reference']).read_bytes())
    assert reference['status']=='passed' and reference['root_evaluation_errors']==0
    assert reference['worker_sha256']==r['native_reference_worker_sha256'] and reference['workspace_sha256']==c['workspace_sha256']
    assert reference['parameters']==c['parameter_names'] and len(c['cases'])==len(reference['cases'])==26
    process=c['native_process'];assert process['exit_code']==0 and sha(process['log'])==process['log_sha256']
    cardmax={k:0. for k in ('nll','gradient_relative','hessian_absolute_fd','hessian_scaled_fd','hessian_old_ad')};hessians=0
    for actual,ref in zip(c['cases'],reference['cases'],strict=True):
        assert actual['theta']==ref['theta'] and actual['native_nll']==ref['combine_nll']
        delta=abs(actual['nll']-ref['combine_nll']);assert delta==actual['nll_absolute_error'] and delta<1e-6;cardmax['nll']=max(cardmax['nll'],delta)
        assert actual['numpy_absolute_error']<1e-6
        g=np.array(actual['gradient']);fd=np.array(ref['combine_fd_gradient']);relative=np.abs(g-fd)/np.maximum(np.abs(fd),1e-8)
        np.testing.assert_array_equal(relative,actual['gradient_relative_error']);assert np.all(relative<1e-4)
        assert np.max(np.abs(g-np.array(ref['jax_gradient'])))==actual['old_ad_gradient_max_error']<1e-8
        cardmax['gradient_relative']=max(cardmax['gradient_relative'],float(np.max(relative)))
        if 'hessian' in actual:
            hessians+=1;actual_h=np.array(actual['hessian']);fd_h=np.array(ref['fd_hessian']);d=np.abs(actual_h-fd_h)
            absolute=float(np.max(d));scaled=float(np.max(d/np.maximum(np.abs(fd_h),1.)));old=float(np.max(np.abs(actual_h-np.array(ref['hessian']))))
            assert absolute==actual['fd_hessian_max_error'] and scaled==actual['fd_hessian_max_scaled_error']<1e-6
            assert old==actual['old_ad_hessian_max_error']<1e-8
            for key,v in [('hessian_absolute_fd',absolute),('hessian_scaled_fd',scaled),('hessian_old_ad',old)]:cardmax[key]=max(cardmax[key],v)
    assert hessians==3
    model_path=Path(c['native_reference']).parent/'parametric_model'
    script='''import sys,numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import evaluate_model_numpy
m=TensorBundle.load(sys.argv[1]);assert m.metadata['provenance']['all_card_channels_imported']
v=float(evaluate_model_numpy(m,m.arrays['scalars__parameter_initial']));assert np.isfinite(v)
assert not any(n in sys.modules for n in ('ROOT','jax','torch'))
print(v)
'''
    run=subprocess.run([sys.executable,'-c',script,str(model_path)],capture_output=True,text=True,check=True)
    assert abs(float(run.stdout)-c['cases'][0]['native_nll'])<1e-6
    baseline='c798cc9';unchanged={}
    for p in sorted(Path('src/combine_jax').glob('*.py')):
        old=subprocess.run(['git','show',f'{baseline}:{p}'],check=True,capture_output=True).stdout
        assert p.read_bytes()==old;unchanged[str(p)]=sha(p)
    testlog=Path('reports/tensor_analytic_shapes_tests.log');assert '102 passed, 2 warnings' in testlog.read_text()
    probe_path=Path('reports/tensor_negated_exponential_native_probe.json');probe=json.loads(probe_path.read_bytes())
    assert probe['status']=='native_mismatch_confirmed' and probe['root_evaluation_errors']==0
    assert sha('scripts/probe_negated_exponential_cache.py')==probe['script_sha256'] and sha(probe['native_source'])==probe['native_source_sha256']
    assert probe['runtime_flags']==r['runtime_flags']
    native_g=probe['cases'][1]['nominal_fd_caching_gradient'];roofit_g=probe['cases'][1]['nominal_fd_roofit_gradient']
    assert abs(native_g-roofit_g)>20
    preflight_path=Path('reports/tensor_analytic_shapes_preflight.json');preflight=json.loads(preflight_path.read_bytes())
    preflight_source=Path('build/tensor_analytic_validation/validation_script_preflight.py')
    assert preflight['status']=='failed' and not preflight['hhvv']['cases']
    assert sha(preflight_source)==preflight['script_sha256'] and 'KeyError' in preflight['exception']
    out=dict(status='passed',report=str(path),report_sha256=sha(path),audit_script_sha256=sha(__file__),
        hhvv=dict(points=26,channels=74,subtotal_channels=434,remaining_channels=remaining,maximum_errors=maxima),
        parametric_card=dict(points=26,channels=2,hessian_points=3,maximum_errors=cardmax,root_evaluation_errors=0,neutral_transport=True),
        tests=dict(passed=102,warnings=2,log=str(testlog),sha256=sha(testlog)),old_backend_baseline=baseline,old_backend_unchanged_files=unchanged,
        rejected_negated_exponential=dict(report=str(probe_path),sha256=sha(probe_path),native_gradient=native_g,roofit_gradient=roofit_g),
        preflight_failure=dict(report=str(preflight_path),sha256=sha(preflight_path),script=str(preflight_source),script_sha256=sha(preflight_source),
            explanation='Input hash inventory mistakenly included an unrelated validation array; corrected to the two bundle files before numerical evaluation.'),
        scope='Saved value/derivative gates only. Other 25 HHVV channels and full-model derivatives, fitting, cache reuse and GPU measurements remain pending.')
    output.write_text(json.dumps(out,indent=2)+'\n');print('passed',maxima,cardmax,flush=True)


if __name__=='__main__':main()
