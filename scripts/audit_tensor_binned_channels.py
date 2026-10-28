"""Recheck binned-channel evidence, reference alignment and neutral transport."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_binned_channels.json');r=json.loads(path.read_bytes())
    output=Path('reports/tensor_binned_channels_audit.json');assert not output.exists()
    assert r['status']=='passed' and r['float64'] and r['backend']=='cpu'
    for file,key in [('scripts/validate_tensor_binned_channels.py','script_sha256'),('scripts/build_binned_fixture.py','fixture_builder_sha256'),
        ('scripts/validate_binned_native_reference.py','native_reference_worker_sha256')]:assert sha(file)==r[key]
    for p,h in r['tensor_sources'].items():assert sha(p)==h
    for p,h in r['artifacts'].items():assert sha(p)==h
    h=r['hhvv'];c=r['complete_card'];assert h['closed_over_arrays']==c['closed_over_arrays']==0
    for field in ('validation_arrays','native_report','template_report','scalar_report','term_report'):assert sha(h[field])==h[field+'_sha256']
    native=json.loads(Path(h['native_report']).read_bytes());terms=json.loads(Path(h['term_report']).read_bytes())
    assert sha(terms['validation_arrays'])==terms['validation_arrays_sha256']
    refs=np.load(terms['validation_arrays'],allow_pickle=False);arrays=np.load(h['validation_arrays'],allow_pickle=False)
    np.testing.assert_array_equal(arrays['points'],refs['points'])
    np.testing.assert_array_equal(arrays['points'],[case['theta'] for case in native['cases']])
    assert arrays['native_channels'].shape==(26,50) and len(set(arrays['subtotal_channels']))==410
    assert arrays['channel_names'].tolist()==h['channels'] and arrays['subtotal_channels'].tolist()==h['subtotal_channels']
    subset_path=Path(h['validation_arrays']).parent/'hhvv_subset'
    from combine_jax.tensor.bundle import TensorBundle
    subset=TensorBundle.load(subset_path)
    assert subset.metadata['provenance']['all_card_channels_imported'] is False
    assert subset.metadata['channel_names']==h['subtotal_channels']
    b=subset.metadata['components']['binned'];assert len(b['program'])==h['density_batches'] and len(b['nll_program'])==h['nll_batches']
    assert len(b['histograms']['schedule'])==h['fast_batches'] and len(b['folded_queries'])==h['folded_queries']
    expected=[];subtotals=[]
    for i,case in enumerate(native['cases']):
        values={q['name']:q['combine'] for q in case['channels']}
        expected.append([values[n] for n in h['channels']])
        subtotals.append(sum(values[n] for n in h['subtotal_channels'])+np.sum(refs['native_constraints'][i])+subset.arrays['offset'])
    np.testing.assert_array_equal(arrays['native_channels'],expected)
    np.testing.assert_array_equal(arrays['native_subtotal'],subtotals)
    maxima={}
    for kind in ('channels','subtotal'):
        for backend in ('jax','numpy'):
            actual=arrays[backend+'_'+kind];reference=arrays['native_'+kind]
            assert actual.shape==reference.shape and np.all(np.isfinite(actual))
            delta=np.abs(actual-reference);assert np.all(delta<1e-6)
            errors=np.max(delta,axis=1) if kind=='channels' else delta
            field=kind if backend=='jax' else backend+'_'+kind
            np.testing.assert_array_equal(errors,[case[field]['max_absolute_error'] for case in h['cases']])
            maxima[backend+'_'+kind]=float(np.max(delta))
    assert sha(c['native_reference'])==c['native_reference_sha256'] and sha(c['workspace'])==c['workspace_sha256']
    reference=json.loads(Path(c['native_reference']).read_bytes())
    assert reference['status']=='passed' and reference['root_evaluation_errors']==0
    assert reference['worker_sha256']==r['native_reference_worker_sha256'] and reference['workspace_sha256']==c['workspace_sha256']
    assert reference['parameters']==c['parameter_names'] and len(reference['cases'])==len(c['cases'])==26
    for name in ('build_fixture_process','native_reference_process'):
        process=c[name];assert process['exit_code']==0 and sha(process['log'])==process['log_sha256']
    card_errors=[];gradient_errors=[];hessian_errors=[]
    for actual,ref in zip(c['cases'],reference['cases'],strict=True):
        assert actual['theta']==ref['theta'] and actual['native_nll']==ref['combine_nll']
        value=abs(actual['nll']-ref['combine_nll']);assert value==actual['nll_absolute_error'] and value<1e-6;card_errors.append(value)
        gradient=np.array(actual['gradient']);fd=np.array(ref['combine_fd_gradient'])
        relative=np.abs(gradient-fd)/np.maximum(np.abs(fd),1e-8)
        np.testing.assert_array_equal(relative,actual['gradient_relative_error']);assert np.all(relative<1e-4)
        assert np.max(np.abs(gradient-np.array(ref['jax_gradient'])))==actual['old_ad_gradient_max_error']
        gradient_errors.append(float(np.max(relative)))
        if 'hessian' in actual:
            delta=float(np.max(np.abs(np.array(actual['hessian'])-np.array(ref['fd_hessian']))))
            assert delta==actual['fd_hessian_max_error'] and delta<1e-6;hessian_errors.append(delta)
            assert np.max(np.abs(np.array(actual['hessian'])-np.array(ref['hessian'])))==actual['old_ad_hessian_max_error']
    assert len(hessian_errors)==3
    model_path=Path(c['workspace']).parents[1]/'complete_model'
    script='''import sys,numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import evaluate_model_numpy
m=TensorBundle.load(sys.argv[1]);value=float(evaluate_model_numpy(m,m.arrays['scalars__parameter_initial']))
assert np.isfinite(value)
assert not any(n in sys.modules for n in ('ROOT','jax','torch'))
print(value)
'''
    run=subprocess.run([sys.executable,'-c',script,str(model_path)],capture_output=True,text=True,check=True)
    assert abs(float(run.stdout)-c['cases'][0]['native_nll'])<1e-6
    failure_path=Path('reports/tensor_binned_channels_attempt1_checkpoint.json');failure=json.loads(failure_path.read_bytes())
    assert failure['status']=='native_process_terminated' and failure['exit_code']==129
    for p,digest in failure['artifacts'].items():assert sha(p)==digest
    baseline='af3dce4';unchanged={}
    for p in subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'src/combine_jax'],text=True).splitlines():
        if p.startswith('src/combine_jax/tensor/'):continue
        assert Path(p).read_bytes()==subprocess.check_output(['git','show',baseline+':'+p]);unchanged[p]=sha(p)
    testlog=Path('reports/tensor_binned_channels_tests.log');assert '77 passed, 2 warnings' in testlog.read_text()
    remaining=sorted({q['name'] for q in native['cases'][0]['channels']}-set(h['subtotal_channels']));assert len(remaining)==49
    result=dict(status='passed',report=str(path),report_sha256=sha(path),script_sha256=sha(__file__),
        hhvv=dict(points=26,new_channels=50,subtotal_channels=410,remaining_channels=remaining,maximum_errors=maxima),
        complete_card=dict(points=26,max_nll_absolute_error=max(card_errors),max_gradient_relative_error=max(gradient_errors),
            hessian_points=3,max_fd_hessian_error=max(hessian_errors),root_evaluation_errors=0,neutral_transport_without_frameworks=True),
        failed_attempt_checkpoint=str(failure_path),failed_attempt_checkpoint_sha256=sha(failure_path),
        tests=dict(passed=77,warnings=2,log=str(testlog),sha256=sha(testlog)),old_backend_baseline=baseline,old_backend_unchanged_files=unchanged,
        scope='Saved values, native point/source alignment, complete-card AD gates and transport. Full HHVV derivatives/fits/GPU performance remain unvalidated.')
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='old_backend_unchanged_files'},indent=2))


if __name__=='__main__':main()
