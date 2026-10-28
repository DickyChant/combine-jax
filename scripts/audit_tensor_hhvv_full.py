"""Audit full-HHVV tensor import and saved native value comparisons."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from collections import defaultdict

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',default='reports/tensor_hhvv_full_values.json')
    parser.add_argument('--output',default='reports/tensor_hhvv_full_audit.json')
    parser.add_argument('--test-log',default='reports/tensor_grid_expressions_tests.log')
    parser.add_argument('--tests',type=int,default=115)
    parser.add_argument('--baseline',default='7ee10d4')
    parser.add_argument('--uncompacted-import',help='Also audit exact row selection and unchanged normalization buffers')
    args=parser.parse_args();path=Path(args.report);r=json.loads(path.read_bytes())
    output=Path(args.output);assert not output.exists()
    assert r['status']=='passed' and r['float64'] and r['backend']=='cpu' and r['closed_over_arrays']==0
    assert sha('scripts/validate_tensor_hhvv_full.py')==r['script_sha256']
    for field in ('import_report','native_report','terms_report','point_report','validation_arrays'):assert sha(r[field])==r[field+'_sha256']
    imp=json.loads(Path(r['import_report']).read_bytes());native=json.loads(Path(r['native_report']).read_bytes())
    terms=json.loads(Path(r['terms_report']).read_bytes());point=json.loads(Path(r['point_report']).read_bytes())
    assert imp['status']=='passed' and imp['root_evaluation_errors']==0 and imp['channels']==459
    assert sha('scripts/import_hhvv_tensor.py')==imp['script_sha256']
    assert imp['tensor_sources']==r.get('import_tensor_sources',r['tensor_sources'])
    assert imp['tensor_sources'].keys()==r['tensor_sources'].keys()
    updates={p:dict(import_sha256=h,runtime_sha256=r['tensor_sources'][p]) for p,h in imp['tensor_sources'].items() if h!=r['tensor_sources'][p]}
    assert updates==r.get('runtime_source_updates',{})
    assert sha(imp['source']['path'])==imp['source']['sha256'] and sha(imp['card']['path'])==imp['card']['sha256']
    for p,digest in {**r['tensor_sources'],**imp['artifacts']}.items():assert sha(p)==digest
    from combine_jax.tensor.bundle import TensorBundle
    model=TensorBundle.load(imp['model']);assert model.structure_key==imp['structure_key']==r['structure_key']
    assert model.metadata['provenance']['all_card_channels_imported'] and model.metadata['provenance']['card_sha256']==imp['card']['sha256']
    assert model.metadata['provenance']['discrete_choices']==point['discrete_choices'] and model.metadata['provenance']['channel_masks']==point['channel_masks']
    assert model.metadata['parameter_names']==native['parameter_names']==r['parameter_names']==point['parameter_names']
    assert model.metadata['channel_names']==r['channel_names'] and len(set(r['channel_names']))==459
    assert len(r['parameter_names'])==2656 and len(r['constraint_names'])==2124 and len(r['profile_names'])==74
    np.testing.assert_array_equal(model.arrays['scalars__parameter_initial'],native['initial'])
    a=np.load(r['validation_arrays'],allow_pickle=False);assert a['jax_channels'].shape==(26,459)
    assert sha(terms['validation_arrays'])==terms['validation_arrays_sha256'];refs=np.load(terms['validation_arrays'],allow_pickle=False)
    np.testing.assert_array_equal(a['points'],refs['points']);np.testing.assert_array_equal(a['points'],[q['theta'] for q in native['cases']])
    for kind in ('channel','constraint','profile'):assert a[kind+'_names'].tolist()==r[kind+'_names']
    order=[refs['constraint_names'].tolist().index(n) for n in r['constraint_names']]
    np.testing.assert_array_equal(a['native_constraints'],refs['native_constraints'][:,order]);np.testing.assert_array_equal(a['native_profiles'],refs['profile_values'])
    np.testing.assert_array_equal(a['native_nll'],[q['combine_nll'] for q in native['cases']])
    for i,case in enumerate(native['cases']):
        values={q['name']:q['combine'] for q in case['channels']};np.testing.assert_array_equal(a['native_channels'][i],[values[n] for n in r['channel_names']])
    maxima={}
    for kind in ('channels','constraints','profiles','nll'):
        actual=a['jax_'+kind];expected=a['native_'+kind];delta=np.abs(actual-expected)
        assert actual.shape==expected.shape and np.all(np.isfinite(actual))
        tolerance=1e-10 if kind=='profiles' else 1e-6;assert np.all(delta<tolerance)
        perpoint=np.max(delta,axis=1) if delta.ndim==2 else delta
        np.testing.assert_array_equal(perpoint,[q[kind]['max_absolute_error'] for q in r['cases']]);maxima[kind]=float(np.max(delta))
    delta=np.abs(a['numpy_nll']-a['native_nll']);assert np.all(np.isfinite(a['numpy_nll'])) and np.all(delta<1e-6)
    np.testing.assert_array_equal(delta,[q['numpy_nll']['max_absolute_error'] for q in r['cases']]);maxima['numpy_nll']=float(np.max(delta))
    for backend in ('native','jax'):
        assembled=np.sum(a[backend+'_channels'],axis=1)+np.sum(a[backend+'_constraints'],axis=1)+model.arrays['offset']
        assert np.max(np.abs(assembled-a[backend+'_nll']))<1e-6
    assert point['status']=='point_checked' and all(q['roofit_evaluation_errors']==0 for q in point['probes'])
    q=r['native_point'];assert q['theta']==point['point'] and q['native_nll']==point['canonical_candidate_nll']
    for backend in ('jax','numpy'):
        delta=abs(q[backend+'_nll']-q['native_nll']);assert delta==q[backend]['max_absolute_error']<1e-6
    b=model.metadata['components']['binned'];grids=b['analytic_grid_queries']
    assert len(grids)==imp['grid_queries'] and sum(q['grid_rows'] for q in grids)==imp['grid_rows']
    assert sum(q['data_rows'] for q in grids)==imp['uncompressed_grid_rows'] and sum(q['folded'] for q in grids)==imp['folded_grid_queries']
    buffer_groups=defaultdict(int)
    for name,array in model.arrays.items():
        group=name.split('__',1)[0]
        if group=='binned':
            suffix=name.split('__',1)[1]
            group+='.'+('histograms' if suffix.startswith('histograms__') else 'queries' if suffix.startswith('queries__')
                else 'density_maps_and_data' if suffix.startswith('program_') else 'nll_maps_and_data' if suffix.startswith('nll_') else 'work_buffers_and_bindings')
        buffer_groups[group]+=array.nbytes
    assert sum(buffer_groups.values())==imp['buffer_bytes']
    compaction={}
    if args.uncompacted_import:
        from audit_tensor_shape_compaction import audit_compaction
        compaction=audit_compaction(model,Path(args.uncompacted_import))
    script='''import sys,numpy as np
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import evaluate_model_numpy
m=TensorBundle.load(sys.argv[1]);assert m.metadata['provenance']['all_card_channels_imported']
v=float(evaluate_model_numpy(m,m.arrays['scalars__parameter_initial']));assert np.isfinite(v)
assert not any(n in sys.modules for n in ('ROOT','jax','torch'))
print(v)
'''
    # Release the large parent artifact before evaluating in a fresh neutral process.
    del model
    run=subprocess.run([sys.executable,'-c',script,imp['model']],check=True,capture_output=True,text=True)
    assert abs(float(run.stdout)-a['native_nll'][0])<1e-6
    baseline=args.baseline;unchanged={}
    for p in sorted(Path('src/combine_jax').glob('*.py')):
        old=subprocess.run(['git','show',f'{baseline}:{p}'],check=True,capture_output=True).stdout
        assert p.read_bytes()==old;unchanged[str(p)]=sha(p)
    testlog=Path(args.test_log);assert f'{args.tests} passed, 2 warnings' in testlog.read_text()
    report=dict(status='passed',report=str(path),report_sha256=sha(path),script_sha256=sha(__file__),
        hhvv=dict(channels=459,parameters=2656,constraints=2124,profiles=74,points=26,maximum_errors=maxima,native_point=r['native_point']),
        coordinate_grids=dict(queries=imp['grid_queries'],folded=imp['folded_grid_queries'],rows=imp['grid_rows'],original_rows=imp['uncompressed_grid_rows']),
        buffer_bytes_by_component=dict(buffer_groups),compressed_array_bytes=Path(imp['model'],'arrays.npz').stat().st_size,
        neutral_transport=True,tests=dict(passed=args.tests,warnings=2,log=str(testlog),sha256=sha(testlog)),
        old_backend_baseline=baseline,old_backend_unchanged_files=unchanged,
        compaction=compaction,runtime_source_updates=updates,
        timing_scope='Standalone full NLL compilation is separate from diagnostics. Import/validation are not warm, derivative, fit or CPU/GPU comparison benchmarks.',
        scope='Complete HHVV value acceptance, including saved native point; full HHVV derivatives, device fits, uncertainties/profiles and new GPU benchmarks remain pending.')
    output.write_text(json.dumps(report,indent=2)+'\n');print('passed',maxima,q['jax']['max_absolute_error'],flush=True)


if __name__=='__main__':main()
