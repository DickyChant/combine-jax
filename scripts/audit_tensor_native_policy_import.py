"""Check a native-policy HHVV import against accepted likelihood buffers."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
import subprocess

import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
from tensor_benchmark_common import save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--import-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-revision',required=True)
    args=parser.parse_args();assert not args.output.exists()
    report=json.loads(args.import_report.read_text())
    assert report['status']=='passed' and report['fit_policy']=='combine_v11_default'
    for path,digest in report['artifacts'].items():assert sha(path)==digest,path
    for path,digest in report['tensor_sources'].items():
        content=subprocess.check_output(['git','show',args.source_revision+':'+path])
        assert hashlib.sha256(content).hexdigest()==digest,path
    baseline=Path('build/tensor_hhvv_fit_import/model')
    before=TensorBundle.load(baseline);after=TensorBundle.load(report['model'])
    old=model_parameter_info(before);new=model_parameter_info(after)
    assert before.metadata['parameter_names']==after.metadata['parameter_names']
    assert {k:v for k,v in before.metadata.items() if k!='parameter_info'}=={
        k:v for k,v in after.metadata.items() if k!='parameter_info'}
    numeric=[n for n in before.arrays if not n.startswith('parameter_info__')]
    assert set(numeric)=={n for n in after.arrays if not n.startswith('parameter_info__')}
    profile_bounds={'templates__profile_minimum':-np.inf,'templates__profile_maximum':np.inf}
    unchanged=[];changed=[]
    profile_names=after.metadata['components']['templates']['profile_names']
    removed=set(new.metadata['fit_policy']['nuisance_bounds_removed'])
    for name in numeric:
        a,b=before.arrays[name],after.arrays[name]
        assert a.dtype==b.dtype and a.shape==b.shape,name
        if a.tobytes()==b.tobytes():unchanged.append(name);continue
        assert name in profile_bounds,name
        expected=a.copy()
        for i,n in enumerate(profile_names):
            if n in removed:expected[i]=profile_bounds[name]
        np.testing.assert_array_equal(b,expected)
        changed.append(dict(name=name,changed_entries=int(np.sum(a!=b))))
    assert len(changed)==2 and all(r['changed_entries']==74 for r in changed)
    for key in ('initial','error'):np.testing.assert_array_equal(old.arrays[key],new.arrays[key])
    for key in ('lower','upper'):np.testing.assert_array_equal(old.arrays[key],new.arrays['stored_'+key])
    coordinates=fit_coordinates(new)
    bounds=json.loads(Path('reports/tensor_combine_effective_bounds_probe.json').read_text())
    assert old.metadata['names']==[r['name'] for r in bounds['parameters']]
    for i,row in enumerate(bounds['parameters']):
        # Zero-width entries are explicitly frozen in the tensor and CLI fit.
        if new.arrays['lower'][i]==new.arrays['upper'][i]:continue
        lo=row['effective_lower'] if row['effective_has_min'] else -np.inf
        hi=row['effective_upper'] if row['effective_has_max'] else np.inf
        assert (lo,hi)==(new.arrays['lower'][i],new.arrays['upper'][i]),row['name']
    explicit_changed=(old.arrays['lower']!=new.arrays['lower'])|(old.arrays['upper']!=new.arrays['upper'])
    assert int(sum(explicit_changed))==bounds['changed_parameters']==1213
    assert new.metadata['fit_policy']['mass_argument']==125.
    assert 'ROOT' not in sys.modules and 'jax' not in sys.modules and 'torch' not in sys.modules
    save_json(args.output,dict(status='passed_native_bounds_buffer_audit',script_sha256=sha(__file__),source_revision=args.source_revision,
        import_report=str(args.import_report),import_report_sha256=sha(args.import_report),
        baseline_hashes={str(baseline/n):sha(baseline/n) for n in ('model.json','arrays.npz')},
        unchanged_likelihood_buffers=len(unchanged),unchanged_likelihood_bytes=sum(before.arrays[n].nbytes for n in unchanged),
        changed_profile_bounds=changed,
        effective_bound_changes=int(sum(explicit_changed)),free_parameters=len(coordinates['free_indices']),fixed_parameters=coordinates['fixed'],
        fit_state=new.metadata['fit_policy'],native_policy_probe_sha256=sha('reports/tensor_combine_effective_bounds_probe.json'),
        scope='Complete reimport under native bounds. All likelihood metadata and all numeric buffers except the two 74-entry profiled-nuisance bound arrays are unchanged. Explicit fit bounds agree with the installed native method. The profiled bounds match the marked-nuisance policy; full native point validation remains required. No fit or timing claim.'))
    print('passed',len(unchanged),'identical likelihood buffers; two profile-bound arrays changed',flush=True)


if __name__=='__main__':main()
