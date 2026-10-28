"""Verify complete HHVV fit metadata and unchanged accepted likelihood buffers."""
import ast
import hashlib
import json
from pathlib import Path
import pickle
import re
import subprocess
import sys

import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info,pack_parameter_info,fit_coordinates


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    path=Path('reports/tensor_hhvv_fit_import.json');output=Path('reports/tensor_parameter_import_audit.json');assert not output.exists()
    r=json.loads(path.read_bytes());assert r['status']=='passed' and r['root_evaluation_errors']==0
    assert r['script_sha256']==sha('scripts/import_hhvv_tensor.py')
    for p,h in {**r['tensor_sources'],**r['artifacts']}.items():assert sha(p)==h
    baseline=Path('reports/tensor_hhvv_compact_import.json');old=json.loads(baseline.read_bytes())
    assert old['status']=='passed' and r['source']==old['source'] and r['card']==old['card']
    for p,h in old['artifacts'].items():assert sha(p)==h
    before=TensorBundle.load(old['model']);after=TensorBundle.load(r['model'])
    assert {k:v for k,v in after.metadata.items() if k!='parameter_info'}==before.metadata
    added={'parameter_info__'+n for n in ('initial','lower','upper','error')}
    assert set(after.arrays)-set(before.arrays)==added and not set(before.arrays)-set(after.arrays)
    for n,a in before.arrays.items():
        b=after.arrays[n];assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes(),n
    assert sum(after.arrays[n].nbytes for n in added)==84992
    census=json.loads(Path(r['census']).read_bytes());export=census['export'];assert sha(export['path'])==export['sha256']
    # Trusted local v11 export, checksum certified by the structural census.
    with Path(export['path']).open('rb') as f:payload=pickle.load(f)
    expected=pack_parameter_info(payload['parameters']);info=model_parameter_info(after)
    assert info.metadata==expected.metadata
    for n,a in info.arrays.items():np.testing.assert_array_equal(a,expected.arrays[n])
    coordinates=fit_coordinates(info);assert len(coordinates['free_indices'])==2652 and len(coordinates['fixed'])==4
    previous=subprocess.run(['git','show','3217d9a:src/combine_jax/tensor/model.py'],check=True,capture_output=True).stdout
    a,b=ast.parse(previous),ast.parse(Path('src/combine_jax/tensor/model.py').read_text());assert len(a.body)==len(b.body)
    changed=[getattr(x,'name',None) for x,y in zip(a.body,b.body,strict=True) if ast.dump(x)!=ast.dump(y)]
    assert changed==['from_workspace']
    log=Path('reports/tensor_parameter_import_tests.log');text=log.read_text();assert '10 passed, 2 warnings' in text
    cards=[json.loads(s) for s in re.findall(r'FIT_VALIDATION_JSON=(\{[^\n]*\})',text)];assert len(cards)==2
    for c in cards:
        assert c['nll_absolute_error']<1e-6 and c['max_parameter_error_over_tolerance']<1 and c['max_error_relative_difference']<.01
        assert len(c['profiles'])==2 and all(p['status']==0 and p['nll_absolute_error']<1e-6 for p in c['profiles'])
    script='''import sys
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info,fit_coordinates
m=TensorBundle.load(sys.argv[1]);c=fit_coordinates(model_parameter_info(m))
assert len(c['free_indices'])==2652
assert not any(n in sys.modules for n in ('jax','ROOT','torch'))
print('neutral HHVV load: 2656 parameters, 2652 free')
'''
    neutral=subprocess.run([sys.executable,'-c',script,r['model']],check=True,capture_output=True,text=True)
    output.write_text(json.dumps(dict(status='passed',script_sha256=sha(__file__),import_report=str(path),import_report_sha256=sha(path),
        prior_import=str(baseline),prior_import_sha256=sha(baseline),prior_source_revision='3217d9a',changed_model_definitions=changed,
        unchanged_likelihood_buffers=len(before.arrays),unchanged_likelihood_buffer_bytes=sum(a.nbytes for a in before.arrays.values()),
        added_parameter_buffer_bytes=84992,parameter_metadata=info.metadata,free_parameters=2652,fixed_parameters=coordinates['fixed'],
        trusted_export=export,test_log=str(log),test_log_sha256=sha(log),tests=10,native_card_cases=cards,
        neutral_process_stdout=neutral.stdout.strip(),
        scope='One workspace extractor call, verified in counting/autoMCStats tests. Every HHVV likelihood metadata field and numeric buffer is unchanged; only fit metadata is added. Accepted likelihood/AD kernels are unmodified. No new full-fit performance result.'),indent=2,allow_nan=False)+'\n')
    print('passed',len(before.arrays),'unchanged buffers, 84992 metadata bytes, 2652 free parameters',flush=True)


if __name__=='__main__':main()
