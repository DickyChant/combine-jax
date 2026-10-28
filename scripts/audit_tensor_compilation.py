"""Audit fresh-process cache reuse and preserve the small probe artifacts."""
import hashlib
import json
from pathlib import Path
import re
import shutil

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    log=Path('reports/tensor_compilation_tests.log');output=Path('reports/tensor_compilation.json')
    assert not output.exists();text=log.read_text();assert '1 passed' in text and 'FAILED' not in text
    matches=re.findall(r'CACHE_VALIDATION_JSON=(\{[^\n]*\})',text);assert len(matches)==1
    raw=json.loads(matches[0]);cases=raw['cases'];assert raw['status']=='passed' and len(cases)==4
    assert [(c['channels'],c['scale']) for c in cases]==[(4,1.),(4,1.4),(8,1.),(8,1.4)]
    directories=set()
    for i,c in enumerate(cases):
        observed=np.linspace(8.,12.,c['channels'])*c['scale']
        nll=float(np.sum(12.-observed-observed*np.log(12./observed))+.25*c['scale'])
        assert abs(c['nll']-nll)<1e-12 and abs(c['nll']-c['numpy_nll'])<1e-12
        assert c['gradient_tolerance']==1e-7 and c['projected_gradient_max']<=1e-7
        np.testing.assert_allclose(c['values'],[np.mean(observed)],atol=2e-7*np.mean(observed)/len(observed),rtol=0.)
        np.testing.assert_allclose(c['errors'],np.array(c['values'])/np.sqrt(np.sum(observed)),atol=1e-12,rtol=0.)
        cache=Path(c['cache']['directory']);directories.add(cache)
        assert {p.name:sha(p) for p in cache.glob('*-cache')}==c['cache_file_sha256']
        if i%2:
            before=cases[i-1]
            assert c['cache']==before['cache'] and c['cache_file_sha256']==before['cache_file_sha256']
            assert len(c['cache_hit_lines'])==2
            for name in ('jit_nll','jit_fit'):
                assert any(f"cache hit for '{name}'" in s for s in c['cache_hit_lines'])
            assert c['nll']!=before['nll'] and c['values']!=before['values']
        else:assert not c['cache_hit_lines']
    assert len(directories)==2
    roots={p.parent.parent for p in directories};assert len(roots)==1
    evidence=Path('build/tensor_compilation_cache_evidence');assert not evidence.exists()
    shutil.copytree(roots.pop(),evidence)
    files=sorted(p for p in evidence.rglob('*') if p.is_file())
    sources=list(Path('src/combine_jax/tensor').glob('*.py'))+[Path('scripts/check_tensor_cache_worker.py'),Path('tests/test_tensor_compilation.py')]
    initial=Path('reports/tensor_compilation_fit_probe.json');failure=json.loads(initial.read_bytes())
    assert failure['status']=='line_search_failed' and failure['result']['status']==2
    report=dict(status='passed',script_sha256=sha(__file__),test_log=str(log),test_log_sha256=sha(log),
        tensor_and_probe_sources={str(p):sha(p) for p in sources},cases=cases,evidence=str(evidence),
        evidence_sha256={str(p):sha(p) for p in files},backend='cpu',float64=True,
        scope='Four fresh JAX processes; cold/warm pairs for two neutral Poisson layouts. Both NLL and whole-fit executables hit disk cache with changed runtime observations and offset. Cache file hashes are unchanged across each pair.',
        strict_tolerance_failure=str(initial),strict_tolerance_failure_sha256=sha(initial),
        limitations=['Cache correctness only; no compile or fit speed benchmark.',
            'At gradient tolerance 1e-8, the eight-channel fit returned line_search_failed at 8.428e-8; it is not accepted at that tolerance.',
            'The 1e-7 target is the already accepted native-card optimizer setting; native-card gates are unchanged.',
            'HHVV cache reuse, compile-time scaling, GPU/JAX 0.10.1 compatibility and batch-job distribution remain unmeasured.'])
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('passed: 2 layouts, 4 processes, 4 executable cache hits; numerical outputs checked',flush=True)


if __name__=='__main__':main()
