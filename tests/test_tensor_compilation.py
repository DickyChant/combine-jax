"""Fresh-process cache hits must preserve new runtime data and fit results."""
import hashlib
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys

import numpy as np

from combine_jax.graph import Parameter
from combine_jax.tensor.scalar import import_scalars
from combine_jax.tensor.constraints import pack_constraints
from combine_jax.tensor.counting import pack_counting_channels
from combine_jax.tensor.model import compose_counting
from combine_jax.tensor.parameter_info import attach_parameter_info,pack_parameter_info


def poisson_model(channels):
    parameters=[Parameter('rate',12.,.01,100.,1.,'poi')]
    nodes={'rate':('RooRealVar',(),dict(value=12.,constant=False)),
        'x':('RooRealVar',(),dict(value=.5,minimum=0.,maximum=1.,constant=False)),
        'uniform':('RooUniform',('x',),{}),
        'pdf':('RooAddPdf',('uniform','rate'),dict(extended=True,pdfs=['uniform'],coefs=['rate']))}
    scalars=import_scalars(nodes,parameters,['rate'],fixed_value=lambda n:nodes[n][2]['value'],observables=['x'])
    constraints=pack_constraints(nodes,[],{},scalars)
    rows=[dict(name=f'channel_{i}',pdf='pdf',observables=['x'],weights=np.array([8.+4.*i/(channels-1)])) for i in range(channels)]
    counting=pack_counting_channels(nodes,rows,scalars)
    return attach_parameter_info(compose_counting(scalars,constraints,counting),pack_parameter_info(parameters))


def test_process_cache_hits_after_relocation_runtime_changes_and_layout_separation(tmp_path):
    root=Path(__file__).resolve().parents[1];cache=tmp_path/'cache';cases=[];keys=[]
    for count in (4,8):
        model=poisson_model(count);artifact=tmp_path/f'model_{count}';model.save(artifact);keys.append(model.structure_key)
        for scale in (1.,1.4):
            relocated = tmp_path/f'relocated_{count}'
            if scale != 1.:
                shutil.copytree(cache, relocated)
            active_cache = cache if scale == 1. else relocated
            run=subprocess.run([sys.executable,str(root/'scripts/check_tensor_cache_worker.py'),str(artifact),str(active_cache),str(scale)],
                capture_output=True,text=True,cwd=root,env={**os.environ,'JAX_LOG_COMPILES':'1'})
            (tmp_path/f'worker_{count}_{scale}.log').write_text(run.stdout+run.stderr)
            assert run.returncode==0,run.stdout+run.stderr
            result=json.loads(run.stdout.strip().splitlines()[-1]);assert result['status']=='passed'
            result['cache_hit_lines']=[s for s in run.stderr.splitlines() if 'Persistent compilation cache hit for' in s
                and any(f"'{name}'" in s for name in ('jit_nll','jit_fit'))]
            assert result['cache']['structure_key']==model.structure_key
            assert result['cache']['auxiliary_xla_caches']=='none'
            namespace=Path(result['cache']['directory'])
            hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in namespace.glob('*-cache')}
            assert any(n.startswith('jit_nll-') for n in hashes) and any(n.startswith('jit_fit-') for n in hashes)
            if scale==1.:
                assert not result['cache_hit_lines'];before=hashes
            else:
                assert len(result['cache_hit_lines'])==2 and hashes==before
                assert result['nll']!=cases[-1]['nll'] and result['values']!=cases[-1]['values']
            result['cache_file_sha256']=hashes;cases.append(result)
    assert keys[0]!=keys[1]
    print('CACHE_VALIDATION_JSON='+json.dumps(dict(status='passed',cases=cases),allow_nan=False))
