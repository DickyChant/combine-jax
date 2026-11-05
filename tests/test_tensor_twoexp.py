import subprocess
import sys

import jax
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.binned_channels import pack_binned_channels, make_binned_function
from test_tensor_binned_channels import bind_model, check_model
from test_twoexp import make_twoexp


@pytest.mark.parametrize('offset,zero', [('default',False),('none',False),('custom',False),('custom',True)])
def test_twoexp_tensor_native_nll_ad_hvp_and_transport(root, tmp_path, offset, zero):
    root=load_caching_nll();fixture=make_twoexp(root,offset)
    if zero:
        for variable in fixture['variables'][:2]:variable.setVal(0.)
    coordinates={'tx':np.array([110.5,118.,125.,139.,150.,159.])}
    model=bind_model(root,fixture['pdf'],[fixture['x']],coordinates);check_model(model)
    rows=[{**model['rows'][0],'name':'copy'+str(i)} for i in range(3)]
    bundle=pack_binned_channels(model['nodes'],rows,model['scalar'])
    buffers=jax.device_put(dict(bundle.arrays));theta=model['old'].initial
    direction=np.random.default_rng(127).normal(size=len(theta))
    sf=model['sf'];scalar_buffers=model['arrays']['s']
    functions={policy:(lambda t,bf=make_binned_function(bundle,gradient=policy,row_chunk_size=2):
                        bf(sf(t,scalar_buffers),buffers).sum())
               for policy in ('ad','structural','structural_sequential')}
    expected=jax.jit(jax.value_and_grad(functions['ad']))(theta)
    reference_hvp=jax.jit(lambda t:jax.jvp(jax.grad(functions['ad']),(t,),(direction,))[1])(theta)
    for policy in ('structural','structural_sequential'):
        actual=jax.jit(jax.value_and_grad(functions[policy]))(theta)
        for a,e in zip(actual,expected,strict=True):np.testing.assert_allclose(a,e,rtol=1e-11,atol=1e-11)
        actual_hvp=jax.jit(lambda t:jax.jvp(jax.grad(functions[policy]),(t,),(direction,))[1])(theta)
        np.testing.assert_allclose(actual_hvp,reference_hvp,rtol=1e-10,atol=1e-10)
    model['scalar'].save(tmp_path/'scalar');bundle.save(tmp_path/'binned')
    script='''import sys,numpy as np
from pathlib import Path
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.binned_channels import evaluate_binned_channels
p=Path(sys.argv[1]);s=TensorBundle.load(p/'scalar');b=TensorBundle.load(p/'binned')
assert np.all(np.isfinite(evaluate_binned_channels(b,evaluate_numpy(s,s.arrays['parameter_initial']))))
assert not any(n in sys.modules for n in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)
