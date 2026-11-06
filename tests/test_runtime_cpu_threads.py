"""Real child processes check configuration before XLA initializes its pools."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def child(code, **environment):
    env = dict(os.environ)
    env.pop('COMBINE_JAX_CPU_THREADS', None)
    env.update(environment)
    return subprocess.run([sys.executable, '-c', code], env=env, capture_output=True,
                          text=True, timeout=60)


@pytest.mark.parametrize('explicit', [True, False])
def test_cpu_configuration_is_effective_before_jax(explicit):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip('Two allocated logical CPUs required')
    result = child('''
import json
from combine_jax.runtime.cpu_threads import configure_cpu_threads
receipt = configure_cpu_threads('cpu', '''+('2' if explicit else 'None')+''')
import jax
import jax.numpy as jnp
value, gradient = jax.jit(jax.value_and_grad(lambda x: jnp.sum(x*x)))(jnp.arange(16., dtype=jnp.float64))
print(json.dumps(dict(receipt=receipt, value=float(value), gradient=list(map(float, gradient)),
                     jax=jax.__version__, float64=jax.config.x64_enabled)))
''', COMBINE_JAX_CPU_THREADS='1' if explicit else '2',
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false --xla_gpu_force_compilation_parallelism=1')
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    receipt = report['receipt']
    assert receipt['requested'] == 2 and len(receipt['affinity']) == 2
    assert receipt['source'] == ('cli' if explicit else 'COMBINE_JAX_CPU_THREADS')
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS'):
        assert receipt['environment'][key] == '2'
    flags = receipt['environment']['XLA_FLAGS']
    assert '--xla_cpu_multi_thread_eigen=true' in flags and 'intra_op_parallelism_threads=2' in flags
    assert '--xla_gpu_force_compilation_parallelism=1' in flags
    assert report['float64'] and report['value'] == 1240.
    assert report['gradient'] == [float(2*i) for i in range(16)]


@pytest.mark.parametrize('statement,reason', [
    ("configure_cpu_threads('cpu', 0)", 'positive integer'),
    ("configure_cpu_threads('cpu', 1.5)", 'positive integer'),
    ("configure_cpu_threads('gpu', 2)", 'requires --device cpu'),
    ("configure_cpu_threads('cpu', len(os.sched_getaffinity(0))+1)", 'affinity allows only'),
    ("import numpy; configure_cpu_threads('cpu', 1)", 'fresh process'),
])
def test_invalid_or_late_configuration_fails(statement, reason):
    result = child('import os; from combine_jax.runtime.cpu_threads import configure_cpu_threads; '+statement)
    assert result.returncode != 0 and reason in result.stderr


def test_no_explicit_count_preserves_user_flags_and_affinity():
    result = child('''
import os, json
from combine_jax.runtime.cpu_threads import configure_cpu_threads
before=sorted(os.sched_getaffinity(0)); receipt=configure_cpu_threads('cpu')
assert receipt['affinity']==before
print(json.dumps(receipt))
''', XLA_FLAGS='--xla_cpu_multi_thread_eigen=true', OMP_NUM_THREADS='3')
    assert result.returncode == 0, result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['requested'] is None and receipt['environment']['OMP_NUM_THREADS']=='3'
    assert receipt['environment']['XLA_FLAGS']=='--xla_cpu_multi_thread_eigen=true'


def test_lcg_launcher_preserves_caller_configuration_across_view(tmp_path):
    prefix=tmp_path/'runtime';(prefix/'code').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'src'
    (prefix/'code/src').symlink_to(source, target_is_directory=True)
    view=tmp_path/'view.sh'
    view.write_text('export PATH="'+str(Path(sys.executable).parent)+':/usr/bin:/bin"\n'
                    'export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n'
                    'export XLA_FLAGS=--xla_cpu_multi_thread_eigen=false\n')
    env=dict(os.environ, XLA_FLAGS='--xla_cpu_enable_fast_math=false',
             OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
             COMBINE_JAX_CPU_THREADS='2')
    code='''
import os, json
assert os.environ['XLA_FLAGS']=='--xla_cpu_enable_fast_math=false'
assert os.environ['OMP_NUM_THREADS']=='2'
from combine_jax.runtime.cpu_threads import configure_cpu_threads
print(json.dumps(configure_cpu_threads('cpu')))
'''
    result=subprocess.run(['bash', str(source/'combine_jax/runtime/run_lcg.sh'), str(prefix),
                           str(view), '-c', code], env=env, capture_output=True, text=True, timeout=60)
    assert result.returncode==0, result.stderr
    receipt=json.loads(result.stdout)
    assert receipt['requested']==2 and len(receipt['affinity'])==2
    assert '--xla_cpu_enable_fast_math=false' in receipt['environment']['XLA_FLAGS']
