"""The CPU ROOT importer must not inherit the GPU worker's CUDA selection."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.fixture
def stage_environment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path('scripts').resolve()))
    spec = importlib.util.spec_from_file_location('observed_orchestrator',
        Path('scripts/orchestrate_hhvv_mx3000_observed.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv('JAX_PLATFORMS', 'cuda')
    monkeypatch.setenv('JAX_PLATFORM_NAME', 'gpu')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-test')
    monkeypatch.setenv('ROOTSYS', '/inherited/cuda-view/root')
    monkeypatch.setenv('LD_LIBRARY_PATH', '/inherited/cuda-view/libs')
    return module.stage_environment


def test_import_executes_cpu_oracle_under_cuda_parent(stage_environment, counting_path):
    env = stage_environment()
    # Exercise the exact workspace-import path that calls jnp.asarray and used
    # to crash in a CPU-only venv before any GPU fit could begin.
    result = subprocess.run([sys.executable, '-c', '''
import json, sys
import jax
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace
assert jax.config.x64_enabled
assert all(d.platform == 'cpu' for d in jax.devices())
with open_workspace(sys.argv[1]) as workspace:
    model = from_workspace(workspace, 'tests/cards/counting.txt')
print(json.dumps(dict(platforms=[d.platform for d in jax.devices()],
                     channels=len(model.metadata['channel_names']))))
''', str(counting_path)], env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == dict(platforms=['cpu'], channels=2)


def test_fit_keeps_cuda_assignment_and_import_clears_foreign_root(stage_environment):
    for gpu, platform in ((False, 'cpu'), (True, 'cuda')):
        env = stage_environment(gpu=gpu)
        assert env['JAX_PLATFORMS'] == platform
        assert env['JAX_ENABLE_X64'] == 'true'
        assert env['CUDA_VISIBLE_DEVICES'] == 'GPU-test'
        assert not {'ROOTSYS','LD_LIBRARY_PATH','JAX_PLATFORM_NAME'} & env.keys()
