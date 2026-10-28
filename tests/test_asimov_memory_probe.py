"""Memory-only worker construction cannot silently launch a fit campaign."""
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_asimov_memory_probe import benchmark_arguments
from submit_asimov_memory_probes import worker_text, submit_text


@pytest.mark.parametrize('family', ['A100', 'H100'])
def test_worker_keeps_transfer_contract_and_excludes_fit_hosts(family):
    worker = worker_text(Path('scripts/condor_tensor_edm.sh').read_text(), family)
    assert 'scripts/run_tensor_edm_campaign.py' not in worker
    assert worker.count('scripts/run_asimov_memory_probe.py --family ' + family) == 1
    assert '--policies ad structural structural_sequential' in worker
    assert 'trap finish EXIT' in worker and 'runtime_patch.tar' in worker
    submit = submit_text(Path('scripts/condor_tensor_edm.sub').read_text(), family, ['running.cern.ch'])
    assert 'request_gpus = 1' in submit and 'request_cpus = 8' in submit
    assert 'request_memory = 12GB' in submit and '!regexp("MIG", GPUs_DeviceName)' in submit
    assert f'regexp("{family}", GPUs_DeviceName)' in submit
    assert '(TARGET.Machine =!= "running.cern.ch")' in submit
    assert '(TARGET.Machine =!= "running")' in submit


@pytest.mark.parametrize('backend', ['cpu', 'gpu'])
@pytest.mark.parametrize('policy', ['ad', 'structural', 'structural_sequential'])
def test_commands_only_measure_gradients_at_identical_saved_points(backend, policy):
    args = benchmark_arguments(dict(model='model', points='points.npz'), backend, policy, 'result.json', 8.)
    assert args[args.index('--operation') + 1] == 'value_gradient'
    assert args[args.index('--points') + 1] == 'points.npz'
    assert args[args.index('--binned-gradient') + 1] == policy
    assert args[args.index('--compiler-memory-limit-gib') + 1] == '8.0'
    assert not any(arg.startswith('--fit') for arg in args)


def test_invalid_family_or_hostname_rejected():
    with pytest.raises(ValueError):
        worker_text('', 'T4')
    with pytest.raises(ValueError):
        submit_text('', 'H100', ['host" || true'])
