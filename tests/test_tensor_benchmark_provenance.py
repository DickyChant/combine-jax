"""Historical measurements must match their producer, even after source changes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('tensor_benchmark_provenance',
    ROOT/'scripts/tensor_benchmark_provenance.py')
provenance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(provenance)


@pytest.fixture
def capture():
    return json.loads((ROOT/'reports/gpu_tensor_t4_hhvv_hhvv_cpu_jit_nll.json').read_text())


def test_historical_capture_checks_exact_committed_producer(capture):
    verified = provenance.verified_source_record(capture, 'scripts/benchmark_tensor.py', '6abb8f2')
    assert verified['source_commit'].startswith('6abb8f2')
    assert verified['lookup'] == 'git_commit'


def test_changed_current_sources_cannot_relabel_a_historical_capture(capture):
    with pytest.raises(AssertionError, match='source (file set differs|hash mismatch)'):
        provenance.verified_source_record(capture, 'scripts/benchmark_tensor.py')


@pytest.mark.parametrize('field', ['sources', 'script_sha256', 'common_script_sha256'])
def test_historical_revision_does_not_waive_tampered_hashes(capture, field):
    changed = copy.deepcopy(capture)
    if field == 'sources':
        changed[field]['src/combine_jax/tensor/model.py'] = '0'*64
    else:
        changed[field] = '0'*64
    with pytest.raises(AssertionError, match='source hash mismatch'):
        provenance.verified_source_record(changed, 'scripts/benchmark_tensor.py', '6abb8f2')


def test_missing_source_cannot_be_omitted_from_historical_provenance(capture):
    capture['sources'].pop('src/combine_jax/tensor/model.py')
    with pytest.raises(AssertionError, match='source file set differs'):
        provenance.verified_source_record(capture, 'scripts/benchmark_tensor.py', '6abb8f2')
