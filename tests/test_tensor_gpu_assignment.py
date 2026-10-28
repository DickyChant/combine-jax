"""Regression for actual Condor UUID prefixes and safe memory attribution."""
import importlib.util
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1]/'scripts/tensor_benchmark_common.py'
spec = importlib.util.spec_from_file_location('tensor_benchmark_common', path)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


def test_condor_prefix_resolves_to_the_assigned_physical_uuid():
    assigned = 'GPU-47e13020-0123-4567-89ab-0123456789ab'
    devices = ['GPU-deadbeef-0123-4567-89ab-0123456789ab', assigned]
    assert common.assigned_gpu_uuid('GPU-47e13020', devices) == assigned
    assert common.assigned_gpu_uuid(assigned, devices) == assigned


def test_ambiguous_or_unavailable_assignment_cannot_select_another_device():
    with pytest.raises(RuntimeError, match='matches 2'):
        common.assigned_gpu_uuid('GPU-ab', ['GPU-abc', 'GPU-abd'])
    with pytest.raises(RuntimeError, match='matches 0'):
        common.assigned_gpu_uuid('GPU-missing', ['GPU-abc'])
    with pytest.raises(ValueError, match='full-device'):
        common.assigned_gpu_uuid('MIG-123', ['GPU-abc'])
