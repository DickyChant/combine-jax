"""Reject misleading fit speed comparisons using captured smoke reports."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('edm_timing_audit', ROOT / 'scripts/audit_tensor_edm_timing.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


@pytest.fixture
def captured():
    prefix = 'gpu_tensor_native_policy_smoke_counting_'
    return [json.loads((ROOT / 'reports' / (prefix + name + '.json')).read_text())
            for name in ('native', 'cpu_fit', 'cpu_fit_audit')]


def test_single_execution_ratio_and_explicit_scope(captured):
    native, tensor, _ = captured
    result = audit.timing_summary(*captured)
    assert result['ratios']['native_migrad_over_tensor_execution'] == (
        native['migrad_seconds'] / tensor['timings']['first_execute_seconds'])
    assert result['tensor_preparation_and_compile_seconds']['compile_or_cache_load_seconds'] > 0
    assert result['counts']['native']['numerical_gradient_evaluations'] == 5
    assert 'single runs' in result['scope']


@pytest.mark.parametrize('index,path,value', [
    (2, ('status',), 'failed_minimum_agreement'),
    (2, ('same_fit_state',), False),
    (2, ('minimum_nll_difference',), .00021),
    (2, ('same_point_nll_error',), .00001),
    (1, ('host',), 'another-node'),
    (1, ('cpu_affinity',), [0]),
    (1, ('float64',), False),
    (1, ('fit_result', 'corrected_estimated_edm'), .001),
    (1, ('first_fit_checkpoint', 'arrays_sha256'), 'different result'),
    (1, ('timings', 'warm_seconds'), [1.]),
    (1, ('timings', 'first_execute_seconds'), float('nan')),
    (0, ('fit', 'edm'), .001),
    (0, ('evaluation_counts', 'actual_nll_evaluations'), 1),
])
def test_no_ratio_for_failed_or_incomparable_fit(captured, index, path, value):
    data = copy.deepcopy(captured)
    target = data[index]
    for name in path[:-1]:
        target = target[name]
    target[path[-1]] = value
    with pytest.raises(AssertionError):
        audit.timing_summary(*data)


def test_gpu_requires_memory_capture_and_clean_monitor(captured):
    captured[1]['backend'] = 'gpu'
    with pytest.raises(AssertionError):
        audit.timing_summary(*captured)
    captured[1]['resources']['gpu'] = {'process_seen': True, 'error': 'NVML failed'}
    with pytest.raises(AssertionError):
        audit.timing_summary(*captured)


@pytest.fixture
def gpu_capture(captured):
    """Synthetic GPU resource fields on generic fit evidence; no physics claim."""
    tensor=captured[1];tensor['backend']='gpu'
    tensor['environment']['COMBINE_JAX_LCG_VIEW']='recorded-view'
    tensor['resources']['gpu']=dict(process_seen=False,process_peak_mib=None,error=None,
        samples=100,period_seconds=.05,device_total_mib=40960.,device_peak_mib=5591.,device_baseline_mib=955.)
    tensor['jax_allocator']=dict(devices=[dict(id=0,available=True,counters={'peak_bytes_in_use':3218020608})])
    return captured


def test_pid_namespace_does_not_turn_device_peak_into_process_memory(gpu_capture):
    result=audit.timing_summary(*gpu_capture)
    memory=result['gpu_memory']
    assert not memory['process_attribution_available']
    assert memory['sampled_process_peak_mib'] is None
    assert memory['sampled_assigned_device_peak_mib']==5591.
    assert memory['jax_peak_live_bytes']==3218020608
    assert 'not process memory' in memory['scope']
    assert result['ratios']['native_migrad_over_tensor_execution']>0


def test_available_process_peak_is_preserved(gpu_capture):
    gpu_capture[1]['resources']['gpu'].update(process_seen=True,process_peak_mib=4600.)
    assert audit.timing_summary(*gpu_capture)['gpu_memory']['sampled_process_peak_mib']==4600.


@pytest.mark.parametrize('field,value',[
    ('error','NVML error'),('samples',0),('period_seconds',float('nan')),
    ('process_peak_mib',0.),('device_peak_mib',float('nan')),
    ('device_peak_mib',41000.),('device_baseline_mib',6000.),('process_seen',True),
])
def test_invalid_gpu_memory_does_not_pass(gpu_capture,field,value):
    gpu_capture[1]['resources']['gpu'][field]=value
    with pytest.raises(AssertionError):audit.timing_summary(*gpu_capture)


@pytest.mark.parametrize('value',[None,[],[dict(available=False)],
    [dict(available=True,counters={'peak_bytes_in_use':float('nan')})]])
def test_missing_or_invalid_allocator_capture_does_not_pass(gpu_capture,value):
    gpu_capture[1]['jax_allocator']['devices']=value
    with pytest.raises((AssertionError,TypeError)):audit.timing_summary(*gpu_capture)


@pytest.fixture
def synthetic_asimov_eligibility(captured):
    """Exercise HHVV gate metadata using generic measurements, not a physics claim."""
    native, tensor, comparison = captured
    native['dataset'] = comparison['dataset'] = 'asimov_prefit'
    comparison['case'] = 'hhvv'
    assert 'workspace_sha256' not in native
    validation = dict(status='passed', nll_passed=True, dataset='asimov_prefit',
        points=[dict(index=i) for i in range(26)],
        workspace_sha256=native['input_workspace_sha256'],
        model_sha256=tensor['model_sha256'], backend=tensor['backend'])
    return captured, validation


def test_asimov_gate_uses_actual_native_input_workspace_field(synthetic_asimov_eligibility):
    captured, validation = synthetic_asimov_eligibility
    assert validation['workspace_sha256'] != captured[0]['prepared_workspace_sha256']
    result = audit.timing_summary(*captured, native_validation=validation)
    assert result['status'] == 'passed_same_minimum_single_run_timing'


@pytest.mark.parametrize('field,value', [
    ('workspace_sha256', 'another workspace'), ('model_sha256', {}), ('backend', 'gpu'),
    ('dataset', 'data_obs'), ('nll_passed', False),
    ('points', [dict(index=0) for _ in range(26)]),
    ('points', [dict(index=i) for i in range(25)]),
])
def test_asimov_validation_must_cover_exact_dataset_backend_and_points(synthetic_asimov_eligibility, field, value):
    captured, validation = synthetic_asimov_eligibility
    validation[field] = value
    with pytest.raises(AssertionError):
        audit.timing_summary(*captured, native_validation=validation)


def command_row(name,start,end):
    return dict(command=['benchmark','--output','reports/'+name+'.json'],status='completed',
        returncode=0,started_unix=start,finished_unix=end)


@pytest.mark.parametrize('native,tensor,expected',[
    ((1,2),(3,4),'native-first'),((1,2),(2,4),'native-first'),
    ((3,4),(1,2),'tensor-first'),((2,4),(1,2),'tensor-first')])
def test_timing_accepts_both_nonoverlapping_orders(native,tensor,expected):
    driver=dict(commands=[command_row('native',*native),command_row('tensor',*tensor)])
    assert audit.fit_command_order(driver,'native.json','tensor.json')==expected


@pytest.mark.parametrize('native,tensor',[
    ((1,3),(2,4)),((1,4),(2,3)),((2,2),(3,4)),((2,1),(3,4)),
    ((1,float('inf')),(3,4)),((1,2),(float('nan'),4))])
def test_timing_rejects_overlap_and_invalid_intervals(native,tensor):
    driver=dict(commands=[command_row('native',*native),command_row('tensor',*tensor)])
    with pytest.raises(AssertionError):audit.fit_command_order(driver,'native.json','tensor.json')


@pytest.mark.parametrize('defect',['duplicate','missing','failed','running'])
def test_timing_requires_one_successful_command_for_each_fit(defect):
    rows=[command_row('native',1,2),command_row('tensor',3,4)]
    if defect=='duplicate':rows.append(rows[0].copy())
    elif defect=='missing':rows.pop()
    elif defect=='failed':rows[0]['returncode']=1
    else:rows[0]['status']='running'
    with pytest.raises(AssertionError):audit.fit_command_order(dict(commands=rows),'native.json','tensor.json')
