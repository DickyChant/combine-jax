"""A failed native minimization must not erase independent tensor evidence."""
import copy
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tensor_edm_campaign as campaign
from tensor_benchmark_common import sha


@pytest.fixture
def native_state():
    policy=dict(policy='combine_v11_default', fixed_real_values={'MH':125.})
    dataset=dict(name='asimov_prefit', identity={'sha256':'synthetic Asimov identity'})
    metadata=dict(parameter_info=dict(fit_policy=policy), provenance=dict(dataset=dataset))
    native=dict(case='hhvv',dataset='asimov_prefit',status='failed',manifest_sha256='manifest',
        input_workspace_sha256='workspace',fit_state=policy,preflight=dict(status='passed',
            returncode=0,initial_state=policy,dataset=dataset),root_version='6.40.00',combine_version='11.0.0',
        fit_scope='migrad',stopping_rule=dict(type='Minuit2',algorithm='Migrad',strategy=1,
            tolerance=.1,error_definition=.5,raw_nll_edm_target=1e-4))
    return native,metadata


def outcome(native_state, code=1):
    native,metadata=native_state
    return campaign.hhvv_native_outcome(native,metadata,manifest_sha256='manifest',
        workspace_sha256='workspace',returncode=code)


def test_failed_fit_preserves_passed_start_but_cannot_establish_a_minimum(native_state):
    result=outcome(native_state)
    assert result['native_state_gate']=='passed'
    assert not result['minimum_comparison_eligible']


@pytest.mark.parametrize('path,value',[
    (('dataset',),'data_obs'),(('manifest_sha256',),'different'),(('input_workspace_sha256',),'different'),
    (('status',),'running'),(('preflight','status'),'failed'),(('preflight','returncode'),1),
    (('preflight','dataset','identity'),{'sha256':'different'}),
    (('preflight','initial_state'),{'policy':'wrong'}),
    (('root_version',),'6.32.00'),(('combine_version',),'10.3.3'),
    (('fit_scope',),'migrad_hesse_minos'),(('stopping_rule','tolerance'),.2),
    (('interruption_signal',),15),(('exception',),'Traceback: KeyboardInterrupt'),
])
def test_wrong_problem_failed_preflight_and_cancellation_stop(native_state,path,value):
    native,metadata=native_state
    # The native report and model metadata must be independently mutable here.
    native=copy.deepcopy(native)
    target=native
    for key in path[:-1]:target=target[key]
    target[path[-1]]=value
    with pytest.raises(AssertionError):outcome((native,metadata))


def test_signal_exit_never_starts_measurements(native_state):
    with pytest.raises(AssertionError,match='Cancellation'):outcome(native_state,code=-15)


@pytest.mark.parametrize('edm,expected',[(9.99e-5,True),(1e-4,False),(-1.,False),(float('nan'),False)])
def test_numeric_edm_gate_is_not_replaced_by_a_reported_boolean(native_state,edm,expected):
    native,_=native_state
    native.update(status='measured_fit_pending_agreement',fit=dict(status=0,
        edm_target_met=True,edm=edm,roofit_evaluation_errors=0))
    assert outcome(native_state,code=0)['minimum_comparison_eligible'] is expected


@pytest.mark.parametrize('condition', ['failed_fit','bad_edm','passed_fit','failed_preflight','cancelled'])
@pytest.mark.parametrize('fit_order',['native-first','tensor-first'])
def test_full_campaign_records_remaining_measurements_without_retry(monkeypatch,tmp_path,native_state,condition,fit_order):
    monkeypatch.chdir(tmp_path)
    import tensor_benchmark_dataset
    monkeypatch.setattr(tensor_benchmark_dataset,'benchmark_dataset',lambda *a,**k:'asimov_prefit')
    monkeypatch.setattr(campaign.os,'sched_setaffinity',lambda *a:None)
    monkeypatch.setattr(signal,'signal',lambda *a:None)
    Path('reports').mkdir();Path('build/tensor_benchmark_inputs').mkdir(parents=True)
    Path('build/edm_runtime_manifest.json').write_text(json.dumps(dict(sources={},source_revision='test')))
    native,metadata=native_state
    native=copy.deepcopy(native)
    manifest={}
    for case in ('counting','automcstats','hhvv'):
        folder=tmp_path/case;folder.mkdir()
        (folder/'model.json').write_text(json.dumps(dict(metadata=metadata)))
        workspace=folder/'input.root';workspace.write_bytes(b'synthetic input identity')
        manifest[case]=dict(model=str(folder),workspace=str(workspace),points='unused-points.npz')
    manifest_path=Path('build/tensor_benchmark_inputs/manifest.json')
    manifest_path.write_text(json.dumps(manifest))
    native.update(manifest_sha256=sha(manifest_path),input_workspace_sha256=sha(manifest['hhvv']['workspace']))
    if condition in ('passed_fit','bad_edm'):
        native.update(status='measured_fit_pending_agreement',fit=dict(status=0,edm_target_met=True,
            edm=1e-5 if condition=='passed_fit' else 1e-4,roofit_evaluation_errors=0))
    if condition=='failed_preflight':native['preflight']['status']='failed'
    if condition=='cancelled':native['interruption_signal']=15
    commands=[]
    class Child:
        pid=123
        def __init__(self,command,**kwargs):
            self.returncode=0;commands.append(command)
            output=Path(command[command.index('--output')+1])
            if 'scripts/verify_tensor_gpu.py' in command:
                output.write_text(json.dumps(dict(devices=[dict(kind='NVIDIA H100 NVL')])))
            if 'scripts/benchmark_combine_fit.py' in command and command[command.index('--case')+1]=='hhvv':
                if '--preflight-only' in command:
                    preflight=copy.deepcopy(native)
                    preflight.pop('interruption_signal',None)
                    preflight['status']='passed_preflight_only'
                    output.write_text(json.dumps(preflight))
                    self.returncode=1 if condition=='failed_preflight' else 0
                else:
                    output.write_text(json.dumps(native))
                    self.returncode=0 if condition in ('passed_fit','bad_edm') else 1
            if 'scripts/benchmark_tensor.py' in command and command[command.index('--operation')+1]=='fit':
                output.write_text(json.dumps(dict(status='measured_pending_validation',operation='fit')))
        def wait(self):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['campaign','--hhvv-fit-order',fit_order,'--gpu-hessian-batch-size','1'])
    stop=condition in ('failed_preflight','cancelled')
    if stop:
        with pytest.raises(AssertionError):campaign.main()
    else:campaign.main()
    native_fits=[c for c in commands if 'scripts/benchmark_combine_fit.py' in c and c[c.index('--case')+1]=='hhvv' and '--preflight-only' not in c]
    preflights=[c for c in commands if '--preflight-only' in c]
    assert len(preflights)==(fit_order=='tensor-first')
    assert len(native_fits)==(0 if fit_order=='tensor-first' and condition=='failed_preflight' else 1), 'No automatic native-fit retry'
    tensor=[c for c in commands if 'scripts/benchmark_tensor.py' in c and c[c.index('--model')+1]==manifest['hhvv']['model']]
    fits=[c for c in tensor if c[c.index('--operation')+1]=='fit']
    hessians=[c for c in tensor if c[c.index('--operation')+1]=='hessian']
    comparisons=[c for c in commands if 'scripts/audit_tensor_edm_comparison.py' in c and c[c.index('--case')+1]=='hhvv']
    expected_fits=2 if not stop or (fit_order=='tensor-first' and condition=='cancelled') else 0
    assert len(fits)==expected_fits
    assert len(hessians)==(0 if stop else 2)
    assert all(c[c.index('--fit-stopping-criterion')+1]=='edm' and '--single-fit' in c for c in fits)
    assert all(c[c.index('--batch-size')+1]=='1' for c in fits+hessians)
    if fits:
        if fit_order=='tensor-first':
            assert commands.index(preflights[0])<min(commands.index(c) for c in fits)
            assert max(commands.index(c) for c in fits)<commands.index(native_fits[0])
        else:assert commands.index(native_fits[0])<min(commands.index(c) for c in fits)
    if hessians:assert commands.index(native_fits[0])<min(commands.index(c) for c in hessians)
    assert len(comparisons)==(2 if condition=='passed_fit' else 0)
    report=json.loads(Path('reports/gpu_tensor_edm_driver.json').read_text())
    assert report['status']==('failed' if stop else 'completed_pending_audit' if condition=='passed_fit' else 'partial_failures_pending_audit')
    if not stop:
        assert report['hhvv_native_state_gate']=='passed'
        assert report['hhvv_native_fit_outcome']['minimum_comparison_eligible']==(condition=='passed_fit')
        assert report['hhvv_fit_order']==fit_order
        assert report['hhvv_gpu_hessian_batch_size']==1
