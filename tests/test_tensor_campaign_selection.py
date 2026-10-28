"""Campaign continuations must preserve the GPU preflight and explicit scope."""
import json
from pathlib import Path
import signal
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_tensor_edm_campaign as campaign
import run_remote_tensor_fit_campaign as remote


@pytest.mark.parametrize('runtime,requested,expected',[
    ('local',None,('cpu',)),('lcg',None,('gpu','cpu')),
    ('lcg',['cpu'],('cpu',)),('lcg',['cpu','gpu'],('cpu','gpu'))])
def test_backend_defaults_and_order(runtime,requested,expected):
    assert campaign.selected_backends(runtime,requested)==expected


@pytest.mark.parametrize('runtime,requested',[
    ('local',['gpu']),('local',['cpu','gpu']),('lcg',['cpu','cpu']),('lcg',[])])
def test_invalid_backend_selection(runtime,requested):
    with pytest.raises(ValueError):campaign.selected_backends(runtime,requested)


def test_cpu_only_lcg_still_checks_gpu_before_native_and_tensor(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path)
    import tensor_benchmark_dataset
    monkeypatch.setattr(tensor_benchmark_dataset,'benchmark_dataset',lambda *a,**k:'data_obs')
    monkeypatch.setattr(campaign.os,'sched_setaffinity',lambda *a:None)
    # Preserve pytest's signal handlers while exercising the real campaign flow.
    monkeypatch.setattr(signal,'signal',lambda *a:None)
    (tmp_path/'build/tensor_benchmark_inputs').mkdir(parents=True)
    (tmp_path/'reports').mkdir()
    (tmp_path/'build/edm_runtime_manifest.json').write_text(json.dumps(dict(sources={},source_revision='test')))
    manifest={}
    for case in ('counting','automcstats'):
        folder=tmp_path/case;folder.mkdir()
        (folder/'model.json').write_text(json.dumps(dict(metadata=dict(parameter_info=dict(fit_policy=dict(policy='combine_v11_default'))))))
        manifest[case]=dict(model=str(folder))
    (tmp_path/'build/tensor_benchmark_inputs/manifest.json').write_text(json.dumps(manifest))
    commands=[]
    class Child:
        pid=123
        returncode=0
        def __init__(self,command,**kwargs):
            commands.append(command)
            if 'scripts/verify_tensor_gpu.py' in command:
                Path(command[command.index('--output')+1]).write_text(json.dumps(dict(devices=[dict(kind='Tesla T4')])))
        def wait(self):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(campaign.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['campaign','--backends','cpu','--gpu-class','t4','--smoke-only'])
    campaign.main()
    assert 'scripts/verify_tensor_gpu.py' in commands[0]
    tensor=[c for c in commands if 'scripts/benchmark_tensor.py' in c]
    assert len(tensor)==2 and all(c[c.index('--backend')+1]=='cpu' for c in tensor)
    assert sum('scripts/benchmark_combine_fit.py' in c for c in commands)==2
    report=json.loads(Path('reports/gpu_tensor_edm_driver.json').read_text())
    assert report['status']=='smoke_checks_passed' and report['backends']==['cpu']


def test_remote_cpu_selection_is_forwarded(monkeypatch,tmp_path):
    monkeypatch.chdir(tmp_path);Path('reports').mkdir()
    monkeypatch.setattr(signal,'signal',lambda *a:None)
    monkeypatch.setattr(remote,'identity',lambda pid:dict(pid=pid,start_ticks=1))
    commands=[]
    class Child:
        pid=123
        returncode=0
        def __init__(self,command,**kwargs):commands.append(command)
        def wait(self):return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(remote.subprocess,'Popen',Child)
    monkeypatch.setattr(sys,'argv',['remote','--campaign','edm','--backends','cpu',
        '--prefix','gpu_tensor_test_','--output','reports/test.json'])
    remote.main()
    assert len(commands)==1 and '--gpu-class t4 --backends cpu' in commands[0][-1]
    assert json.loads(Path('reports/test.json').read_text())['requested_backends']==['cpu']


def test_legacy_remote_campaign_rejects_backend_selection(monkeypatch):
    monkeypatch.setattr(sys,'argv',['remote','--backends','cpu','--prefix','gpu_tensor_test_',
                                  '--output','reports/test.json'])
    with pytest.raises(SystemExit,match='2'):remote.main()
