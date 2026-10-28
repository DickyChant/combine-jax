"""Fresh single-family submissions preserve payloads and explicit worker options."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import submit_asimov_campaigns as submit
from tensor_benchmark_common import sha


@pytest.mark.parametrize('families',[['A100'],['A100','H100']])
def test_submit_only_requested_families_with_exact_payload(monkeypatch,tmp_path,families):
    work=tmp_path/'assembly/work';(work/'scripts').mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts'
    for name in ('condor_tensor_edm.sh','condor_tensor_edm.sub'):
        (work/'scripts'/name).write_bytes((source/name).read_bytes())
    manifest=work/'build/tensor_benchmark_inputs/manifest.json'
    manifest.parent.mkdir(parents=True);manifest.write_text(json.dumps(dict(hhvv=dict(dataset='asimov_prefit'))))
    payload=work.parent/'payload.tar.gz';payload.write_bytes(b'immutable payload')
    patch=work.parent/'runtime_patch.tar';patch.write_bytes(b'committed sources')
    assembly=tmp_path/'assembly.json'
    assembly.write_text(json.dumps(dict(status='assembled_asimov_pending_numerical_validation',
        source_revision='revision',payload=str(payload),patch=str(patch),payload_sha256=sha(payload),
        patch_sha256=sha(patch),dataset_identity={'sha256':'Asimov identity'})))
    output=tmp_path/'submission.json';directory=tmp_path/'jobs'
    monkeypatch.setattr(Path,'cwd',classmethod(lambda cls:Path('/tmp/sqian/combine-jax')))
    monkeypatch.setattr(submit.socket,'gethostname',lambda:'lxplus905.cern.ch')
    commands=[]
    def run(command,**kwargs):
        commands.append((command,kwargs['cwd']))
        return SimpleNamespace(returncode=0,stdout=f'1 job(s) submitted to cluster {100+len(commands)}.\n',stderr='')
    monkeypatch.setattr(submit.subprocess,'run',run)
    monkeypatch.setattr(sys,'argv',['submit','--assembly',str(assembly),'--directory',str(directory),
        '--output',str(output),'--families',*families,'--hhvv-fit-order','tensor-first','--gpu-hessian-batch-size','1'])
    submit.main()
    report=json.loads(output.read_text())
    assert report['status']=='submitted_asimov_pending_measurements'
    assert [r['family'] for r in report['jobs']]==families
    assert len(commands)==len(families)
    assert sorted(p.name for p in directory.iterdir())==sorted(f.lower() for f in families)
    for family in families:
        folder=directory/family.lower()
        assert (folder/'payload.tar.gz').samefile(payload) and (folder/'runtime_patch.tar').samefile(patch)
        worker=(folder/'worker.sh').read_text()
        assert worker.endswith('scripts/run_tensor_edm_campaign.py --hhvv-fit-order tensor-first --gpu-hessian-batch-size 1\n')
        assert f'regexp("{family}", GPUs_DeviceName)' in (folder/'job.sub').read_text()
        assert 'request_gpus = 1' in (folder/'job.sub').read_text()
    assert (work/'scripts/condor_tensor_edm.sh').read_bytes()==(source/'condor_tensor_edm.sh').read_bytes()


@pytest.mark.parametrize('options',[
    ['--families','A100','A100'],['--gpu-hessian-batch-size','0'],['--gpu-hessian-batch-size','-1']])
def test_invalid_options_do_not_submit(monkeypatch,options):
    monkeypatch.setattr(sys,'argv',['submit','--assembly','unused','--directory','unused','--output','unused',*options])
    with pytest.raises(SystemExit,match='2'):submit.main()
