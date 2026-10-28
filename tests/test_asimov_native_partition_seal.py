import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from seal_asimov_native_partition import worker_identity, validate_prefix
from audit_asimov_native_reference import native_point_records
from tensor_benchmark_common import sha
from test_asimov_native_shards import evidence


def test_worker_identity_rejects_reused_pid_wrong_script_and_wrong_directory(tmp_path):
    proc=tmp_path/'proc';work=tmp_path/'work';work.mkdir();process=proc/'42';process.mkdir(parents=True)
    fields=['S']+['0']*18+['123']
    (process/'stat').write_text('42 (python) '+' '.join(fields))
    command=b'python\0scripts/capture_asimov_native_fd_shard.py\0'
    (process/'cmdline').write_bytes(command);(process/'cwd').symlink_to(work)
    controller=dict(pid=42,start_ticks=123,workdir=str(work))
    assert worker_identity(controller,proc)
    with pytest.raises(AssertionError,match='reused'):worker_identity({**controller,'start_ticks':124},proc)
    with pytest.raises(AssertionError,match='directory'):worker_identity({**controller,'workdir':str(tmp_path)},proc)
    (process/'cmdline').write_bytes(b'python\0other.py\0')
    with pytest.raises(AssertionError,match='script'):worker_identity(controller,proc)
    assert not worker_identity({**controller,'pid':43},proc)


@pytest.fixture
def sealed(evidence):
    path,native,shard,*_=evidence
    source=copy.deepcopy(shard);source.update(status='running',workdir='/runtime',points=shard['points'][:1])
    source.pop('root_only')
    frozen=path.with_name('frozen.json');frozen.write_text(json.dumps(source))
    wrapper=dict(status='sealed_native_partition',dataset='asimov_prefit',owned_indices=[14],
        script_sha256=sha('scripts/seal_asimov_native_partition.py'),worker_verified_stopped=True,
        signal='SIGTERM',started_unix=1.,stopped_unix=2.,worker=dict(workdir='/runtime'),
        source_report=str(frozen),source_report_sha256=sha(frozen))
    return path,native,source,wrapper


def test_sealed_prefix_preserves_original_plan_and_allows_disjoint_tail(sealed,evidence):
    path,native,source,wrapper=sealed;path.write_text(json.dumps(wrapper))
    assert validate_prefix(source,[14])
    records,complete=native_point_records(native,[path])
    assert complete and [r['index'] for r in records]==[0,14]
    assert source['selected_indices']==[14,15] and source['status']=='running'
    waiting={**source,'points':[]};assert not validate_prefix(waiting,[14])
    tail=copy.deepcopy(evidence[2]);tail.update(selected_indices=[15],points=tail['points'][1:])
    tail_path=path.with_name('tail.json');tail_path.write_text(json.dumps(tail))
    records,complete=native_point_records(native,[path,tail_path])
    assert complete and [r['index'] for r in records]==[0,14,15]
    with pytest.raises(AssertionError,match='outside'):
        validate_prefix({**source,'points':evidence[2]['points']},[14])


@pytest.mark.parametrize('field,value',[('worker_verified_stopped',False),('dataset','data_obs'),
    ('source_report_sha256','wrong'),('owned_indices',[15]),('owned_indices',[14,15]),
    ('script_sha256','wrong'),('signal','SIGKILL')])
def test_unverified_or_incomplete_checkpoint_cannot_count_as_finished(sealed,field,value):
    path,native,_,wrapper=sealed;wrapper[field]=value;path.write_text(json.dumps(wrapper))
    with pytest.raises(AssertionError):native_point_records(native,[path])
