import copy
import gzip
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_native_reference import native_point_records,checked_native_point
from native_fd_scan import FRACTIONS
from tensor_benchmark_common import sha


@pytest.fixture
def evidence(tmp_path):
    names=np.array(['p']);points=np.zeros((26,1));values=np.arange(26.)
    def record(i):
        raw=dict(index=i,canonical_nll=float(values[i]),zero_point_nll=0.,restoration_error=0.,
            cases=[dict(index=0,name='p',coordinate=0.,status='fixed_bounds')])
        path=tmp_path/f'point_{i}.json.gz';path.write_bytes(gzip.compress(json.dumps(raw).encode()))
        return dict(index=i,artifact=str(path),artifact_sha256=sha(path),restoration_error=0.)
    native=dict(points=[record(0)],workspace_sha256='workspace',generation_report_sha256='generation',
        nll_arrays_sha256='values',runtime_flags={'NATIVE':1})
    shard=dict(status='shard_complete_pending_audit',root_only=True,dataset='asimov_prefit',
        workspace_sha256='workspace',generation_report_sha256='generation',native_values_sha256='values',
        root_version='6.40.00',combine_version='11.0.0',runtime_flags=native['runtime_flags'],
        script_sha256=sha('scripts/capture_asimov_native_fd_shard.py'),
        fd_script_sha256=sha('scripts/native_fd_scan.py'),value_helper_sha256=sha('scripts/capture_asimov_native_reference.py'),
        fractions=list(FRACTIONS),selected_indices=[14,15],points=[record(14),record(15)])
    return tmp_path/'shard.json',native,shard,names,points,values


def merge(evidence,shard):
    path,native,*_=evidence;path.write_text(json.dumps(shard))
    return native_point_records(native,[path])


def test_disjoint_complete_subset_and_point_identity(evidence):
    _,_,shard,names,points,values=evidence
    records,complete=merge(evidence,shard)
    assert complete and [r['index'] for r in records]==[0,14,15]
    for row in records:checked_native_point(row,names,points,values)


def test_running_subset_cannot_claim_complete(evidence):
    shard=copy.deepcopy(evidence[2]);shard.update(status='running',points=shard['points'][:1]);shard.pop('root_only')
    records,complete=merge(evidence,shard)
    assert not complete and [r['index'] for r in records]==[0,14]


@pytest.mark.parametrize('field,value',[('dataset','data_obs'),('workspace_sha256','other'),
    ('native_values_sha256','other'),('fd_script_sha256','other'),('selected_indices',[0,14,15]),
    ('selected_indices',[14,14]),('runtime_flags',{})])
def test_mixed_provenance_or_overlapping_points_are_rejected(evidence,field,value):
    shard=copy.deepcopy(evidence[2]);shard[field]=value
    with pytest.raises(AssertionError):merge(evidence,shard)


def test_parameter_identity_and_canonical_values_are_bound(evidence):
    _,native,_,names,points,values=evidence;record=native['points'][0]
    with pytest.raises(AssertionError):checked_native_point(record,np.array(['other']),points,values)
    changed=points.copy();changed[0,0]=1.
    with pytest.raises(AssertionError):checked_native_point(record,names,changed,values)
    with pytest.raises(AssertionError):checked_native_point(record,names,points,values+1e-5)
