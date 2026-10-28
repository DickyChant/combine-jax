from pathlib import Path
import json
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from wait_asimov_t4 import native_controller_active, validation_ready


@pytest.mark.parametrize('script',['capture_asimov_native_reference.py','probe_asimov_cached_deltas.py',
                                  'capture_asimov_native_fd_shard.py'])
def test_both_native_reference_processes_block_t4(tmp_path,script):
    process=tmp_path/'123';process.mkdir()
    (process/'cmdline').write_bytes(b'python\0scripts/'+script.encode()+b'\0')
    (process/'stat').write_text('123 (python) S '+'0 '*18+'456 0')
    assert native_controller_active(dict(pid=123,start_ticks=456),tmp_path)
    assert native_controller_active(dict(pid=123),tmp_path)
    assert not native_controller_active(dict(pid=123,start_ticks=457),tmp_path)
    (process/'cmdline').write_bytes(b'python\0unrelated.py\0')
    assert not native_controller_active(dict(pid=123,start_ticks=456),tmp_path)
    assert not native_controller_active(dict(pid=124,start_ticks=456),tmp_path)


def gate_report():
    return dict(status='passed',dataset='asimov_prefit',model_sha256={'model.json':'model','arrays.npz':'arrays'},
        nll_passed=True,max_nll_absolute_error=1e-9,status_counts={'passed':52,'fixed_bounds':26},
        points=[dict(index=i,status_counts={'passed':2,'fixed_bounds':1},unresolved_or_failed=[]) for i in range(26)])


def test_full_native_gate_waits_for_missing_or_incomplete_report(tmp_path):
    path=tmp_path/'gate.json';report=gate_report();hashes=report['model_sha256']
    assert not validation_ready(path,hashes,3)['ready']
    report['status']='incomplete_or_failed_native_gradient_gate'
    path.write_text(json.dumps(report))
    assert not validation_ready(path,hashes,3)['ready']
    report['status']='passed';path.write_text(json.dumps(report))
    assert validation_ready(path,hashes,3)['ready']
    assert len(validation_ready(path,hashes,3)['sha256'])==64
    assert validation_ready(None,hashes,3)==dict(required=False,ready=True)


@pytest.mark.parametrize('fault',['dataset','model','duplicate_point','missing_point','nll',
                                  'nll_error','unresolved','status','total','parameter_count'])
def test_completed_gate_must_match_model_and_certify_all_points(tmp_path,fault):
    report=gate_report();hashes=report['model_sha256'].copy();count=3
    if fault=='dataset':report['dataset']='data_obs'
    elif fault=='model':report['model_sha256']['arrays.npz']='different'
    elif fault=='duplicate_point':report['points'][25]['index']=24
    elif fault=='missing_point':report['points'].pop()
    elif fault=='nll':report['nll_passed']=False
    elif fault=='nll_error':report['max_nll_absolute_error']=1e-6
    elif fault=='unresolved':report['points'][0]['unresolved_or_failed']=[{'index':1}]
    elif fault=='status':report['points'][0]['status_counts']={'passed':1,'failed_gradient':1,'fixed_bounds':1}
    elif fault=='total':report['status_counts']['passed']+=1
    elif fault=='parameter_count':count=4
    path=tmp_path/'gate.json';path.write_text(json.dumps(report))
    with pytest.raises(AssertionError):validation_ready(path,hashes,count)
