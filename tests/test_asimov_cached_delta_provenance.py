import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_native_reference import verified_cached_deltas, compare_coordinate
from native_fd_refinement import REFINEMENT_FRACTIONS, select_reference
from tensor_benchmark_common import sha


@pytest.fixture
def reference(tmp_path):
    native = dict(workspace_sha256='workspace', generation_report_sha256='generation', nll_arrays_sha256='values')
    probes = [dict(fraction=h, step=h, derivative=1e-10, stencil='central', direction=1., status='valid')
              for h in REFINEMENT_FRACTIONS]
    case = dict(coordinate=0., probes=probes, **select_reference(probes))
    assert case['status'] == 'roundoff_sensitive'
    scripts = ['scripts/probe_asimov_cached_deltas.py', 'scripts/native_cached_deltas.py',
               'scripts/native_fd_refinement.py', 'scripts/native_fd_scan.py']
    report = dict(status='diagnostic_complete_pending_audit', root_only=True, dataset='asimov_prefit',
        workspace_sha256='workspace', generation_report_sha256='generation', native_values_sha256='values',
        root_version='6.40.00', combine_version='11.0.0', longdouble_mantissa_bits=63,
        max_delta_parity_error=1e-10, artifacts={p: sha(p) for p in scripts}, fractions=list(REFINEMENT_FRACTIONS),
        workdir='/runtime', cases=[dict(point=0, index=0, name='p', reference=case,
            canonical_reference_error=0., canonical_nll=10., max_delta_parity_error=1e-10)])
    return tmp_path/'reference.json', report, native


def verify(fixture, report):
    path, _, native = fixture
    path.write_text(json.dumps(report))
    return verified_cached_deltas([path], native, np.array(['p']), np.array([[0.]]), np.array([10.]))


def test_small_stable_reference_keeps_unchanged_ad_tolerance(reference):
    item = verify(reference, reference[1])[(0, 0)]
    assert item['original_reference_status'] == 'roundoff_sensitive'
    assert compare_coordinate(1e-10, item['case'])['status'] == 'passed'
    assert compare_coordinate(1.02e-10, item['case'])['status'] == 'failed_gradient'
    assert reference[1]['cases'][0]['reference']['status'] == 'roundoff_sensitive'


@pytest.mark.parametrize('field,value', [('dataset', 'data_obs'), ('workspace_sha256', 'wrong'),
    ('longdouble_mantissa_bits', 52), ('max_delta_parity_error', 1e-6)])
def test_evidence_or_precision_mismatch_cannot_certify(reference, field, value):
    report = copy.deepcopy(reference[1]); report[field] = value
    with pytest.raises(AssertionError):
        verify(reference, report)


def test_ad_dependent_selection_cannot_certify(reference):
    report = copy.deepcopy(reference[1])
    report['cases'][0]['reference']['selected_first_probe_index'] = 1
    with pytest.raises(AssertionError, match='without AD'):
        verify(reference, report)


def test_longer_native_plateau_is_preferred_before_reading_ad(reference):
    report=copy.deepcopy(reference[1]);case=report['cases'][0]['reference']
    derivatives=[1.,1.,1.,1.5,2.5,3.,*([2.]*13)]
    for probe,value in zip(case['probes'],derivatives,strict=True):probe['derivative']=value
    case.update(select_reference(case['probes']))
    item=verify(reference,report)[(0,0)]
    assert item['unconfirmed_case']['selected_first_probe_index']==0
    assert item['case']['selected_first_probe_index']==6
    assert item['case']['reference_probe_count']==6
    assert compare_coordinate(1.,item['case'])['status']=='failed_gradient'
    assert compare_coordinate(2.,item['case'])['status']=='passed'
    assert case['selected_first_probe_index']==0


def test_absent_longer_plateau_preserves_original_native_policy(reference):
    report=copy.deepcopy(reference[1]);case=report['cases'][0]['reference']
    derivatives=[1.,1.,1.,*map(float,range(2,18))]
    for probe,value in zip(case['probes'],derivatives,strict=True):probe['derivative']=value
    case.update(select_reference(case['probes']))
    item=verify(reference,report)[(0,0)]
    assert item['confirmation']['status']=='unresolved'
    assert item['case']['selected_first_probe_index']==0
    assert item['case'].get('reference_probe_count',3)==3
    assert compare_coordinate(1.,item['case'])['status']=='passed'
    assert compare_coordinate(1.001,item['case'])['status']=='failed_gradient'


def test_runtime_absolute_script_path_is_bound_to_workdir(reference):
    report = copy.deepcopy(reference[1])
    script = 'scripts/probe_asimov_cached_deltas.py'
    report['artifacts']['/runtime/'+script] = report['artifacts'].pop(script)
    verify(reference, report)
    report['workdir'] = '/other'
    with pytest.raises(ValueError):
        verify(reference, report)
