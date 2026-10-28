import copy
import json

import numpy as np
import pytest

from test_asimov_cached_delta_provenance import reference
from audit_asimov_native_reference import verified_cached_deltas, add_centered_references, compare_coordinate
from tensor_benchmark_common import sha


@pytest.fixture
def centered(reference):
    path, original, native = reference
    report = copy.deepcopy(original)
    report.update(reference_method='native_centered_histogram_score', runtime_flags={'NATIVE': 1})
    native['runtime_flags'] = report['runtime_flags']
    del report['artifacts']['scripts/probe_asimov_cached_deltas.py']
    for script in ('scripts/probe_asimov_centered_hist_score.py', 'scripts/native_centered_hist_score.py'):
        report['artifacts'][script] = sha(script)
    report['cases'][0]['normalized_components'] = [dict(class_name='FastVerticalInterpHistPdf2D2',
        full_grid_rows=4, integral=1.)]
    return path, report, native


def verify(fixture, report):
    path, _, native = fixture
    path.write_text(json.dumps(report))
    return verified_cached_deltas([path], native, np.array(['p']), np.array([[0.]]), np.array([10.]), centered=True)


def test_centered_reference_preserves_previous_failure_and_ad_tolerance(centered):
    result = verify(centered, centered[1])
    item = result[(0, 0)]
    assert compare_coordinate(1e-10, item['case'])['status'] == 'passed'
    assert compare_coordinate(1.02e-10, item['case'])['status'] == 'failed_gradient'
    previous = dict(reference_method='native_cached_longdouble_delta', case=dict(status='unresolved'),
                    source='previous.json', source_sha256='previous_hash')
    references = {(0, 0): previous}
    add_centered_references(references, result)
    assert references[(0, 0)]['previous_reference'] == previous
    assert previous['case']['status'] == 'unresolved'
    previous['case']['status'] = 'converged'
    with pytest.raises(AssertionError):
        add_centered_references({(0, 0): previous}, result)


@pytest.mark.parametrize('mutation', ['method', 'flags', 'source', 'class', 'integral', 'selection'])
def test_unproved_centering_cannot_certify(centered, mutation):
    report = copy.deepcopy(centered[1])
    if mutation == 'method': report['reference_method'] = 'different'
    if mutation == 'flags': report['runtime_flags'] = {}
    if mutation == 'source': report['artifacts']['scripts/native_centered_hist_score.py'] = 'wrong'
    if mutation == 'class': report['cases'][0]['normalized_components'][0]['class_name'] = 'RooGaussian'
    if mutation == 'integral': report['cases'][0]['normalized_components'][0]['integral'] = .99
    if mutation == 'selection': report['cases'][0]['reference']['selected_first_probe_index'] = 1
    with pytest.raises(AssertionError): verify(centered, report)


def test_unsupported_case_leaves_original_reference_uncertified(centered):
    report = copy.deepcopy(centered[1])
    report['cases'][0].update(reference=dict(status='unsupported', coordinate=0., probes=[]),
                              reason='Only unconditional histograms are supported')
    assert verify(centered, report) == {}
