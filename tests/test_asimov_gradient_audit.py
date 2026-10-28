from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_asimov_native_reference import compare_coordinate


def test_all_selected_native_estimates_must_pass():
    case=dict(status='converged',selected_first_probe_index=1,probes=[dict(status='invalid')]+
        [dict(status='valid',derivative=x) for x in (1.,1.00002,1.00004)])
    assert compare_coordinate(1.00001,case)['status']=='passed'
    assert compare_coordinate(1.00012,case)['status']=='failed_gradient'


def test_uncertain_and_zero_native_references_cannot_pass_by_agreement():
    for status in ('unresolved','roundoff_sensitive'):
        assert compare_coordinate(0.,dict(status=status))['status']=='uncertified_native_'+status
    assert compare_coordinate(0.,dict(status='fixed_bounds'))['status']=='fixed_bounds'
    assert compare_coordinate(1e-15,dict(status='fixed_bounds'))['status']=='failed_fixed_derivative'
