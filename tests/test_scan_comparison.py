import pytest

from combine_jax.scan_comparison import compare_scan_points,linear_intervals,check_global_baseline


def test_disconnected_intervals_and_unbracketed_boundaries():
    points=[[-3,0],[-2,2],[-1,0],[0,2],[1,0],[2,0]]
    intervals=linear_intervals(points,1)['intervals']
    assert intervals==[
        dict(lower=None,upper=-2.5,lower_bracketed=False,upper_bracketed=True),
        dict(lower=-1.5,upper=-.5,lower_bracketed=True,upper_bracketed=True),
        dict(lower=.5,upper=None,lower_bracketed=True,upper_bracketed=False)]


def test_touch_and_threshold_plateau_are_preserved():
    assert linear_intervals([[-1,2],[0,1],[1,2]],1)['intervals']==[
        dict(lower=0.,upper=0.,lower_bracketed=True,upper_bracketed=True)]
    result=linear_intervals([[-1,2],[0,1],[1,1],[2,2]],1)['intervals'][0]
    assert result['lower']==0 and result['upper']==1


def test_point_comparison_never_shifts_or_interpolates():
    result=compare_scan_points([[1,3],[0,2]],[[0,0],[1,1]])
    assert result['max_absolute_difference']==2
    assert not result['statistic_shift_applied']
    with pytest.raises(ValueError,match='same coordinates'):
        compare_scan_points([[0,0],[2,1]],[[0,0],[1,1]])
    with pytest.raises(ValueError,match='unique'):
        linear_intervals([[1,0],[1,1]],1)


def test_edm_status_zero_cannot_hide_lower_conditional_point():
    result=check_global_baseline([dict(value=125831523.35924175,status=0),
        dict(task='kl2',fixed={'kl':2.},value=125831512.2580724,status=0)])
    assert not result['valid'] and result['lower_points'][0]['deltaNLL'] < -11
    assert not result['statistic_shift_applied']
    assert check_global_baseline([dict(value=1e8),dict(value=1e8-1e-8)])['valid']
    assert not check_global_baseline([dict(value=1.),dict(value=float('nan'))])['valid']
