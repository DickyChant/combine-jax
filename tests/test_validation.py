import numpy as np
import pytest

from combine_jax.validation import finite_difference


@pytest.mark.parametrize("point", [np.array([0.,1.]),np.array([1.,0.]),np.array([.000001,.999999]),np.array([.4,.7])])
def test_bounded_finite_difference_never_probes_outside_range(point):
    def function(x):
        assert np.all((x>=0)&(x<=1))
        return np.array([x[0]**4+2*x[1]**3,x[0]*x[1]])
    actual=finite_difference(function,point,np.array([.001,.002]),bounds=(np.zeros(2),np.ones(2)))
    expected=np.array([[4*point[0]**3,point[1]],[6*point[1]**2,point[0]]])
    np.testing.assert_allclose(actual,expected,atol=1e-11,rtol=1e-10)


def test_bounded_finite_difference_reduces_step_for_narrow_ranges():
    def f(x):
        assert 0<=x[0]<=.001
        return x[0]**2
    actual=finite_difference(f,np.array([.001]),np.array([.2]),bounds=([0.],[.001]))
    np.testing.assert_allclose(actual,[.002],atol=1e-14)
