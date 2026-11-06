"""Reference FD arithmetic must not invent derivatives of a fixed integral."""
import numpy as np
import pytest

from combine_jax import reference_numerics as module

@pytest.mark.parametrize('point',[0.,.5,1.])
def test_five_point_reference_annihilates_constant_offsets(point):
    value=module.derivative_column(lambda p:np.array([39.8123456789,1e9]),np.array([point]),0,1e-7,
                                   np.array([0.]),np.array([1.]),bound_aware=True)
    np.testing.assert_array_equal(value,[0.,0.])


@pytest.mark.parametrize('point',[0.,.5,1.])
def test_bounded_reference_keeps_fourth_order_polynomial_derivative(point):
    result=module.derivative_column(lambda p:np.array([3*p[0]**4-2*p[0]**2+p[0]+7]),np.array([point]),0,.01,
                                    np.array([0.]),np.array([1.]),bound_aware=True)
    np.testing.assert_allclose(result,[12*point**3-4*point+1],atol=1e-11,rtol=0.)
