"""Never silently invent active coordinates while replaying fixed-choice gates."""
import numpy as np
import pytest

from combine_jax.graph import Parameter
from combine_jax.tensor.parameter_info import pack_parameter_info
from combine_jax.tensor.point_replay import extend_inactive_parameters


def inputs():
    info=pack_parameter_info([Parameter('r',1.,-5.,5.,.3,'poi'),
        Parameter('inactive',.4,-2.,2.,.1,'nuisance'),Parameter('shape',.2,-1.,1.,.05,'nuisance')])
    source=dict(parameter_names=np.array(['shape','r']),points=np.array([[.2,1.],[.1,.8],[.4,1.3]]),
        scales=np.array([.05,.3]),fd_steps=np.array([1e-5,6e-5]),lower=np.array([-1.,-5.]),
        upper=np.array([1.,5.]),free_indices=np.array([0,1],dtype=np.int32))
    return info,source,np.array([True,False,True])


def test_exact_binding_and_inactive_extension():
    info,source,active=inputs();result,proof=extend_inactive_parameters(info,source,active)
    np.testing.assert_array_equal(result['points'][:,[2,0]],source['points'])
    np.testing.assert_array_equal(result['points'][:,1],np.full(3,.4))
    np.testing.assert_array_equal(result['fd_steps'][[2,0]],source['fd_steps'])
    assert proof['original_to_profile_indices']==[2,0]
    assert proof['additional_inactive_parameters']==[{'name':'inactive','initial':.4}]
    assert not any(proof[k] for k in ('resampled','clipped','source_native_values_reused','other_discrete_choices_validated'))


@pytest.mark.parametrize('fault', ['active','missing','nominal','bounds','nonfinite','outside','free','steps'])
def test_replay_rejects_mismatched_or_invented_coordinates(fault):
    info,source,active=inputs()
    if fault=='active':active[1]=True
    elif fault=='missing':source['parameter_names']=np.array(['unknown','r'])
    elif fault=='nominal':source['points'][0,0]+=.1
    elif fault=='bounds':source['lower'][0]-=.1
    elif fault=='nonfinite':source['points'][1,0]=np.nan
    elif fault=='outside':source['points'][1,0]=10.
    elif fault=='free':source['free_indices']=np.array([1],dtype=np.int32)
    elif fault=='steps':source['fd_steps'][0]=0.
    with pytest.raises(ValueError):extend_inactive_parameters(info,source,active)
