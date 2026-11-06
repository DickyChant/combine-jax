"""Continuous candidate fits: batching, offsets and cap semantics, on small arrays."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import Parameter
from combine_jax.tensor.discrete_batch import orthogonal_offsets
from combine_jax.tensor.discrete_batched_fit import make_batched_discrete_fit_function
from combine_jax.tensor.parameter_info import pack_parameter_info,fit_coordinates,COORDINATE_FIELDS


def fixture(anchor):
    table=np.zeros((3,2,5),dtype=bool)
    table[0,0,0]=table[1,0,1]=True
    table[0,1,2]=table[1,1,2]=True
    table[2,0,3]=table[2,1,4]=True
    sizes=np.array([2,2,2],dtype=np.int32);order=np.array([1,0,2],dtype=np.int32)
    arrays=dict(binned__discrete_indices=np.array(anchor,dtype=np.int32),
        discrete_profile__alternative_parameters=table,discrete_profile__parameter_union=table.any(axis=(0,1)),
        discrete_profile__channel_parameters=table.any(axis=1),discrete_profile__channel_categories=np.eye(3,dtype=bool),
        discrete_profile__category_sizes=sizes,discrete_profile__native_order=order,
        discrete_profile__orthogonal_offsets=orthogonal_offsets(sizes,order))
    info=pack_parameter_info([Parameter(str(i),v,-5.,5.,1.,'nuisance')
                              for i,v in enumerate([.9,-.7,1.3,-.4,1.1])])
    coordinates=fit_coordinates(info)
    return SimpleNamespace(metadata={'discrete_profile':{}},arrays=arrays),{k:coordinates[k] for k in COORDINATE_FIELDS}


def channels(theta,arrays):
    a,b,shared,c,d=theta;i=arrays['binned__discrete_indices']
    return jnp.array([jnp.where(i[0]==0,(a-.2)**2,(shared-1.)**2-.7),
                      jnp.where(i[1]==0,(b+.3)**2,(shared+2.)**2-.4),
                      jnp.where(i[2]==0,c**2,(d-.5)**2-.2)])


@pytest.mark.parametrize('anchor',[(0,0,0),(0,1,0),(1,1,1)])
def test_batched_fit_preserves_sequential_results_and_reports_physical_nll(anchor):
    model,c=fixture(anchor);arrays=jax.device_put(model.arrays);outputs=[]
    full=lambda theta,a:jnp.sum(channels(theta,a))
    for width in (1,3):
        function=make_batched_discrete_fit_function(model,_full_function=full,_channel_function=channels,
            candidate_batch_size=width,maximum_discrete_iterations=4,return_matrices=False,
            stopping_criterion='edm',edm_validation='estimated',edm_tolerance=1e-8,
            rescale_small_secants=True)
        outputs.append(jax.device_get(jax.jit(function)(c,c,arrays)))
    a,b=outputs
    assert int(a['candidate_failures'])==0
    for name in ('discrete_indices','discrete_iteration','status','candidate_status','discrete_index_history'):
        np.testing.assert_array_equal(a[name],b[name])
    for name in ('raw_value','masking_offset','native_rank_value','values','discrete_raw_values','discrete_native_values'):
        np.testing.assert_allclose(a[name],b[name],atol=1e-11,rtol=0,equal_nan=True)
    physical=float(full(a['values'],{**model.arrays,'binned__discrete_indices':a['discrete_indices']}))
    assert abs(physical-float(a['raw_value']))<1e-12
    assert abs(float(a['native_rank_value'])-physical-float(a['masking_offset']))<1e-12
    if a['discrete_iteration']==4 and a['discrete_delta']>=.001:
        assert a['status']==6
    assert b['candidate_waves']<=a['candidate_waves']


def test_shared_fitter_requires_native_category_order():
    model,_=fixture([0,0,0]);del model.arrays['discrete_profile__native_order']
    with pytest.raises(ValueError,match='category order'):
        make_batched_discrete_fit_function(model,_full_function=lambda *a:0.,_channel_function=channels)
