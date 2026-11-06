"""Explicit native-order maps retain shared shapes without bypassing old guards."""
import jax
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.model import from_workspace, make_model_function
from combine_jax.tensor.discrete import make_discrete_channels_function
from combine_jax.tensor.discrete_fit import make_discrete_fit_function
from test_tensor_fixed_shared_discrete import fixture


def registered_fixture():
    root=load_caching_nll();w,held=fixture(root,(0,0))
    order=root.RooArgSet('discreteParams')
    order.add(w.cat('index_b'));order.add(w.cat('index_a'))
    getattr(w,'import')(order,'discreteParams')
    return root,w,held


def test_shared_import_requires_registered_native_order():
    root=load_caching_nll();w,held=fixture(root,(0,0))
    with pytest.raises(ValueError,match='order is missing'):
        from_workspace(w,data='toy_asimov',discrete_policy='profile',discrete_profile_mode='native_batched')


def test_native_shared_maps_and_likelihoods_for_all_choices():
    root,w,held=registered_fixture()
    model=from_workspace(w,data='toy_asimov',discrete_policy='profile',discrete_profile_mode='native_batched')
    names=model.metadata['parameter_names'];a=model.arrays
    meta=model.metadata['discrete_profile']
    assert meta['native_category_order']==['index_b','index_a']
    assert meta['driver']=='native_batched'
    assert a['discrete_profile__native_order'].tolist()==[1,0]
    assert a['discrete_profile__orthogonal_offsets'].tolist()==[[0,0],[0,1],[1,0]]
    union=a['discrete_profile__parameter_union']
    assert {n for n,m in zip(names,union) if m}=={'shared_slope','slope_a','slope_b'}
    membership=a['discrete_profile__channel_parameters']
    assert membership[:,names.index('shared_slope')].all()
    assert not membership[:,names.index('kl')].any()
    with pytest.raises(ValueError,match='batched masking replay'):
        make_discrete_fit_function(model)
    full=jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient='structural_sequential')))
    local=jax.jit(make_discrete_channels_function(model))
    theta=a['parameter_info__initial'].copy();theta[names.index('shared_slope')]=-.37
    for indices in ((0,0),(0,1),(1,0),(1,1)):
        for label,index in zip(('a','b'),indices):w.cat('index_'+label).setIndex(index)
        frozen=from_workspace(w,data='toy_asimov')
        fixed_names=frozen.metadata['parameter_names'];take=[names.index(n) for n in fixed_names]
        oracle=jax.jit(jax.value_and_grad(make_model_function(frozen,binned_gradient='structural_sequential')))
        expected,grad=oracle(theta[take],dict(frozen.arrays))
        buffers={**a,'binned__discrete_indices':np.array(indices,dtype=np.int32)}
        value,gradient=full(theta,buffers)
        expected_gradient=np.zeros(len(names));expected_gradient[take]=grad
        np.testing.assert_allclose(value,expected,rtol=0,atol=1e-10)
        np.testing.assert_allclose(gradient,expected_gradient,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(np.sum(local(theta,buffers)),value,rtol=0,atol=1e-10)
