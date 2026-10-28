"""Sparse observed rows retain complete normalization and native derivatives."""
from itertools import product

import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.root_io import load_caching_nll
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.binned_channels import pack_binned_channels,make_binned_function,unpack_binned,validate_binned_channels
from combine_jax.tensor.shape_data import prepare_shape_data
from test_tensor_binned_channels import bind_model,check_model
from test_vertical_nd import make_multidimensional
from test_tensor_analytic_shapes import shape_fixture


def test_exact_source_row_selection_and_padding():
    row=dict(name='data',observables=('x',),coordinates={'x':np.arange(7.)},weights=np.array([0.,2.5,0.,4.,0.,1.,0.]))
    rows,layouts,arrays=prepare_shape_data([row]);layout=layouts[0]
    np.testing.assert_array_equal(rows[0]['weights'],[2.5,4.,1.,0.])
    np.testing.assert_array_equal(rows[0]['coordinates']['x'],[1.,3.,5.,1.])
    np.testing.assert_array_equal(arrays[layout['source_indices']],[1,3,5,1])
    np.testing.assert_array_equal(arrays[layout['source_mask']],[True,True,True,False])
    assert layout['observed_sum']==7.5 and layout['original_rows']==7 and layout['packed_rows']==4
    kept,_,_=prepare_shape_data([row],compact=False)
    np.testing.assert_array_equal(kept[0]['coordinates']['x'],row['coordinates']['x'])
    bad={**row,'coordinates':{'x':np.array([np.nan,1.,2.,3.,4.,5.,6.])}}
    with pytest.raises(UnsupportedModelError,match='data.*invalid data coordinates'):prepare_shape_data([bad])
    bad={**row,'weights':np.array([0.,2.5,-1.,4.,0.,1.,0.])}
    with pytest.raises(UnsupportedModelError,match='data.*invalid weights'):prepare_shape_data([bad])


@pytest.mark.parametrize('logarithmic',[False,True])
def test_sparse_histogram_channels_keep_all_normalization_bins(root,logarithmic):
    root=load_caching_nll();axes,a,b,shape,edges,held=make_multidimensional(root,'FastVerticalInterpHistPdf2D2',True,logarithmic)
    points=np.array(list(product(*[(e[:-1]+e[1:])/2 for e in edges])))
    # The fixture's first x slice is identically zero, including in the native
    # logarithmic morph. Keep observations in bins with positive support.
    weights=np.zeros(len(points));weights[[2,3,-1]]=[2.5,4.,1.]
    model=bind_model(root,shape,axes,dict(zip(('ndx','ndy'),points.T,strict=True)),weights=weights);check_model(model)
    compact=model['binned'];full=pack_binned_channels(model['nodes'],model['rows'],model['scalar'],compact_data=False)
    cparts,fparts=unpack_binned(compact),unpack_binned(full)
    assert cparts['histograms'].metadata==fparts['histograms'].metadata
    for name,array in cparts['histograms'].arrays.items():np.testing.assert_array_equal(array,fparts['histograms'].arrays[name])
    old=make_binned_function(full);sa=model['arrays']['s'];ba=jax.device_put(dict(full.arrays));sf=model['sf']
    def reference(theta):return old(sf(theta,sa),ba)[0]
    point=model['old'].initial
    np.testing.assert_allclose(jax.jit(reference)(point),model['native'](point),atol=1e-10)
    np.testing.assert_allclose(jax.hessian(reference)(point),jax.jit(jax.hessian(model['function']))(point,model['arrays']),atol=1e-10)
    layout=compact.metadata['data_layouts'][0];assert layout['retained_rows']==3 and layout['packed_rows']==4
    bad=dict(compact.arrays);key=layout['source_indices'];bad[key]=np.full_like(bad[key],layout['original_rows'])
    with pytest.raises(ValueError,match='Invalid shape data source indices'):validate_binned_channels(TensorBundle(compact.metadata,bad))


@pytest.mark.parametrize('family',['gaussian','doublecb','erfexp','power'])
def test_zero_observation_channels_keep_extended_rate_and_derivatives(root,family):
    root=load_caching_nll();f=shape_fixture(root,family)
    count=len(next(iter(f['coordinates'].values())))
    model=bind_model(root,f['shape'],[f['x']],f['coordinates'],weights=np.zeros(count));check_model(model)
    assert model['binned'].metadata['data_layouts'][0]['packed_rows']==0
    value,gradient=jax.jit(jax.value_and_grad(model['function']))(model['old'].initial,model['arrays'])
    rate=model['old'].parameter_names.index('test_rate');expected=np.zeros(len(gradient));expected[rate]=1.
    np.testing.assert_allclose(value,12.,atol=1e-13);np.testing.assert_allclose(gradient,expected,atol=1e-13)
