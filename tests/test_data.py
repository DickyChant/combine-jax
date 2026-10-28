import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.data import bind_data, bind_channels
from combine_jax.graph import UnsupportedModelError
from combine_jax.likelihood import extended_nll


def test_weighted_rows_keep_actual_coordinates_and_zero_weights(root):
    x = root.RooRealVar("data_x", "", 0., -1., 2.)
    weight = root.RooRealVar("weight", "", 1., -5., 5.)
    variables = root.RooArgSet(x, weight)
    data = root.RooDataSet("data", "", variables, root.RooFit.WeightVar(weight))
    for coordinate, w in ((.13, .2), (.43, 0.), (1.72, 2.75)):
        x.setVal(coordinate)
        data.add(variables, w)
    bound = bind_data(data, ["data_x"])
    np.testing.assert_array_equal(bound.coordinates["data_x"], [.13, .43, 1.72])
    np.testing.assert_array_equal(bound.weights, [.2, 0., 2.75])
    assert not bound.weights.flags.writeable
    data.add(variables, -.1)
    with pytest.raises(UnsupportedModelError, match="Negative"):
        bind_data(data, ["data_x"])


def test_binned_rows(root):
    x = root.RooRealVar("hist_x", "", 0., 0., 3.)
    x.setBins(3)
    hist = root.TH1D("hist_data", "", 3, 0., 3.)
    hist.SetDirectory(0)
    for i, content in enumerate([0., 1.5, 3.], 1):
        hist.SetBinContent(i, content)
    data = root.RooDataHist("data_hist", "", root.RooArgList(x), hist)
    bound = bind_data(data, ["hist_x"])
    np.testing.assert_array_equal(bound.coordinates["hist_x"], [.5, 1.5, 2.5])
    np.testing.assert_array_equal(bound.weights, [0., 1.5, 3.])


def test_bulk_channel_export_noncontiguous_categories_and_empty_channel(root):
    x=root.RooRealVar("bulk_x","",.2,-1.,3.)
    y=root.RooRealVar("bulk_y","",.4,0.,5.)
    category=root.RooCategory("bulk_category","")
    for label,index in (("zeta",7),("alpha",2),("empty",19)):category.defineType(label,index)
    weight=root.RooRealVar("bulk_weight","",1.,0.,10.)
    collection=root.RooArgSet(x,y,category,weight)
    data=root.RooDataSet("bulk_data","",collection,root.RooFit.WeightVar(weight))
    for label,xx,yy,w in [("zeta",.23,.75,.5),("alpha",1.47,2.3,0.),("zeta",2.1,3.9,2.75)]:
        category.setLabel(label);x.setVal(xx);y.setVal(yy);data.add(collection,w)
    schemas={"alpha":("bulk_y",),"empty":("bulk_x",),"zeta":("bulk_x","bulk_y")}
    result=bind_channels(data,"bulk_category",schemas)
    for label,names in schemas.items():
        reduced=data.reduce(root.RooFit.Cut(f"bulk_category==bulk_category::{label}"))
        expected=bind_data(reduced,names)
        np.testing.assert_array_equal(result[label].weights,expected.weights)
        for name in names:np.testing.assert_array_equal(result[label].coordinates[name],expected.coordinates[name])
    np.testing.assert_array_equal(result["zeta"].coordinates["bulk_x"],[.23,2.1])
    np.testing.assert_array_equal(result["zeta"].weights,[.5,2.75])
    assert not len(result["empty"].weights)
    data.reset()
    np.testing.assert_array_equal(result["zeta"].weights,[.5,2.75])


def test_zero_weight_zero_density_and_empty_sample_have_finite_derivatives():
    def objective(theta):
        return extended_nll(jnp.array([0., theta[0]]), theta[1], jnp.array([0., 1.2]))
    point = jnp.array([.7, 2.])
    assert np.isfinite(objective(point))
    assert np.all(np.isfinite(jax.grad(objective)(point)))
    assert np.all(np.isfinite(jax.hessian(objective)(point)))
    empty = lambda y: extended_nll(jnp.array([]), y, jnp.array([]))
    assert empty(2.) == 2.
    assert jax.grad(empty)(2.) == 1.
