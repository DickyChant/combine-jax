from array import array

import jax
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import UnsupportedModelError, parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_shape(root, family):
    x = root.RooRealVar("sx", "", .4, .1, 5.)
    axes = [x]
    if family == "power":
        coefficients = [root.RooRealVar("sc0", "", -1.3, -2.5, .5)]
        pdf = root.RooPower("shape", "", x, coefficients[0])
        coordinates = {"sx": np.linspace(.15, 4.9, 11)}
    elif family == "step":
        edges = [.1, .7, 2., 5.]
        limits = root.TArrayD(len(edges))
        for i,v in enumerate(edges): limits[i] = v
        coefficients = [root.RooRealVar(f"sc{i}", "", .1, .01, .25) for i in range(2)]
        pdf = root.RooParametricStepFunction("shape", "", x, root.RooArgList(*coefficients), limits, 3)
        coordinates = {"sx": np.linspace(.15, 4.9, 11)}
    else:
        y = root.RooRealVar("sy", "", .4, -1., 4.)
        axes.append(y)
        histogram = root.TH2D("shape_geometry", "", 3, array("d", [.1,.7,2.,5.]), 2, array("d", [-1.,1.5,4.]))
        histogram.SetDirectory(0)
        coefficients = [root.RooRealVar(f"sc{i}", "", i+1., .2, 8.) for i in range(6)]
        pdf = root.RooParametricHist2D("shape", "", x, y, root.RooArgList(*coefficients), histogram)
        coordinates = {"sx": np.array([.4, 1.1, 3.5, .4, 1.1, 3.5]), "sy": np.array([0., 0., 0., 3., 3., 3.])}
    return locals()


@pytest.mark.parametrize("family", ["power", "step", "hist2d"])
def test_remaining_shape_values_integrals_and_gradients(root, family):
    model = make_shape(root, family)
    pdf, axes, coefficients, coordinates = (model[n] for n in ("pdf", "axes", "coefficients", "coordinates"))
    axis_names = tuple(v.GetName() for v in axes); names = tuple(v.GetName() for v in coefficients)
    registry = default_registry(); nodes = registry.export([pdf], axis_names)
    norm = root.RooArgSet(*axes); integral = pdf.createIntegral(norm)
    def evaluate(theta):
        ctx = Evaluation(nodes, registry, names, theta, coordinates)
        return ctx.value("shape"), ctx.integral("shape", axis_names), ctx.density("shape", axis_names)
    evaluate, gradient = jax.jit(evaluate), jax.jit(jax.jacrev(lambda t: evaluate(t)[2]))
    def reference(theta):
        for v,p in zip(coefficients,theta,strict=True): v.setVal(float(p))
        raw,density=[],[]
        for i in range(len(next(iter(coordinates.values())))):
            for axis in axes: axis.setVal(float(coordinates[axis.GetName()][i]))
            raw.append(pdf.getVal()); density.append(pdf.getVal(norm))
        return np.array(raw),integral.getVal(),np.array(density)
    for theta in np.random.default_rng(7611).uniform([v.getMin() for v in coefficients], [v.getMax() for v in coefficients], (25,len(coefficients))):
        for actual,expected in zip(evaluate(theta),reference(theta),strict=True):
            np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=2e-12,equal_nan=False)
        fd=finite_difference(lambda t:reference(t)[2],theta,np.full(len(theta),1e-4)).T
        np.testing.assert_allclose(gradient(theta),fd,atol=1e-9,rtol=1e-6,equal_nan=False)


@pytest.mark.parametrize("exponent", [-2., -1., -1.+1e-7, -1.-1e-7, 0., 1.])
def test_power_limit_gradients_and_hessian(root, exponent):
    model=make_shape(root,"power"); x=model["x"]; c=model["coefficients"][0]; pdf=model["pdf"]
    c.setRange(-3.,2.); c.setVal(exponent)
    registry=default_registry(); nodes=registry.export([pdf])
    f=jax.jit(lambda t: Evaluation(nodes,registry,("sc0",),[t]).integral("shape",("sx",)))
    norm=root.RooArgSet(x); integral=pdf.createIntegral(norm)
    np.testing.assert_allclose(f(exponent),integral.getVal(),atol=2e-8,rtol=2e-9,equal_nan=False)
    def ref(point): c.setVal(float(point[0]));return integral.getVal()
    fd=finite_difference(ref,np.array([exponent]),np.array([1e-3]))[0]
    np.testing.assert_allclose(jax.grad(f)(exponent),fd,atol=1e-8,rtol=2e-8,equal_nan=False)
    fdh=finite_difference(lambda t:float(jax.grad(f)(t[0])),np.array([exponent]),np.array([1e-3]))[0]
    np.testing.assert_allclose(jax.hessian(f)(exponent),fdh,atol=1e-8,rtol=2e-8,equal_nan=False)


def test_power_zero_lower_endpoint_and_unsupported_range(root):
    model=make_shape(root,"power");x=model["x"];c=model["coefficients"][0];pdf=model["pdf"]
    x.setRange(0.,5.);registry=default_registry();nodes=registry.export([pdf]);norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    for exponent in [-2.,-1.,-.5,0.,.4]:
        c.setVal(exponent);actual=Evaluation(nodes,registry,("sc0",),[exponent]).integral("shape",("sx",))
        np.testing.assert_allclose(actual,integral.getVal(),atol=1e-13,rtol=1e-13)
    x.setRange(-1.,5.);nodes=registry.export([pdf])
    with pytest.raises(UnsupportedModelError,match="nonnegative"):
        Evaluation(nodes,registry,(),()).integral("shape",("sx",))


@pytest.mark.parametrize("family", ["step", "hist2d"])
def test_parametric_bin_edges_and_default_integral_conventions(root, family):
    model=make_shape(root,family);pdf=model["pdf"];axes=model["axes"]
    axes[0].setRange(-1.,6.)
    if family=="hist2d":axes[1].setRange(-2.,5.)
    registry=default_registry();nodes=registry.export([pdf],[v.GetName() for v in axes]);names=tuple(v.GetName() for v in axes)
    norm=root.RooArgSet(*axes);integral=pdf.createIntegral(norm)
    for coordinate in [-1.,.1,np.nextafter(.7,-np.inf),.7,2.,np.nextafter(5.,-np.inf),5.,6.]:
        axes[0].setVal(float(coordinate))
        for y in ([-2.,-1.,1.5,4.,5.] if family=="hist2d" else [None]):
            if y is not None:axes[1].setVal(y)
            data={v.GetName():float(v.getVal()) for v in axes}
            ctx=Evaluation(nodes,registry,(),(),data)
            np.testing.assert_allclose(ctx.value("shape"),pdf.getVal(),atol=1e-13,rtol=1e-13,equal_nan=False)
    axes[0].setRange(.4,1.1)
    nodes=registry.export([pdf]);restricted=pdf.createIntegral(norm)
    np.testing.assert_allclose(Evaluation(nodes,registry,(),()).integral("shape",names),restricted.getVal(),atol=1e-13)
    if family=="hist2d":
        with pytest.raises(UnsupportedModelError,match="joint integral"):
            Evaluation(nodes,registry,(),()).integral("shape",("sx",))


def test_step_final_bin_floor_keeps_advertised_integral(root):
    model=make_shape(root,"step");x=model["x"];pdf=model["pdf"]
    for v in model["coefficients"]:v.setRange(0.,2.);v.setVal(1.)
    x.setVal(3.);registry=default_registry();nodes=registry.export([pdf]);ctx=Evaluation(nodes,registry,(),())
    norm=root.RooArgSet(x);integral=pdf.createIntegral(norm)
    assert float(ctx.value("shape"))==pdf.getVal()==1e-6
    assert float(ctx.integral("shape",("sx",)))==integral.getVal()==1.


@pytest.mark.parametrize("count", [1,2,3,6])
def test_recursive_fraction_order_and_derivatives(root,count):
    variables=[root.RooRealVar(f"fraction{i}","",.2+i*.05,-.2,1.2) for i in range(count)]
    f=root.RooRecursiveFraction("recursive","",root.RooArgList(*variables))
    registry=default_registry();nodes=registry.export([f]);names=tuple(v.GetName() for v in variables)
    fn=jax.jit(lambda t: Evaluation(nodes,registry,names,t).value("recursive"))
    def reference(theta):
        for v,p in zip(variables,theta,strict=True):v.setVal(float(p))
        return f.getVal()
    for theta in np.random.default_rng(7612).uniform(-.15,1.15,(25,count)):
        np.testing.assert_allclose(fn(theta),reference(theta),atol=1e-15,rtol=2e-15)
        fd=finite_difference(reference,theta,np.full(count,1e-4))
        np.testing.assert_allclose(jax.grad(fn)(theta),fd,atol=1e-11,rtol=1e-8)


@pytest.mark.parametrize("family", ["power", "step", "hist2d", "recursive"])
def test_remaining_shape_extended_nll(root,family):
    root=load_caching_nll()
    model=make_shape(root,"power" if family=="recursive" else family)
    axes=model["axes"];shape=model["pdf"];extra=[]
    if family=="recursive":
        u=root.RooUniform("recursive_uniform","",root.RooArgSet(*axes))
        gmean=root.RooConstVar("recursive_mean","",2.);gsigma=root.RooConstVar("recursive_sigma","",1.)
        g=root.RooGaussian("recursive_gaussian","",axes[0],gmean,gsigma)
        f1=root.RooRealVar("sf1","",.3,.1,.8);f2=root.RooRealVar("sf2","",.4,.1,.8)
        shape=root.RooAddPdf("recursive_mixture","",root.RooArgList(shape,u,g),root.RooArgList(f1,f2),True)
        extra=[u,gmean,gsigma,g,f1,f2]
    rate=root.RooRealVar("srate","",20.,10.,30.)
    pdf=root.RooAddPdf("shape_sum","",root.RooArgList(shape),root.RooArgList(rate))
    weight=root.RooRealVar("sweight","",1.,0.,100.);collection=root.RooArgSet(*axes,weight)
    data=root.RooDataSet("shape_data","",collection,root.RooFit.WeightVar(weight))
    for i in range(len(model["coordinates"]["sx"])):
        for axis in axes:axis.setVal(float(model["coordinates"][axis.GetName()][i]))
        data.add(collection,0. if i==1 else i*.1+1.)
    names=tuple(v.GetName() for v in axes);registry=default_registry();nodes=registry.export([pdf],names)
    pars=parameters(pdf,names)
    likelihood=Likelihood(nodes,registry,pars,[Channel("new_shapes","shape_sum",names,bind_data(data,names))],[])
    refnll=root.cacheutils.CachingAddNLL("shape_ref","",pdf,data,True)
    refnll.clearConstantZeroPoint();refnll.clearZeroPoint()
    variables={v.GetName():v for v in [*model["coefficients"],rate,*extra]}
    def reference(theta):
        for n,p in zip(likelihood.parameter_names,theta,strict=True):variables[n].setVal(float(p))
        return refnll.getVal()
    for index,theta in enumerate(np.random.default_rng(7613).uniform([p.minimum for p in pars],[p.maximum for p in pars],(25,len(pars)))):
        value,gradient=likelihood.value_and_grad(theta)
        np.testing.assert_allclose(value,reference(theta),atol=1e-6,rtol=0,equal_nan=False)
        fd=finite_difference(reference,theta,np.full(len(pars),1e-4))
        assert np.max(np.abs(gradient-fd)/np.maximum(np.abs(fd),1e-8))<1e-4
        if index==0:
            fdh=finite_difference(lambda t:likelihood.value_and_grad(t)[1],theta,np.full(len(pars),1e-4)).T
            np.testing.assert_allclose(likelihood.hessian(theta),fdh,atol=1e-6,rtol=1e-6,equal_nan=False)
