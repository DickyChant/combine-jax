from array import array

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import UnsupportedModelError, parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def make_histfunc(root, *, log=False, active=False, divide=True, rebin=None):
    edges = [0., .5, 2., 4., 7.]
    x = root.RooRealVar("hx", "", 1., edges[0], edges[-1])
    observable_edges = edges if rebin is None else rebin
    x.setBinning(root.RooBinning(len(observable_edges)-1, array("d", observable_edges)))
    a = root.RooRealVar("a", "", 0., -3., 3.)
    b = root.RooRealVar("b", "", 0., -3., 3.)
    values = [[2., 5., 0., 0.], [1., 6., 0., 0.], [3.5, 4., 0., 0.],
              [1.5, 4., .3, 0.], [2.5, 6., .2, 0.]]
    histograms = []
    for i, row in enumerate(values):
        hist = root.TH1D(f"stored_{i}", "", 4, array("d", edges))
        hist.SetDirectory(0)
        for j, v in enumerate(row, 1):
            hist.SetBinContent(j, v)
        histograms.append(hist)
    histfunc = root.CMSHistFunc("histfunc", "", x, histograms[0], divide)
    histfunc.setVerticalMorphs(root.RooArgList(a, b))
    histfunc.setVerticalSmoothRegion(.7)
    if log:
        histfunc.setVerticalType(root.CMSHistFunc.LogQuadLinear)
    histfunc.prepareStorage()
    for i, hist in enumerate(histograms[1:]):
        histfunc.setShape(0, 0, i//2 + 1, i%2, hist)
    if active:
        histfunc.setActiveBins(3)
    return x, a, b, histfunc


@pytest.mark.parametrize("log,active,divide,rebin", [
    (False,False,True,None), (False,True,True,None), (False,False,False,None),
    (True,False,True,None), (True,True,True,None), (True,False,False,None),
]+[(log,active,divide,edges) for log in (False,True) for active in (False,True)
   for divide in (False,True) for edges in ((0.,2.,7.),(0.,1.25,4.,7.))])
def test_cms_histfunc_values_integrals_and_derivatives(root, log, active, divide, rebin):
    x, a, b, func = make_histfunc(root, log=log, active=active, divide=divide,rebin=rebin)
    registry = default_registry()
    graph = registry.export([func], ["hx"])
    coords = np.array([0., .25, .5, .500001, 1.25, 2., 3., 4., 5.5, 7.])
    if rebin is not None:
        coords = np.unique(np.concatenate((coords,rebin,np.array(rebin[:-1])+1e-6)))
        assert graph["histfunc"].attrs["active"] == len(rebin)-1
        assert len(graph["histfunc"].attrs["rebin_indices"]) == 4

    def value(theta):
        c = Evaluation(graph, registry, ("a", "b"), theta, {"hx": coords})
        return c.value("histfunc")

    def integral(theta):
        return Evaluation(graph, registry, ("a", "b"), theta).integral("histfunc", ("hx",))

    value, integral = jax.jit(value), jax.jit(integral)
    gradient = jax.jit(jax.grad(integral))
    root_integral = func.createIntegral(root.RooArgSet(x))

    def ref_integral(theta):
        a.setVal(float(theta[0])); b.setVal(float(theta[1]))
        return float(root_integral.getVal())

    points = [np.array([t, -.31]) for t in (-2., -.70001, -.7, -.69999, 0., .69999, .7, .70001, 2.)]
    points.extend(np.random.default_rng(2026).uniform(-2., 2., (12, 2)))
    for theta in points:
        ref_int = ref_integral(theta)
        expected = []
        for coord in coords:
            x.setVal(float(coord)); expected.append(float(func.getVal()))
        np.testing.assert_allclose(value(theta), expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(integral(theta), ref_int, atol=2e-14, rtol=2e-14)
        fd = finite_difference(ref_integral, theta, np.full(2, 1e-5))
        np.testing.assert_allclose(gradient(theta), fd, atol=2e-9, rtol=1e-6)


def test_histfunc_horizontal_morph_is_explicitly_rejected(root):
    x, a, b, func = make_histfunc(root)
    mass = root.RooRealVar("mass", "", 1., 0., 3.)
    points = root.TVectorD(2)
    points[0], points[1] = 1., 2.
    func.addHorizontalMorph(mass, points)
    with pytest.raises(UnsupportedModelError, match="histfunc.*horizontal morphing"):
        default_registry().export([func], ["hx"])


@pytest.mark.parametrize("rebin,binned", [(None,False),((0.,2.,7.),False),((0.,1.25,4.,7.),False),
                                         ((0.,2.,7.),True),((0.,1.25,4.,7.),True)])
def test_real_sum_histfunc_nll_against_combine(root,rebin,binned):
    root = load_caching_nll()
    x, a, b, func = make_histfunc(root,rebin=rebin)
    coefficient = root.RooRealVar("scale", "", 1., .1, 4.)
    pdf = root.RooRealSumPdf("real_sum", "", root.RooArgList(func), root.RooArgList(coefficient), True)
    weight = root.RooRealVar("hist_weight", "", 1., 0., 20.)
    variables = root.RooArgSet(x, weight)
    data = root.RooDataSet("hist_data", "", variables, root.RooFit.WeightVar(weight))
    rows = [(.25,2.3),(1.25,4.7),(3.,0.),(5.5,0.)]
    if binned:
        counts = np.zeros(len(rebin)-1)
        for coordinate,w in rows:
            counts[np.searchsorted(rebin,coordinate,side="left")-1] += w
        rows = list(zip((np.array(rebin[1:])+rebin[:-1])/2,counts,strict=True))
    for coordinate, w in rows:
        x.setVal(coordinate); data.add(variables, w)
    registry = default_registry()
    graph = registry.export([pdf], ["hx"])
    likelihood = Likelihood(graph, registry, parameters(pdf, ["hx"]),
                            [Channel("hist", "real_sum", ("hx",), bind_data(data, ["hx"]))], [])
    reference = root.cacheutils.CachingAddNLL("real_sum_reference", "", pdf, data, True)
    reference.clearConstantZeroPoint()
    reference.clearZeroPoint()

    def ref(theta):
        a.setVal(float(theta[0])); b.setVal(float(theta[1])); coefficient.setVal(float(theta[2]))
        return float(reference.getVal())

    for theta in np.random.default_rng(177).uniform([-2., -2., .5], [2., 2., 2.], (20, 3)):
        value, gradient = likelihood.value_and_grad(theta)
        np.testing.assert_allclose(value, ref(theta), atol=1e-6, rtol=0)
        # Rebinning can make a shape-only nuisance exactly invisible. Fixed
        # larger steps avoid cancellation in the ROOT NLL near that zero slope.
        steps = (1e-2,5e-3) if rebin else (1e-4,)
        references = [finite_difference(ref,theta,np.full(3,step)) for step in steps]
        if rebin:
            assert np.max(abs(references[0]-references[1])/np.maximum(abs(references[1]),1e-8)) < 1e-4
        for fd in references:
            assert np.max(np.abs(np.asarray(gradient)-fd) / np.maximum(np.abs(fd), 1e-8)) < 1e-4


@pytest.mark.parametrize("log", (False,True))
def test_rebinned_histfunc_in_error_propagator(root,log):
    x,a,b,func = make_histfunc(root,log=log,rebin=(0.,1.25,4.,7.))
    scale = root.RooRealVar("rebin_scale","",1.,.1,3.)
    prop = root.CMSHistErrorPropagator("rebin_prop","",x,root.RooArgList(func),root.RooArgList(scale))
    binpars = prop.setupBinPars(0.)
    assert binpars.getSize() > 0
    registry = default_registry()
    graph = registry.export([prop],["hx"])
    info = parameters(prop,["hx"])
    names = tuple(p.name for p in info)
    variables = {str(v.GetName()):v for v in [a,b,scale,*binpars]}
    positions = np.array([.625,2.625,5.5])
    integral_ref = prop.createIntegral(root.RooArgSet(x))

    def evaluate(theta):
        return Evaluation(graph,registry,names,theta,{"hx":positions}).value("rebin_prop")

    def integral(theta):
        return Evaluation(graph,registry,names,theta).integral("rebin_prop",("hx",))

    evaluate,integral,gradient = jax.jit(evaluate),jax.jit(integral),jax.jit(jax.grad(integral))

    def reference(theta):
        for name,value in zip(names,theta,strict=True):
            variables[name].setVal(float(value))
        return float(integral_ref.getVal())

    for theta in np.random.default_rng(2381).uniform(-1.,1.,(12,len(names))):
        theta[names.index("rebin_scale")] = 1.+theta[names.index("rebin_scale")]/2
        expected_integral = reference(theta)
        expected = []
        for coordinate in positions:
            x.setVal(float(coordinate)); expected.append(float(prop.getVal()))
        np.testing.assert_allclose(evaluate(theta),expected,atol=2e-12,rtol=2e-12)
        np.testing.assert_allclose(integral(theta),expected_integral,atol=2e-12,rtol=2e-12)
        fd = finite_difference(reference,theta,np.full(len(names),1e-4))
        np.testing.assert_allclose(gradient(theta),fd,atol=2e-9,rtol=1e-6)
