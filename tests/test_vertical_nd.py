from array import array
from itertools import product

import jax
import numpy as np
import pytest

from combine_jax.data import bind_data
from combine_jax.graph import parameters
from combine_jax.likelihood import Channel, Likelihood
from combine_jax.registry import Evaluation
from combine_jax.root_io import load_caching_nll
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


CLASSES = ("FastVerticalInterpHistPdf2D", "FastVerticalInterpHistPdf2D2", "FastVerticalInterpHistPdf3D")


def make_multidimensional(root, cls, conditional=False, logarithmic=False):
    dimension = 3 if cls.endswith("3D") else 2
    edges = [np.array([0., .12345678901234567, 2., 4.]), np.array([-2., -.3, 1.]),
             np.array([1., 2.5, 3.5, 6.])][:dimension]
    variables, hist_args = [], []
    for name, edge in zip(("ndx", "ndy", "ndz"), edges):
        variable = root.RooRealVar(name, "", float((edge[1]+edge[2])/2), float(edge[0]), float(edge[-1]))
        variable.setBinning(root.RooBinning(len(edge)-1, array("d", edge)))
        variables.append(variable)
        hist_args.extend((len(edge)-1, array("d", edge)))
    a = root.RooRealVar("nda", "", 0., -3., 3.)
    b = root.RooRealVar("ndb", "", 0., -3., 3.)
    held, templates, functions = [], root.TList(), root.RooArgList()
    shape = tuple(len(e)-1 for e in edges)
    for i in range(5):
        hist = getattr(root, f"TH{dimension}D")(f"ndhist_{i}", "", *hist_args)
        hist.SetDirectory(0)
        for j, cell in enumerate(np.ndindex(shape)):
            value = (j+1)*(.5 + np.cos(.1*i*j)**2) if cell[0] else 0.
            hist.SetBinContent(*(k+1 for k in cell), value)
        templates.Add(hist)
        held.append(hist)
        if cls != "FastVerticalInterpHistPdf2D2":
            data = root.RooDataHist(f"nddata_{i}", "", root.RooArgList(*variables), hist)
            component = root.RooHistPdf(f"ndcomponent_{i}", "", root.RooArgSet(*variables), data, 0)
            functions.add(component); held.extend((data, component))
    pdf = getattr(root, cls)("ndpdf", "", *variables, conditional,
                            templates if cls == "FastVerticalInterpHistPdf2D2" else functions,
                            root.RooArgList(a, b), .7, -1 if logarithmic else 1)
    return variables, a, b, pdf, edges, held


def root_values(root, pdf, variables, coordinates, theta, a, b):
    a.setVal(float(theta[0])); b.setVal(float(theta[1]))
    norm = root.RooArgSet(*variables)
    values = []
    for row in coordinates:
        for variable, value in zip(variables, row, strict=True): variable.setVal(float(value))
        values.append(float(pdf.getVal(norm)))
    return np.array(values)


@pytest.mark.parametrize("cls,conditional,logarithmic", list(product(CLASSES, (False, True), (False, True))))
def test_multidimensional_values_integrals_and_derivatives(root, cls, conditional, logarithmic):
    variables, a, b, pdf, edges, held = make_multidimensional(root, cls, conditional, logarithmic)
    registry = default_registry()
    names = tuple(str(v.GetName()) for v in variables)
    nodes = registry.export([pdf], names)
    for actual, original in zip(nodes["ndpdf"].attrs["edges"], edges, strict=True):
        np.testing.assert_array_equal(actual, original)
    coordinates = np.array(list(product(*(np.sort(np.r_[e, (e[:-1]+e[1:])/2]) for e in edges))))
    bound = {name: coordinates[:, i] for i, name in enumerate(names)}

    def values(theta):
        return Evaluation(nodes, registry, ("nda", "ndb"), theta, bound).density("ndpdf", names)

    values = jax.jit(values)
    gradient = jax.jit(jax.jacrev(values))
    centers = np.array(list(product(*((e[:-1]+e[1:])/2 for e in edges))))
    shape = tuple(len(e)-1 for e in edges)
    points = [np.array([v, -.31]) for v in (-2., -.7, 0., .7, 2.)]
    points.extend(np.random.default_rng(6001).uniform(-2., 2., (12, 2)))
    fixed = {name: float((e[1]+e[2])/2) for name, e in zip(names, edges, strict=True)}
    for theta in points:
        ref = lambda t: root_values(root, pdf, variables, coordinates, t, a, b)
        np.testing.assert_allclose(values(theta), ref(theta), atol=8e-14, rtol=8e-14, equal_nan=False)
        fd = finite_difference(ref, theta, np.full(2, 1e-5)).T
        np.testing.assert_allclose(gradient(theta), fd, atol=3e-9, rtol=3e-6, equal_nan=False)
        table = root_values(root, pdf, variables, centers, theta, a, b).reshape(shape)
        context = Evaluation(nodes, registry, ("nda", "ndb"), theta, fixed)
        # Check every nonempty integration subset, using exact bin quadrature of
        # independently evaluated ROOT bins and fixed remaining coordinates.
        for mask in range(1, 2**len(names)):
            axes = tuple(n for i, n in enumerate(names) if mask & (1 << i))
            expected = 0.
            for index in np.ndindex(shape):
                if any(index[i] != 1 for i, n in enumerate(names) if n not in axes):
                    continue
                volume = np.prod([edges[i][index[i]+1]-edges[i][index[i]]
                                  for i, n in enumerate(names) if n in axes])
                expected += table[index]*volume
            np.testing.assert_allclose(context.integral("ndpdf", axes), expected, atol=8e-14, rtol=8e-14, equal_nan=False)


@pytest.mark.parametrize("cls,conditional,logarithmic", list(product(CLASSES, (False, True), (False, True))))
def test_multidimensional_extended_nll_against_combine(root, cls, conditional, logarithmic):
    root = load_caching_nll()
    variables, a, b, shape, edges, held = make_multidimensional(root, cls, conditional, logarithmic)
    rate = root.RooRealVar("ndrate", "", 7., .1, 30.)
    pdf = root.RooAddPdf("ndsum", "", root.RooArgList(shape), root.RooArgList(rate))
    weight = root.RooRealVar("ndweight", "", 1., 0., 100.)
    collection = root.RooArgSet(*variables, weight)
    data = root.RooDataSet("nddata", "", collection, root.RooFit.WeightVar(weight))
    for i, row in enumerate(product(*((e[:-1]+e[1:])/2 for e in edges))):
        for variable, value in zip(variables, row, strict=True): variable.setVal(float(value))
        # Include the identically zero x slice with zero observation weights.
        data.add(collection, 0. if row[0] < edges[0][1] else .3+i*.2)
    names = tuple(str(v.GetName()) for v in variables)
    registry = default_registry()
    nodes = registry.export([pdf], names)
    likelihood = Likelihood(nodes, registry, parameters(pdf, names),
                            [Channel("nd", "ndsum", names, bind_data(data, names))], [])
    reference = root.cacheutils.CachingAddNLL("ndreference", "", pdf, data, True)
    reference.clearConstantZeroPoint(); reference.clearZeroPoint()

    def ref(theta):
        a.setVal(float(theta[0])); b.setVal(float(theta[1])); rate.setVal(float(theta[2]))
        return float(reference.getVal())

    points = [likelihood.initial]
    points.extend(np.random.default_rng(6002).uniform([-2., -2., 4.], [2., 2., 10.], (25, 3)))
    for theta in points:
        value, gradient = likelihood.value_and_grad(theta)
        assert np.isfinite(value) and np.all(np.isfinite(gradient))
        np.testing.assert_allclose(value, ref(theta), atol=1e-6, rtol=0, equal_nan=False)
        fd = finite_difference(ref, theta, np.full(3, 1e-4))
        assert np.max(np.abs(gradient-fd)/np.maximum(np.abs(fd), 1e-8)) < 1e-4
    fd_hessian = finite_difference(lambda t: np.asarray(likelihood.value_and_grad(t)[1]), points[1], np.full(3, 1e-4)).T
    np.testing.assert_allclose(likelihood.hessian(points[1]), fd_hessian, atol=2e-7, rtol=2e-6)


def test_stored_2d_integral_with_restricted_ranges(root):
    variables, a, b, pdf, edges, held = make_multidimensional(root, "FastVerticalInterpHistPdf2D2", True, True)
    variables[0].setRange(.3, 3.)
    variables[1].setRange(-1., .7)
    registry = default_registry()
    names = tuple(str(v.GetName()) for v in variables)
    nodes = registry.export([pdf], names)
    theta = np.array([.4, -.8])
    context = Evaluation(nodes, registry, ("nda", "ndb"), theta)
    expected = 0.
    a.setVal(theta[0]); b.setVal(theta[1])
    for ix, iy in product(range(3), range(2)):
        low = [max(e[i], v.getMin()) for e, i, v in zip(edges, (ix, iy), variables, strict=True)]
        high = [min(e[i+1], v.getMax()) for e, i, v in zip(edges, (ix, iy), variables, strict=True)]
        if min(np.subtract(high, low)) <= 0:
            continue
        for v, lo, hi in zip(variables, low, high, strict=True): v.setVal((lo+hi)/2)
        expected += pdf.getVal()*np.prod(np.subtract(high, low))
    np.testing.assert_allclose(context.integral("ndpdf", names), expected, atol=3e-14, rtol=3e-14)
