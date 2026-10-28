from array import array
from itertools import product

import jax
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry


def make_hist_pdf(root, dimension=1, remap=False):
    edges = [np.array([0., .5, 2., 4.]), np.array([-2., -.3, 1.]), np.array([1., 2.5, 3.5])][:dimension]
    args, variables = [], []
    for i, edge in enumerate(edges):
        variables.append(root.RooRealVar(f"hpdf_{i}", "", float(edge[0]), float(edge[0]), float(edge[-1])))
        args.extend((len(edge)-1, array("d", edge)))
    hist = getattr(root, f"TH{dimension}D")("hpdf_hist", "", *args)
    hist.SetDirectory(0)
    shape = tuple(len(e)-1 for e in edges)
    for i, cell in enumerate(np.ndindex(shape)):
        hist.SetBinContent(*(j+1 for j in cell), float(i+1) if i != 1 else -.3)
    data = root.RooDataHist("hpdf_data", "", root.RooArgList(*variables), hist)
    if remap:
        mapped = [root.RooRealVar(f"mapped_{i}", "", float(e[0]), float(e[0]), float(e[-1])) for i, e in enumerate(edges)]
        pdf = root.RooHistPdf("hpdf", "", root.RooArgList(*mapped), root.RooArgList(*variables), data, 0)
    else:
        mapped = variables
        pdf = root.RooHistPdf("hpdf", "", root.RooArgSet(*variables), data, 0)
    return mapped, pdf, edges, (variables, data, hist)


@pytest.mark.parametrize("dimension,remap", [(1, False), (2, False), (3, False), (1, True), (2, True)])
def test_hist_pdf_values_edges_and_integrals(root, dimension, remap):
    variables, pdf, edges, held = make_hist_pdf(root, dimension, remap)
    registry = default_registry()
    names = tuple(str(v.GetName()) for v in variables)
    nodes = registry.export([pdf], names)
    # Test each edge, bin interior, and both histogram boundaries.
    coordinates = np.array(list(product(*(np.sort(np.r_[e, (e[1:]+e[:-1])/2]) for e in edges))))
    bound = {name: coordinates[:, i] for i, name in enumerate(names)}
    context = Evaluation(nodes, registry, (), (), bound)
    expected = []
    normset = root.RooArgSet(*variables)
    integral = pdf.createIntegral(normset)
    raw = []
    for row in coordinates:
        for variable, value in zip(variables, row, strict=True): variable.setVal(float(value))
        raw.append(float(pdf.getVal()))
        expected.append(float(pdf.getVal(normset)))
    np.testing.assert_allclose(jax.jit(lambda: Evaluation(nodes, registry, (), (), bound).value("hpdf"))(),
                               raw, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(context.integral("hpdf", names), integral.getVal(), atol=1e-13, rtol=1e-14)
    np.testing.assert_allclose(context.density("hpdf", names), expected, atol=1e-14, rtol=1e-14)


@pytest.mark.parametrize("option", ["interpolation", "cdf", "unit_norm"])
def test_unsupported_hist_pdf_modes(root, option):
    variables, pdf, edges, held = make_hist_pdf(root)
    if option == "interpolation": pdf.setInterpolationOrder(1)
    elif option == "cdf": pdf.setCdfBoundaries(True)
    else: pdf.setUnitNorm(True)
    with pytest.raises(UnsupportedModelError, match="hpdf.*interpolation, CDF boundaries and unit-norm"):
        default_registry().export([pdf], [v.GetName() for v in variables])


def test_partial_hist_pdf_normalisation_rejected(root):
    variables, pdf, edges, held = make_hist_pdf(root, 2)
    registry = default_registry()
    names = tuple(str(v.GetName()) for v in variables)
    nodes = registry.export([pdf], names)
    with pytest.raises(UnsupportedModelError, match="partial RooHistPdf integral"):
        Evaluation(nodes, registry, (), ()).integral("hpdf", names[:1])


def test_restricted_hist_pdf_range_rejected(root):
    variables, pdf, edges, held = make_hist_pdf(root)
    variables[0].setRange(.25, 3.)
    registry = default_registry()
    names = tuple(str(v.GetName()) for v in variables)
    nodes = registry.export([pdf], names)
    with pytest.raises(UnsupportedModelError, match="restricted RooHistPdf integral range"):
        Evaluation(nodes, registry, (), ()).integral("hpdf", names)
