"""Check actual cached mixture components against ROOT densities and NLL changes."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from cached_root_intensities import CachedRootIntensities
from combine_jax.root_io import load_caching_nll


@pytest.mark.parametrize("include_zero", (False, True))
def test_nested_cached_mixture_matches_root(include_zero):
    r = load_caching_nll()
    label = "cached_"+str(include_zero)
    x = r.RooRealVar(label+"x", "", -3., 4.)
    mean = r.RooRealVar(label+"mean", "", .2, -1., 1.)
    sigma = r.RooConstVar(label+"sigma", "", 1.2)
    signal = r.RooGaussian(label+"signal", "", x, mean, sigma)
    uniform = r.RooUniform(label+"flat", "", r.RooArgSet(x))
    fraction = r.RooRealVar(label+"fraction", "", .3, .1, .8)
    nested = r.RooAddPdf(label+"nested", "", r.RooArgList(signal, uniform), r.RooArgList(fraction))
    ns = r.RooRealVar(label+"ns", "", 12., 1., 30.)
    nb = r.RooConstVar(label+"nb", "", 8.)
    pdf = r.RooAddPdf(label+"pdf", "", r.RooArgList(nested, uniform), r.RooArgList(ns, nb))
    weight = r.RooRealVar(label+"weight", "", 1., 0., 100.)
    axes = r.RooArgSet(x)
    columns = r.RooArgSet(x, weight)
    data = r.RooDataSet(label+"data", "", columns, r.RooFit.WeightVar(weight))
    positions = [-2., -.5, .5, 2.]
    weights = [.2, 0., 2.5, 3.]
    for coordinate, w in zip(positions, weights, strict=True):
        # CachingSimNLL prefilters zero rows for its ordinary channel path.
        # With a generic CachingPdf, directly constructing CachingAddNLL on
        # unfiltered zero rows and includeZeroWeights=False is unsafe in v11:
        # its generic PDF vector retains all rows but weights_ does not.
        if include_zero or w != 0:
            x.setVal(coordinate); weight.setVal(w); data.add(columns, w)
    nll = r.cacheutils.CachingAddNLL(label+"nll", "", pdf, data, include_zero)
    nll.clearConstantZeroPoint(); nll.clearZeroPoint()
    predictor = CachedRootIntensities(r, nll)
    baseline = predictor.predict()
    assert predictor.report["components"] == 3
    nll.setZeroPoint()
    for shift, f, yield_value in ((.2,.3,12.), (-.4,.5,20.), (.7,.2,5.)):
        mean.setVal(shift); fraction.setVal(f); ns.setVal(yield_value)
        expected, intensity = predictor.predict()
        assert abs(expected-(yield_value+8)) < 1e-13
        direct = []
        for coordinate, w in zip(positions, weights, strict=True):
            if include_zero or w != 0:
                x.setVal(coordinate)
                direct.append((yield_value+8)*pdf.getVal(axes))
        np.testing.assert_allclose(intensity, direct, rtol=2e-15, atol=2e-15)
        assert abs(float(predictor.delta(baseline))-float(nll.getVal())) < 1e-12
    del predictor, nll
