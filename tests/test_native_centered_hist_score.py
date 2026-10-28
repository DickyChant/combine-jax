from array import array
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from native_centered_hist_score import CenteredHistogramScore, centered_adjoint
from native_cached_deltas import CachedDeltas


def test_centering_preserves_score_for_normalized_shape():
    ld = np.longdouble
    volume = np.array([.5, 2., 3.], dtype=ld)
    raw = np.array([2., 3., 5.], dtype=ld)
    slope = np.array([1., -2., 4.], dtype=ld)
    norm = sum(volume*raw)
    density = raw/norm
    derivative = (slope-density*sum(volume*slope))/norm
    observed = np.array([3., 2., 4.], dtype=ld)
    adjoint = centered_adjoint(17*density, observed, density, volume)
    np.testing.assert_allclose(sum(adjoint*derivative), sum(observed/(17*density)*derivative), rtol=1e-17)
    # A stationary Poisson mean yields a zero adjoint before taking differences.
    zero = centered_adjoint(17*density, 17*density*volume, density, volume)
    assert max(abs(zero)) < 1e-18


def test_centering_rejects_unproved_normalization():
    with pytest.raises(ValueError, match='unit full-grid'):
        centered_adjoint([1, 2], [1, 2], [.2, .3], [1, 1])
    with pytest.raises(ValueError, match='Nonpositive counted'):
        centered_adjoint([0, 2], [1, 2], [.5, .5], [1, 1])
    with pytest.raises(ValueError, match='Mismatched'):
        centered_adjoint([1, 2], [1], [.5, .5], [1, 1])


@pytest.mark.parametrize('mode', ['normal', 'gaussian', 'conditional', 'missing_row', 'floating_coefficient', 'restricted_range'])
def test_native_full_grid_score_and_rejections(root, mode):
    from combine_jax.root_io import load_caching_nll
    root = load_caching_nll()
    x = root.RooRealVar('score_x', '', .3, 0., 3.)
    y = root.RooRealVar('score_y', '', .5, -1., 2.)
    xe, ye = [0., .5, 3.], [-1., 0., 2.]
    x.setBinning(root.RooBinning(2, array('d', xe)))
    y.setBinning(root.RooBinning(2, array('d', ye)))
    morph = root.RooRealVar('score_morph', '', .2, -3., 3.)
    templates, held = root.TList(), []
    for i, bins in enumerate(([2., 3., 5., 7.], [3., 2., 6., 8.], [1., 4., 4., 6.])):
        hist = root.TH2D('score_hist_'+str(i), '', 2, array('d', xe), 2, array('d', ye))
        hist.SetDirectory(0)
        for j, value in enumerate(bins):
            hist.SetBinContent(j//2+1, j%2+1, value)
        templates.Add(hist); held.append(hist)
    shape = root.FastVerticalInterpHistPdf2D2('score_shape', '', x, y, mode == 'conditional',
        templates, root.RooArgList(morph), 1., 1)
    rate = root.RooRealVar('score_rate', '', 17., .1, 30.)
    add = root.RooAddPdf('score_add', '', root.RooArgList(shape), root.RooArgList(rate))
    if mode == 'gaussian':
        mean = root.RooConstVar('score_mean', '', 0.)
        sigma = root.RooConstVar('score_sigma', '', 1.)
        constraint = root.SimpleGaussianConstraint('score_constraint', '', morph, mean, sigma)
        channel = root.RooProdPdf('score_joint', '', root.RooArgList(add, constraint))
    else:
        channel = add
    category = root.RooCategory('score_channel', '')
    category.defineType('one'); category.setLabel('one')
    simultaneous = root.RooSimultaneous('score_sim', '', category)
    simultaneous.addPdf(channel, 'one')
    weight = root.RooRealVar('score_weight', '', 1., 0., 100.)
    obs = root.RooArgSet(x, y, category, weight)
    data = root.RooDataSet('score_data', '', obs, root.RooFit.WeightVar(weight))
    for j, (xv, yv) in enumerate((a, b) for a in (.25, 1.75) for b in (-.5, 1.)):
        if mode == 'missing_row' and j == 3:
            continue
        x.setVal(xv); y.setVal(yv); data.add(obs, [3., 8., 4., 5.][j])
    if mode == 'restricted_range':
        x.setRange(.1, 3.)
    nuisances = root.RooArgSet(morph)
    nll = root.cacheutils.CachingSimNLL(simultaneous, data, nuisances)
    nll.clearConstantZeroPoint(); nll.clearZeroPoint(); nll.getVal()
    stable = CachedDeltas(root, nll)
    score = CenteredHistogramScore(stable)
    if mode not in ('normal', 'gaussian'):
        messages = dict(conditional='unconditional', missing_row='complete histogram grid',
                        floating_coefficient='Floating mixture', restricted_range='Restricted')
        with pytest.raises(NotImplementedError, match=messages[mode]):
            score.reset(rate if mode == 'floating_coefficient' else morph)
        return
    for point in (0., .2, -.4, 1.5):
        morph.setVal(point); score.reset(morph)
        def fd(function, step):
            samples = []
            try:
                for multiple in (-2, -1, 1, 2):
                    morph.setVal(point+multiple*step); samples.append(function())
            finally:
                morph.setVal(point)
            return (samples[0]-8*samples[1]+8*samples[2]-samples[3])/(12*step)
        actual = fd(score.value, 1e-3)
        expected = fd(lambda: float(nll.getVal()), 1e-3)
        assert abs(actual-expected)/max(abs(expected), 1e-8) < 1e-7
        assert stable.max_delta_parity_error < 1e-10
        assert score.value() == 0.
