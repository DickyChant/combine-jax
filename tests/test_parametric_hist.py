"""RooParametricHist bin yields, variable volumes and multiplicative morphs."""
from array import array

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def set_morph_flag(root, pdf, enabled):
    """Construct a standard ROOT object with legacy serialized flag metadata."""
    if not hasattr(root, 'ParametricHistFlagFixture'):
        assert root.gInterpreter.Declare(r'''
          #include "HiggsAnalysis/CombinedLimit/interface/RooParametricHist.h"
          #include <limits>
          struct ParametricHistFlagFixture : public RooParametricHist {
            ParametricHistFlagFixture(RooParametricHist const& p, bool flag)
              : RooParametricHist(p, "flag_fixture") {
              _hasMorphs = flag;
              if (_coeffList.getSize() == 0)
                _smoothRegion = std::numeric_limits<double>::quiet_NaN();
            }
          };
        ''')
    modified = root.ParametricHistFlagFixture(pdf, enabled)
    return root.RooParametricHist(modified, pdf.GetName())


def make_hist(root, region=None):
    edges = np.array([0., .5, 2., 5.])
    x = root.RooRealVar('phx', '', .2, -1., 6.)
    x.setBinning(root.RooBinning(3, array('d', edges)))
    histogram = root.TH1D('ph_geometry', '', 3, array('d', edges))
    histogram.SetDirectory(0)
    bins = [root.RooRealVar(f'phbin{i}', '', 8.+i, 6., 14.) for i in range(3)]
    pdf = root.RooParametricHist('ph', '', x, root.RooArgList(*bins), histogram)
    morphs, held = [], []
    if region == 'empty':
        pdf = set_morph_flag(root, pdf, True)
    elif region is not None:
        for j in range(2):
            nuisance = root.RooRealVar(f'phmorph{j}', '', .23, -1.5, 1.5)
            templates = []
            for label, scale in [('up', .17), ('down', -.11)]:
                h = histogram.Clone(f'ph_{label}{j}')
                h.SetDirectory(0)
                for i, value in enumerate(bins, 1):
                    h.SetBinContent(i, value.getVal()*(1+scale*(j+1)*i/3.))
                data = root.RooDataHist(f'phdata_{label}{j}', '', root.RooArgList(x), h)
                held += [h, data]
                templates.append(data)
            pdf.addMorphs(*templates, nuisance, region)
            morphs.append(nuisance)
    variables = bins+morphs
    coordinates = {'phx': np.array([.1, .4, .5, 1.9, 2., 3., 4.9])}
    return locals()


@pytest.mark.parametrize('region', [None, 0., .7, 'empty'])
def test_parametric_hist_root_values_integral_gradients_and_hessian(root, region):
    fixture = make_hist(root, region)
    pdf, x, variables = [fixture[n] for n in ('pdf', 'x', 'variables')]
    names = tuple(v.GetName() for v in variables)
    registry = default_registry()
    nodes = registry.export([pdf], ['phx'])
    norm = root.RooArgSet(x)
    integral = pdf.createIntegral(norm)
    def function(theta):
        context = Evaluation(nodes, registry, names, theta, fixture['coordinates'])
        return jnp.concatenate((jnp.atleast_1d(context.integral('ph', ('phx',))),
                                context.value('ph'), context.density('ph', ('phx',))))
    function = jax.jit(function)
    gradient = jax.jit(jax.jacrev(function))
    hessian = jax.jit(jax.jacfwd(jax.jacrev(function)))
    def reference(theta):
        for var, value in zip(variables, theta, strict=True):
            var.setVal(float(value))
        raw, density = [], []
        for coordinate in fixture['coordinates']['phx']:
            x.setVal(float(coordinate))
            raw.append(pdf.getVal())
            density.append(pdf.getVal(norm))
        return np.array([integral.getVal(), *raw, *density])
    points = np.random.default_rng(731).uniform([v.getMin() for v in variables],
                                               [v.getMax() for v in variables], (25, len(variables)))
    for index, theta in enumerate(points):
        np.testing.assert_allclose(function(theta), reference(theta), rtol=3e-14, atol=2e-14)
        fd = finite_difference(reference, theta, np.full(len(theta), 2e-4)).T
        np.testing.assert_allclose(gradient(theta), fd, rtol=2e-7, atol=2e-9)
        if index == 0:
            fdh = np.moveaxis(finite_difference(lambda t: np.asarray(gradient(t)), theta,
                                               np.full(len(theta), 2e-4)), 0, -1)
            np.testing.assert_allclose(hessian(theta), fdh, rtol=2e-7, atol=2e-9)
    # Density * stored bin volume integrates to one, independently of ROOT.
    centers = {'phx': (fixture['edges'][1:]+fixture['edges'][:-1])/2}
    context = Evaluation(nodes, registry, names, points[-1], centers)
    assert np.dot(context.density('ph', ('phx',)), np.diff(fixture['edges'])) == pytest.approx(1., abs=3e-15)


@pytest.mark.parametrize('region', [None, 0., .7, 'empty'])
def test_parametric_hist_roundtrip_edges_and_default_range(root, tmp_path, region):
    f = make_hist(root, region)
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(f['pdf'])
    path = tmp_path/'hist.root'
    workspace.writeToFile(str(path))
    handle = root.TFile.Open(str(path))
    w = handle.Get('w')
    pdf, x = w.pdf('ph'), w.var('phx')
    registry = default_registry()
    x.setRange(-1., 6.)
    nodes = registry.export([pdf], ['phx'])
    for coordinate in [-1., 0., np.nextafter(.5, -np.inf), .5, 2., np.nextafter(5., -np.inf), 5., 6.]:
        x.setVal(float(coordinate))
        ctx = Evaluation(nodes, registry, (), (), {'phx': coordinate})
        np.testing.assert_allclose(ctx.value('ph'), pdf.getVal(), rtol=3e-14, atol=2e-14)
    # ROOT's default integral remains the full yield on a restricted default range.
    x.setRange(.1, .2)
    nodes = registry.export([pdf])
    integral = pdf.createIntegral(root.RooArgSet(x))
    np.testing.assert_allclose(Evaluation(nodes, registry, (), ()).integral('ph', ('phx',)),
                               integral.getVal(), rtol=3e-14, atol=2e-14)
    handle.Close()


@pytest.mark.parametrize('region', [None, 'empty'])
def test_parametric_hist_zero_bin_without_morph_and_explicit_unsupported_modes(root, region):
    f = make_hist(root, region)
    f['bins'][0].setRange(0., 14.)
    f['bins'][0].setVal(0.)
    registry = default_registry()
    nodes = registry.export([f['pdf']], ['phx'])
    assert float(Evaluation(nodes, registry, (), (), {'phx': .2}).value('ph')) == f['pdf'].getVal() == 0.
    with pytest.raises(UnsupportedModelError, match='independent bin/morph'):
        Evaluation(nodes, registry, (), ()).integral('ph', ('wrong_axis',))
    f['x'].setRange('subset', .1, 1.)
    f['pdf'].setNormRange('subset')
    with pytest.raises(UnsupportedModelError, match='named-range'):
        registry.export([f['pdf']])
    f['pdf'].setNormRange('')
    f['pdf'].forceNumInt(True)
    with pytest.raises(UnsupportedModelError, match='forced numerical'):
        registry.export([f['pdf']])


def test_parametric_hist_nonempty_disabled_morphs_remain_rejected(root):
    f = make_hist(root, .7)
    pdf = set_morph_flag(root, f['pdf'], False)
    with pytest.raises(UnsupportedModelError, match='morph flag/coefficients'):
        default_registry().export([pdf], ['phx'])
