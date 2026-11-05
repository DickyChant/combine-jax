"""RooRealIntegral caches full conditional-product norms at the first row."""
from array import array
from dataclasses import replace

import jax
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll
from combine_jax.data import DataBinding
from combine_jax.likelihood import Likelihood
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.tensor.binned_channels import pack_binned_channels, make_binned_function
from test_tensor_binned_channels import bind_model, check_model


def fixture(root):
    x = root.RooRealVar('cache_x', '', 1., 0., 4.)
    y = root.RooRealVar('cache_y', '', .2, -2., 1.)
    h = root.TH2D('cache_hist', '', 3, array('d', [0., .5, 2., 4.]),
                  2, array('d', [-2., -.3, 1.]))
    h.SetDirectory(0)
    for i, values in enumerate(((2., 3.), (5., 8.), (4., 1.)), 1):
        for j, value in enumerate(values, 1): h.SetBinContent(i, j, value)
    data = root.RooDataHist('cache_data', '', root.RooArgList(x, y), h)
    hist = root.RooHistPdf('cache_hist_pdf', '', root.RooArgSet(x, y), data)
    mean = root.RooRealVar('cache_mean', '', .8, .5, 1.1)
    morph = root.RooRealVar('cache_morph', '', .2, -.9, .9)
    held = [x, y, h, data, hist, mean, morph]; functions = []
    for i, value in enumerate((.8, 1.1, .6)):
        sigma = root.RooConstVar('cache_sigma'+str(i), '', value)
        gaussian = root.RooGaussian('cache_gaussian'+str(i), '', x, mean, sigma)
        prod = root.RooProdPdf('cache_product'+str(i), '', root.RooArgSet(gaussian),
            root.RooFit.Conditional(root.RooArgSet(hist), root.RooArgSet(y)))
        simple = root.SimpleProdPdf('cache_simple'+str(i), '', prod)
        functions.append(simple); held.extend((sigma, gaussian, prod, simple))
    pdf = root.VerticalInterpPdf('cache_vertical', '', root.RooArgList(*functions),
                                 root.RooArgList(morph), 1., 0)
    return pdf, [x, y], held


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('compact', [False, True])
def test_conditional_norm_row_order_zero_weights_and_ad(root, reverse, compact):
    root = load_caching_nll(); pdf, axes, held = fixture(root)
    order = np.arange(4)[::-1] if reverse else np.arange(4)
    coordinates = {'cache_x': np.array([.25, 1., 3., .25])[order],
                   'cache_y': np.array([-1., .5, .2, -.4])[order]}
    # The first zero-weight row must not select the normalization slice.
    # CachingSimNLL::splitWithWeights removes zero rows before constructing
    # the channel's CachingAddNLL. Reproduce that native input, while retaining
    # the zero row in the JAX source to test its independent compaction.
    weights = np.array([0., 2., 3., 1.])
    model = bind_model(root, pdf, axes, {n:v[1:] for n,v in coordinates.items()}, weights=weights[1:])
    binding = DataBinding(coordinates, weights, 'synthetic_cache_probe')
    old = model['old']
    model['old'] = Likelihood(old.nodes, old.registry, old.parameters,
                             [replace(old.channels[0], data=binding)], [])
    model['rows'][0].update(coordinates=coordinates, weights=weights)
    binned = pack_binned_channels(model['nodes'], model['rows'], model['scalar'], compact_data=compact)
    bf = make_binned_function(binned); sf = model['sf']
    model.update(binned=binned, arrays=jax.device_put(dict(s=dict(model['scalar'].arrays), b=dict(binned.arrays))),
                 function=lambda theta, arrays: bf(sf(theta, arrays['s']), arrays['b'])[0])
    check_model(model)


def test_empty_shape_rows_do_not_require_a_normalization_anchor(root):
    pdf, axes, held = fixture(root)
    registry = default_registry(); names = [a.GetName() for a in axes]
    nodes = registry.export([pdf], names)
    def density():
        context = Evaluation(nodes, registry, (), (), {n:np.empty(0) for n in names})
        return context.density(pdf.GetName(), names)
    assert jax.jit(density)().shape == (0,)
