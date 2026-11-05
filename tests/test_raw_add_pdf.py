"""Raw mixtures inside SimpleProdPdf use raw component normalization."""
import jax
import numpy as np
import pytest

from combine_jax.graph import parameters, UnsupportedModelError
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


def fixture(root, kind):
    x = root.RooRealVar('raw_x', '', .4, -3., 4.)
    mean = root.RooRealVar('raw_mean', '', .1, -.8, .8)
    sigma = root.RooRealVar('raw_sigma', '', .7, .4, 1.1)
    wide = root.RooConstVar('raw_wide', '', 1.5)
    broad = root.RooConstVar('raw_broad', '', 2.2)
    g0 = root.RooGaussian('raw_g0', '', x, mean, sigma)
    g1 = root.RooGaussian('raw_g1', '', x, mean, wide)
    g2 = root.RooGaussian('raw_g2', '', x, mean, broad)
    a = root.RooRealVar('raw_a', '', .3, .1, .6)
    b = root.RooRealVar('raw_b', '', .2, .1, .3)
    pdfs = root.RooArgList(g0, g1, g2)
    if kind == 'yields':
        c = root.RooRealVar('raw_c', '', .7, .4, 1.2)
        mix = root.RooAddPdf('raw_mix', '', pdfs, root.RooArgList(a, b, c))
    else:
        mix = root.RooAddPdf('raw_mix', '', pdfs, root.RooArgList(a, b), kind == 'recursive')
    parent = root.SimpleProdPdf('raw_parent', '', root.RooArgList(mix))
    return locals()


@pytest.mark.parametrize('kind', ['fractions', 'yields', 'recursive'])
def test_raw_mixture_parent_value_integral_density_gradient(root, kind, tmp_path):
    f = fixture(root, kind)
    w0 = root.RooWorkspace('w'); getattr(w0, 'import')(f['parent'])
    path = tmp_path / 'mixture.root'; w0.writeToFile(str(path))
    handle = root.TFile.Open(str(path)); w = handle.Get('w')
    parent, x = w.pdf('raw_parent'), w.var('raw_x')
    norm = root.RooArgSet(x)
    if not hasattr(root, 'raw_add_test_integral'):
        assert root.gInterpreter.Declare('RooAbsReal* raw_add_test_integral(SimpleProdPdf const& p, RooArgSet const& a) '
                                        '{return p.createIntegral(a,nullptr,nullptr,nullptr);}')
    integral = root.raw_add_test_integral(parent, norm)
    registry = default_registry(); nodes = registry.export([parent], ('raw_x',))
    info = parameters(parent, ('raw_x',)); names = tuple(p.name for p in info)
    coordinates = np.array([-2., -.3, .4, 2.7])
    def evaluate(theta):
        context = Evaluation(nodes, registry, names, theta, {'raw_x': coordinates})
        return (context.value('raw_parent'), context.integral('raw_parent', ('raw_x',)),
                context.density('raw_parent', ('raw_x',)))
    compiled = jax.jit(evaluate)
    gradient = jax.jit(jax.jacrev(lambda t: evaluate(t)[2]))
    def native(theta):
        for name, value in zip(names, theta, strict=True): w.var(name).setVal(float(value))
        values, densities = [], []
        for value in coordinates:
            x.setVal(float(value)); values.append(parent.getVal()); densities.append(parent.getVal(norm))
        return np.asarray(values), integral.getVal(), np.asarray(densities)
    rng = np.random.default_rng(913)
    points = [[p.value for p in info]] + list(rng.uniform([p.minimum for p in info], [p.maximum for p in info], (25, len(info))))
    for point in points:
        point = np.asarray(point)
        for a, b in zip(compiled(point), native(point), strict=True):
            np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)
        fd = finite_difference(lambda t: native(t)[2], np.asarray(point), np.full(len(info), 2e-4)).T
        np.testing.assert_allclose(gradient(point), fd, rtol=1e-6, atol=2e-9)
    handle.Close()


@pytest.mark.parametrize('kind', ['fractions', 'yields', 'recursive'])
def test_raw_mixture_tensor_native_nll_gradient_hessian(root, kind):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model, check_model
    root = load_caching_nll(); f = fixture(root, kind)
    model = bind_model(root, f['parent'], [f['x']], {'raw_x': np.array([-2., -.3, .4, 2.7])})
    check_model(model)


def test_raw_integral_rejects_observable_dependent_coefficient(root):
    f = fixture(root, 'fractions')
    coefficient = root.RooFormulaVar('bad_fraction', '.3+.01*@0', root.RooArgList(f['x']))
    pdf = root.RooAddPdf('bad_mix', '', root.RooArgList(f['g0'], f['g1']), root.RooArgList(coefficient))
    registry = default_registry(); nodes = registry.export([pdf], ('raw_x',))
    context = Evaluation(nodes, registry, (), [], {'raw_x': np.array([.2])})
    with pytest.raises(UnsupportedModelError, match='observable-dependent'):
        context.integral('bad_mix', ('raw_x',))


def test_mixed_raw_and_normalized_shared_mixture_is_explicitly_rejected(root):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model
    root = load_caching_nll(); f = fixture(root, 'fractions')
    registry = default_registry(); nodes = registry.export([f['parent']], ('raw_x',))
    context = Evaluation(nodes, registry, (), [], {'raw_x': np.array([.2])})
    context.value('raw_parent')
    with pytest.raises(UnsupportedModelError, match='mixed raw/normalized'):
        context.density('raw_mix', ('raw_x',))
    fraction = root.RooConstVar('outer_fraction', '', .4)
    outer = root.RooAddPdf('outer', '', root.RooArgList(f['parent'], f['mix']), root.RooArgList(fraction))
    with pytest.raises(UnsupportedModelError, match='mixed raw/normalized'):
        bind_model(root, outer, [f['x']], {'raw_x': np.array([-.2, .3, 1.])})


def test_raw_mixture_integral_preserves_native_supplemental_normalization(root):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model, check_model
    root = load_caching_nll(); f = fixture(root, 'fractions')
    y = root.RooRealVar('raw_y', '', .2, -1., 2.)
    gy = root.RooGaussian('raw_gy', '', y, f['mean'], f['sigma'])
    mixture = root.RooAddPdf('raw_xy_mix', '', root.RooArgList(f['g0'], gy), root.RooArgList(f['a']))
    parent = root.SimpleProdPdf('raw_xy_parent', '', root.RooArgList(mixture))
    model = bind_model(root, parent, [f['x'], y],
                       {'raw_x': np.array([-2., -.3, .4, 2.7]), 'raw_y': np.array([-.6, .2, .7, 1.6])})
    check_model(model)
