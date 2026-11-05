"""ROOT comparisons for optimized scalar nodes used by HIG-23-006."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.optimized_scalars import HGG_CLASSES
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry


def evaluator(node, variables):
    registry = default_registry()
    graph = registry.export([node])
    names = tuple(v.GetName() for v in variables)
    value = jax.jit(lambda theta: Evaluation(graph, registry, names, theta).value(node.GetName()))
    return value, jax.jit(jax.grad(value)), graph


def root_finite_difference(node, variables, point, step=2.e-4):
    def value(theta):
        for variable, coordinate in zip(variables, theta, strict=True):
            variable.setVal(float(coordinate))
        return node.getVal()
    gradient = []
    for i in range(len(variables)):
        direction = np.eye(len(variables))[i] * step
        gradient.append((-value(point + 2*direction) + 8*value(point + direction)
                         - 8*value(point - direction) + value(point - 2*direction)) / (12*step))
    value(point)
    return np.array(gradient)


@pytest.mark.parametrize('cls', HGG_CLASSES)
@pytest.mark.parametrize('empty_terms', [False, True])
def test_hgg_value_gradient_and_serialized_constants(root, tmp_path, cls, empty_terms):
    variables = [root.RooRealVar(f'p{i}', '', .1, -4., 4.) for i in range(4)]
    # Repeated ordered terms matter: the ROOT server list itself deduplicates.
    terms = root.RooArgList() if empty_terms else root.RooArgList(variables[2], variables[1], variables[2])
    coefficients = root.std.vector('double')([] if empty_terms else [.4, -.7, -.11])
    constructors = {
        'A1': lambda: (*variables, terms, coefficients),
        'A2': lambda: (variables[0], -.31, variables[2], variables[3], terms, coefficients),
        'B1': lambda: (variables[0], terms, coefficients),
        'B2': lambda: (2.37, terms, coefficients),
        'C1': lambda: (terms, coefficients),
        'D1': lambda: (variables[0], variables[1]),
        'D2': lambda: (variables[0], .25),
    }
    node = getattr(root, cls)('optimized', '', *constructors[cls[-2:]]())
    value, gradient, graph = evaluator(node, variables)
    points = np.random.default_rng(430).uniform(-2., 2., (25, 4))
    points = np.vstack((points, np.array([.75, .25, 0., 0.])))
    for index, point in enumerate(points):
        for variable, coordinate in zip(variables, point, strict=True):
            variable.setVal(float(coordinate))
        np.testing.assert_allclose(value(point), node.getVal(), rtol=3.e-14, atol=2.e-14)
        # Last point deliberately lands on D's kink: value only, since a
        # mathematical two-sided derivative does not exist there.
        if index < 25:
            np.testing.assert_allclose(gradient(point), root_finite_difference(node, variables, point),
                                       rtol=2.e-7, atol=2.e-10)
    if cls[-2:] in ('D1', 'D2'):
        np.testing.assert_allclose(gradient(points[-1])[0], 1., atol=0.)
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(node)
    path = tmp_path / f'{cls}.root'
    workspace.writeToFile(str(path))
    handle = root.TFile.Open(str(path), 'READ')
    reread = handle.Get('w').function('optimized')
    restored, _, restored_graph = evaluator(reread, variables)
    assert dict(restored_graph['optimized'].attrs) == dict(graph['optimized'].attrs)
    np.testing.assert_allclose(restored(points[0]), value(points[0]), atol=0., rtol=0.)
    handle.Close()


@pytest.mark.parametrize('prune', [False, True])
@pytest.mark.parametrize('constant', [2.5, -1.25, 0.])
def test_cheap_product_folded_factor_and_repeated_terms(root, tmp_path, prune, constant):
    x = root.RooRealVar('x', '', .4, -3., 3.)
    y = root.RooRealVar('y', '', -.2, -3., 3.)
    frozen = root.RooRealVar('frozen', '', -.73)
    frozen.setConstant(True)
    literal = root.RooConstVar('literal', '', constant)
    node = root.RooCheapProduct('product', '', root.RooArgList(x, literal, y, x, frozen), prune)
    value, gradient, graph = evaluator(node, [x, y])
    assert graph['product'].attrs['folded_factor'] == constant * (-.73 if prune else 1.)
    for point in np.random.default_rng(47).uniform(-2., 2., (25, 2)):
        x.setVal(float(point[0])); y.setVal(float(point[1]))
        np.testing.assert_allclose(value(point), node.getVal(), rtol=3.e-14, atol=2.e-14)
        np.testing.assert_allclose(gradient(point), root_finite_difference(node, [x, y], point),
                                   rtol=2.e-7, atol=2.e-10)
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(node)
    path = tmp_path / 'product.root'
    workspace.writeToFile(str(path))
    handle = root.TFile.Open(str(path), 'READ')
    reread = handle.Get('w').function('product')
    restored, _, restored_graph = evaluator(reread, [x, y])
    assert dict(restored_graph['product'].attrs) == dict(graph['product'].attrs)
    np.testing.assert_allclose(restored(jnp.array([.3, -.2])), value(jnp.array([.3, -.2])), atol=0., rtol=0.)
    handle.Close()


def test_cheap_product_without_live_terms(root):
    two = root.RooConstVar('two', '', 2.3)
    minus = root.RooConstVar('minus', '', -.7)
    node = root.RooCheapProduct('only_constants', '', root.RooArgList(two, minus))
    value, _, graph = evaluator(node, [])
    assert not graph['only_constants'].inputs
    np.testing.assert_allclose(value(jnp.array([])), node.getVal(), rtol=0., atol=0.)
