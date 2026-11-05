"""Optimized scalars must reach the batched tensor path, including coordinate grids."""
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np

from combine_jax.graph import Parameter
from combine_jax.optimized_scalars import HGG_CLASSES
from combine_jax.registry import Evaluation
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import import_scalars, evaluate_numpy
from combine_jax.translators import default_registry


def test_all_optimized_scalar_classes_batch_values_derivatives_and_transport(root, tmp_path):
    variables = [root.RooRealVar(f'bp{i}', '', .2, -4., 4.) for i in range(4)]
    terms = root.RooArgList(variables[2], variables[1], variables[2])
    coefficients = root.std.vector('double')([.4, -.7, -.11])
    args = [(*variables, terms, coefficients),
            (variables[0], -.31, variables[2], variables[3], terms, coefficients),
            (variables[0], terms, coefficients), (2.37, terms, coefficients),
            (terms, coefficients), (variables[0], variables[1]), (variables[0], .25)]
    roots = [getattr(root, cls)(cls, '', *a) for cls, a in zip(HGG_CLASSES, args, strict=True)]
    literal = root.RooConstVar('literal', '', -2.7)
    roots.append(root.RooCheapProduct('cheap', '', root.RooArgList(literal, *variables)))
    registry = default_registry()
    exported = registry.export(roots)
    nodes = {n: (v.class_name, v.inputs, dict(v.attrs)) for n, v in exported.items()}
    parameters = [Parameter(v.GetName(), .2, -4., 4., 1., 'nuisance') for v in variables]
    names, outputs = [p.name for p in parameters], [r.GetName() for r in roots]
    fixed = lambda name: Evaluation(exported, registry, names, jnp.array([.2]*4)).value(name)
    bundle = import_scalars(nodes, parameters, outputs, fixed_value=fixed)
    arrays = jax.device_put(dict(bundle.arrays))
    batched = jax.jit(lambda theta: make_function(bundle)(theta, arrays))
    oracle = lambda theta: jnp.stack([Evaluation(exported, registry, names, theta).value(n) for n in outputs])
    grad, hess = jax.jit(jax.jacrev(batched)), jax.jit(jax.jacfwd(jax.jacrev(batched)))
    refgrad, refhess = jax.jit(jax.jacrev(oracle)), jax.jit(jax.jacfwd(jax.jacrev(oracle)))
    points = np.random.default_rng(990).uniform(-2., 2., (25, 4))
    points = np.vstack((points, [.75, .25, 0., 0.]))
    for point in points:
        for variable, coordinate in zip(variables, point, strict=True):
            variable.setVal(float(coordinate))
        native = [r.getVal() for r in roots]
        np.testing.assert_allclose(batched(point), native, rtol=3.e-14, atol=2.e-14)
        np.testing.assert_allclose(evaluate_numpy(bundle, point), native, rtol=3.e-14, atol=2.e-14)
        np.testing.assert_allclose(grad(point), refgrad(point), atol=2.e-13, rtol=3.e-13)
        np.testing.assert_allclose(hess(point), refhess(point), atol=2.e-13, rtol=3.e-13)
    fixed_bundle = import_scalars(nodes, [], outputs, fixed_value=lambda n: fixed(n))
    assert not fixed_bundle.metadata['schedule']
    assert set(fixed_bundle.metadata['folded_nodes']) == set(outputs)
    directory = tmp_path / 'bundle'
    bundle.save(directory)
    script = '''import sys
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
b=TensorBundle.load(sys.argv[1]);evaluate_numpy(b,b.arrays['parameter_initial'])
assert not any(x in sys.modules for x in ('jax','torch','ROOT'))
'''
    subprocess.run([sys.executable, '-c', script, str(directory)], check=True, capture_output=True)


def test_optimized_scalars_inside_conditional_grid(root, tmp_path):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model, check_model
    root = load_caching_nll()
    x = root.RooRealVar('ox', '', 1., -4., 4.)
    y = root.RooRealVar('oy', '', .3, -1., 1.)
    theta = root.RooRealVar('ot', '', .1, -.3, .3)
    literal = root.RooConstVar('offset', '', 1.7)
    product = root.RooCheapProduct('conditional_product', '', root.RooArgList(y, literal))
    mean = root.CMSHggFormulaC1('conditional_mean', '', root.RooArgList(theta, product),
                              root.std.vector('double')([.2, .3]))
    width = root.CMSHggFormulaB2('conditional_width', '', 1.2, root.RooArgList(y, theta),
                               root.std.vector('double')([.1, .2]))
    gaussian = root.RooGaussian('conditional_gaussian', '', x, mean, width)
    marginal = root.RooUniform('y_marginal', '', root.RooArgSet(y))
    pdf = root.RooProdPdf('optimized_conditional', '', root.RooArgSet(marginal),
                         root.RooFit.Conditional(root.RooArgSet(gaussian), root.RooArgSet(x)))
    coordinates = {'ox': np.tile([-.7, .3, 2.1], 3), 'oy': np.repeat([-.8, 0., .7], 3)}
    # Match the workspace-driven production boundary. The native CMSHgg copy
    # constructors copy transient vterms_ pointers: fresh observable-dependent
    # nodes can retain stale pointers when CachingPdf redirects its clone.
    # Deserialization clears these nonserialized caches. See the recorded
    # fresh-vs-deserialized native diagnostic; do not alter the JAX formula.
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(pdf)
    path = tmp_path / 'conditional.root'
    workspace.writeToFile(str(path))
    handle = root.TFile.Open(str(path), 'READ')
    loaded = handle.Get('w')
    model = bind_model(root, loaded.pdf(pdf.GetName()), [loaded.var('ox'), loaded.var('oy')], coordinates)
    check_model(model)
    handle.Close()
