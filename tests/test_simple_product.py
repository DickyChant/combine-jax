"""SimpleProdPdf preserves its own ordered conditional integral prescription."""
import jax
import numpy as np
import pytest

from combine_jax.graph import parameters
from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference
from test_products import make_product


@pytest.mark.parametrize('mode', ['independent', 'nset', 'overlap', 'vector_settings'])
def test_simple_product_native_values_density_integral_gradient(root, tmp_path, mode):
    f = make_product(root, 'independent' if mode in ('independent','vector_settings') else 'nset')
    if mode in ('independent', 'nset'):
        pdf = root.SimpleProdPdf('simple', '', f['pdf'])
    elif mode == 'vector_settings':
        pdf = root.SimpleProdPdf('simple', '', root.RooArgList(f['gx'], f['gy'], f['ux']), root.std.vector('int')([0,0,0]))
    else:
        # Overlapping independent factors intentionally retain the native
        # product-of-integrals convention; it is not a joint integration.
        pdf = root.SimpleProdPdf('simple', '', root.RooArgList(f['gx'], f['ux']))
    workspace = root.RooWorkspace('w')
    getattr(workspace, 'import')(pdf)
    path = tmp_path/'product.root';workspace.writeToFile(str(path))
    handle = root.TFile.Open(str(path));w = handle.Get('w');pdf = w.pdf('simple')
    axes = [w.var('px')]+([w.var('py')] if w.var('py') else [])
    axis_names = tuple(a.GetName() for a in axes)
    registry = default_registry();nodes = registry.export([pdf], axis_names)
    if mode == 'vector_settings':assert nodes['simple'].attrs['settings'] == (0,1,2)
    info = parameters(pdf, axis_names);names = tuple(p.name for p in info)
    coordinates = {'px': np.array([-1.3,-.1,.7,2.1]), 'py': np.array([-1.9,.4,1.2,2.8])}
    norm = root.RooArgSet(*axes)
    if not hasattr(root, 'simple_product_test_integral'):
        assert root.gInterpreter.Declare('RooAbsReal* simple_product_test_integral(SimpleProdPdf const& p, RooArgSet const& a) '
                                         '{return p.createIntegral(a, nullptr, nullptr, nullptr);}')
    integral = root.simple_product_test_integral(pdf, norm)
    def function(theta):
        ctx = Evaluation(nodes,registry,names,theta,coordinates)
        return (ctx.value('simple'),ctx.integral('simple',axis_names),ctx.density('simple',axis_names))
    function = jax.jit(function);gradient = jax.jit(jax.jacrev(lambda t: function(t)[2]))
    def reference(theta):
        for n,value in zip(names,theta,strict=True):w.var(n).setVal(float(value))
        values, integrals, densities = [], [], []
        for i in range(4):
            for axis in axes:axis.setVal(float(coordinates[axis.GetName()][i]))
            values.append(pdf.getVal());integrals.append(integral.getVal());densities.append(pdf.getVal(norm))
        return np.array(values),np.array(integrals),np.array(densities)
    for point in np.random.default_rng(2718).uniform([p.minimum for p in info],[p.maximum for p in info],(25,len(info))):
        for actual,expected in zip(function(point),reference(point),strict=True):
            np.testing.assert_allclose(actual,expected,rtol=2e-12,atol=2e-12)
        fd=finite_difference(lambda t: reference(t)[2],point,np.full(len(point),2e-4)).T
        np.testing.assert_allclose(gradient(point),fd,rtol=1e-6,atol=1e-8)
    handle.Close()


@pytest.mark.parametrize('mode', ['independent','nset'])
def test_simple_product_tensor_native_nll_and_higher_derivatives(root,mode):
    from combine_jax.root_io import load_caching_nll
    from test_tensor_binned_channels import bind_model,check_model
    root=load_caching_nll();f=make_product(root,mode)
    pdf=root.SimpleProdPdf('simple','',f['pdf'])
    coordinates={'px':np.tile([-.7,.2,2.1],3),'py':np.repeat([-1.9,.3,2.8],3)}
    model=bind_model(root,pdf,[f['x'],f['y']],coordinates)
    check_model(model)
