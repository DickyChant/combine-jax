import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.registry import Evaluation
from combine_jax.translators import default_registry
from combine_jax.validation import finite_difference


@pytest.mark.parametrize("rounding", [False, True])
@pytest.mark.parametrize("observed", [0., 2., 2.5, 17.])
def test_poisson_value_and_both_analytic_integrals(root, rounding, observed):
    x = root.RooRealVar("observed", "", observed, 0., 40.)
    mean = root.RooRealVar("mean", "", 3., .2, 28.)
    poisson = root.RooPoisson("poisson", "", x, mean, not rounding)
    registry = default_registry()
    graph = registry.export([poisson])
    mu_integral = poisson.createIntegral(root.RooArgSet(mean))
    x_integral = poisson.createIntegral(root.RooArgSet(x))
    for mu in [.2, .7, 3., 20.]:
        mean.setVal(mu)
        context = Evaluation(graph, registry, ("mean",), jnp.array([mu]))
        np.testing.assert_allclose(context.value("poisson"), poisson.getVal(), atol=1e-14, rtol=2e-13)
        np.testing.assert_allclose(context.integral("poisson", ("mean",)), mu_integral.getVal(), atol=2e-14, rtol=2e-13)
        np.testing.assert_allclose(context.integral("poisson", ("observed",)), x_integral.getVal(), atol=2e-14, rtol=2e-13)


@pytest.mark.parametrize("observed", [0., 2., 2.5, 999999., 1e6, 2e6])
def test_fast_poisson_constraint_matches_combine_branch(root, observed):
    x = root.RooConstVar("observed", "", observed)
    mean = root.RooRealVar("mean", "", max(observed, 1.), 0., max(2*observed, 20.))
    pdf = root.SimplePoissonConstraint("simple", "", x, mean, True)
    registry = default_registry()
    nodes = registry.export([pdf])
    value = jax.jit(lambda t: Evaluation(nodes, registry, ("mean",), t).log_value("simple"))
    def reference(theta):
        mean.setVal(float(theta[0]))
        return float(pdf.getLogValFast())
    for mu in [max(1., observed*.9), max(2., observed), max(3., observed*1.1)]:
        point = np.array([mu])
        np.testing.assert_allclose(value(point), reference(point), atol=1e-8, rtol=1e-12)
        fd = finite_difference(reference, point, np.array([max(1e-4, mu*1e-5)]))
        np.testing.assert_allclose(jax.grad(value)(point), fd, atol=2e-8, rtol=1e-5)


def test_empty_poisson_at_zero_mean_has_finite_gradient(root):
    x = root.RooConstVar("observed", "", 0.)
    mean = root.RooRealVar("mean", "", 0., 0., 10.)
    pdf = root.RooPoisson("empty", "", x, mean, True)
    registry = default_registry()
    nodes = registry.export([pdf])
    f = lambda t: Evaluation(nodes, registry, ("mean",), t).log_value("empty")
    np.testing.assert_allclose(f(jnp.array([0.])), 0.)
    np.testing.assert_allclose(jax.grad(f)(jnp.array([0.])), [-1.])
