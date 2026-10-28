"""Class translations. Unsupported normalisation configurations raise explicitly."""

import math

import jax.numpy as jnp
from jax.scipy.special import erf

from .graph import UnsupportedModelError
from .registry import Registry, leaf_value


def ordered(items):
    return tuple(str(item.GetName()) for item in items)


def servers(root):
    return ordered(root.servers()), {}


def extract_leaf(root):
    attrs = {"value": float(root.getVal()), "constant": bool(root.isConstant())}
    if root.ClassName() == "RooRealVar":
        attrs.update(minimum=float(root.getMin()), maximum=float(root.getMax()))
    return (), attrs


def log_kappa(theta, low, high):
    x = 2 * theta
    alpha = x * (x*x * (3*x*x - 10) + 15) / 8
    middle = (high - low) / 2 + alpha * (high + low) / 2
    return jnp.where(jnp.abs(theta) >= .5, jnp.where(theta >= 0, high, -low), middle)


def asympow(node, context):
    low, high, theta = context.values(node.inputs)
    return jnp.exp(theta * log_kappa(theta, jnp.log(low), jnp.log(high)))


def extract_normalization(root):
    symmetric = ordered(root.thetaList())
    asymmetric = ordered(root.asymmThetaList())
    other = ordered(root.otherFactorList())
    attrs = {"nominal": float(root.nominalValue()), "symmetric": symmetric,
             "asymmetric": asymmetric, "other": other,
             "log_kappa": tuple(float(v) for v in root.logKappa()),
             "log_asymmetric": tuple((float(v.first), float(v.second)) for v in root.logAsymmKappa())}
    return symmetric + asymmetric + other, attrs


def normalization(node, context):
    a = node.attrs
    exponent = sum(context.value(n) * k for n, k in zip(a["symmetric"], a["log_kappa"], strict=True))
    for name, (low, high) in zip(a["asymmetric"], a["log_asymmetric"], strict=True):
        theta = context.value(name)
        exponent += theta * log_kappa(theta, low, high)
    result = a["nominal"] * jnp.exp(exponent)
    for name in a["other"]:
        result *= context.value(name)
    return result


def extract_product(root):
    if len(root.categoricalComponents()):
        raise UnsupportedModelError("categorical scalar products are not yet supported")
    return ordered(root.realComponents()), {}


def product(node, context):
    result = jnp.asarray(1.0)
    for value in context.values(node.inputs):
        result *= value
    return result


def extract_recursive_fraction(root):
    inputs = ordered(root.variables())
    if not inputs:
        raise UnsupportedModelError("Recursive fraction needs at least one input")
    return inputs, {}


def gaussian(node, context):
    x, mean, sigma = context.values(node.inputs)
    return jnp.exp(-.5 * ((x - mean) / sigma)**2)


def gaussian_log_value(node, context):
    x, mean, sigma = context.values(node.inputs)
    return -.5 * ((x-mean)/sigma)**2


def gaussian_integral(node, context, axes):
    if len(axes) != 1 or axes[0] not in node.inputs[:2]:
        raise UnsupportedModelError(f"{node.name}: Gaussian integral over {axes}")
    variable = axes[0]
    # x and mean are interchangeable for a constant sigma.
    other = node.inputs[1] if variable == node.inputs[0] else node.inputs[0]
    if context.depends_on(node.inputs[2], variable) or context.depends_on(other, variable):
        raise UnsupportedModelError(f"{node.name}: Gaussian parameters depend on integration variable")
    center, sigma = context.value(other), context.value(node.inputs[2])
    low, high = context.bounds(variable)
    scale = jnp.sqrt(2.0) * jnp.abs(sigma)
    return jnp.sqrt(jnp.pi / 2) * jnp.abs(sigma) * (erf((high-center)/scale) - erf((low-center)/scale))


def uniform_integral(node, context, axes):
    if not set(axes) <= set(node.inputs):
        raise UnsupportedModelError(f"{node.name}: Uniform integral over {axes}")
    return math.prod(context.bounds(name)[1] - context.bounds(name)[0] for name in axes)


def extract_add(root):
    pdfs, coefs = ordered(root.pdfList()), ordered(root.coefList())
    if len(coefs) not in (len(pdfs), len(pdfs) - 1):
        raise UnsupportedModelError("automatically extended components are not yet supported")
    if str(root.getCoefRange()) or len(root.getCoefNormalization()):
        raise UnsupportedModelError("coefficient reference ranges/normalisation are not yet supported")
    return pdfs + coefs, {"pdfs": pdfs, "coefs": coefs, "extended": len(coefs) == len(pdfs)}


def add_density(node, context, axes):
    if any(context.depends_on(coef, axis) for coef in node.attrs["coefs"] for axis in axes):
        raise UnsupportedModelError(f"{node.name}: observable-dependent mixture coefficients")
    coefs = context.values(node.attrs["coefs"])
    if node.attrs["extended"]:
        total = sum(coefs)
        coefs = [c / total for c in coefs]
    else:
        coefs.append(1 - sum(coefs))
    return sum(c * context.density(pdf, axes)
               for c, pdf in zip(coefs, node.attrs["pdfs"], strict=True))


def unsupported_raw(node, context):
    raise UnsupportedModelError(f"{node.name} ({node.class_name}) requires a normalisation set")


def extract_real_sum(root):
    functions, coefs = ordered(root.funcList()), ordered(root.coefList())
    if len(functions) != len(coefs):
        raise UnsupportedModelError("RooRealSumPdf implicit last coefficient is not yet supported")
    if root.getFloor() or root.getFloorGlobal() or root.getAttribute("BinnedLikelihoodActive"):
        raise UnsupportedModelError("RooRealSumPdf floor/binned-likelihood mode is not yet supported")
    return functions + coefs, {"functions": functions, "coefs": coefs, "extended": bool(root.canBeExtended())}


def real_sum_value(node, context):
    return sum(context.value(f) * context.value(c)
               for f, c in zip(node.attrs["functions"], node.attrs["coefs"], strict=True))


def real_sum_integral(node, context, axes):
    if any(context.depends_on(c, axis) for c in node.attrs["coefs"] for axis in axes):
        raise UnsupportedModelError(f"{node.name}: observable-dependent real-sum coefficients")
    return sum(context.integral(f, axes) * context.value(c)
               for f, c in zip(node.attrs["functions"], node.attrs["coefs"], strict=True))


def default_registry(*, template_convention='native'):
    from functools import partial
    from .template_conventions import check_convention
    check_convention(template_convention)
    from .discrete import extract_category, extract_multipdf
    from .parametric_templates import (extract_step, step_value, step_integral,
                                       extract_parametric_2d, parametric_2d_value, parametric_2d_integral)
    from .resonance import (extract_doublecb, doublecb_value, doublecb_integral,
                            extract_erfexp, erfexp_value, erfexp_integral)
    from .products import extract_pdf_product, pdf_product_value, pdf_product_integral, pdf_product_density
    from .bernstein_fast import extract_fast_bernstein, fast_bernstein_value, fast_bernstein_integral
    from .splines import extract_spline, spline_value, spline_integral
    from .analytic import (extract_exponential, exponential_value, exponential_integral,
                           extract_bernstein, bernstein_value, bernstein_integral,
                           extract_power, power_value, power_integral)
    from .vertical_nd import (extract_multidimensional, multidimensional_value,
                              multidimensional_integral, multidimensional_density)
    from .binned_pdf import extract_hist_pdf, hist_pdf_value, hist_pdf_integral
    from .vertical import (extract_vertical, vertical_value, vertical_integral,
                           extract_fast_vertical, fast_vertical_value, fast_vertical_integral, fast_vertical_density)
    from .formula import extract_formula, formula_value
    from .bin_statistics import extract_error_propagator, error_value, error_integral
    from .constraints import extract_poisson, poisson_log_value, poisson_integral, fast_poisson_log_value
    from .histograms import extract_histfunc, hist_value, hist_integral
    extract_fast_vertical = partial(extract_fast_vertical, template_convention=template_convention)
    extract_multidimensional = partial(extract_multidimensional, template_convention=template_convention)
    registry = Registry()
    for cls in ("RooRealVar", "RooConstVar"):
        registry.register(cls, extract_leaf, leaf_value)
    registry.register("RooCategory", extract_category, lambda n,c: jnp.asarray(n.attrs["index"]))
    registry.register("RooMultiPdf", extract_multipdf,
                      lambda n,c: c.value(n.attrs["selected"]),
                      density=lambda n,c,a: c.density(n.attrs["selected"],a))
    registry.register("AsymPow", lambda r: (ordered((r.kappaLow(), r.kappaHigh(), r.theta())), {}), asympow)
    registry.register("ProcessNormalization", extract_normalization, normalization)
    registry.register("RooProduct", extract_product, product)
    registry.register("RooRecursiveFraction", extract_recursive_fraction,
                      lambda n,c: c.value(n.inputs[0])*math.prod(1-c.value(i) for i in n.inputs[1:]))
    registry.register("RooFormulaVar", extract_formula, formula_value)
    registry.register("RooSpline1D", extract_spline, spline_value, integral=spline_integral)
    registry.register("RooAddition", lambda r: (ordered(r.list()), {}),
                      lambda n, c: sum(c.values(n.inputs)))
    gaussian_inputs = lambda r: (ordered((r.getX(), r.getMean(), r.getSigma())), {})
    registry.register("RooGaussian", gaussian_inputs, gaussian, integral=gaussian_integral, log_value=gaussian_log_value)
    registry.register("SimpleGaussianConstraint", gaussian_inputs, gaussian, integral=lambda n, c, a: 1., log_value=gaussian_log_value)
    registry.register("RooPoisson", extract_poisson, lambda n, c: jnp.exp(poisson_log_value(n, c)),
                      integral=poisson_integral, log_value=poisson_log_value)
    registry.register("SimplePoissonConstraint", extract_poisson, lambda n, c: jnp.exp(poisson_log_value(n, c)),
                      integral=poisson_integral, log_value=fast_poisson_log_value)
    registry.register("RooUniform", servers, lambda n, c: jnp.asarray(1.), integral=uniform_integral)
    registry.register("RooExponential", extract_exponential, exponential_value, integral=exponential_integral)
    registry.register("RooBernstein", extract_bernstein, bernstein_value, integral=bernstein_integral)
    registry.register("RooPower", extract_power, power_value, integral=power_integral)
    registry.register("RooParametricStepFunction", extract_step, step_value, integral=step_integral)
    registry.register("RooParametricHist2D", extract_parametric_2d, parametric_2d_value, integral=parametric_2d_integral)
    registry.register("RooDoubleCB", extract_doublecb, doublecb_value, integral=doublecb_integral)
    registry.register("RooErfExpPdf", extract_erfexp, erfexp_value, integral=erfexp_integral)
    for degree in range(1, 8):
        registry.register(f"RooBernsteinFast<{degree}>", extract_fast_bernstein, fast_bernstein_value,
                          integral=fast_bernstein_integral)
    registry.register("RooHistPdf", extract_hist_pdf, hist_pdf_value, integral=hist_pdf_integral)
    registry.register("RooAddPdf", extract_add, unsupported_raw, density=add_density)
    registry.register("RooProdPdf", extract_pdf_product, pdf_product_value,
                      integral=pdf_product_integral, density=pdf_product_density)
    registry.register("CMSHistFunc", extract_histfunc, hist_value, integral=hist_integral)
    registry.register("CMSHistErrorPropagator", extract_error_propagator, error_value, integral=error_integral)
    registry.register("RooRealSumPdf", extract_real_sum, real_sum_value, integral=real_sum_integral)
    registry.register("VerticalInterpPdf", extract_vertical, vertical_value, integral=vertical_integral)
    registry.register("FastVerticalInterpHistPdf2", extract_fast_vertical, fast_vertical_value,
                      integral=fast_vertical_integral, density=fast_vertical_density)
    registry.register("FastVerticalInterpHistPdf", extract_fast_vertical, fast_vertical_value,
                      integral=fast_vertical_integral, density=fast_vertical_density)
    for cls in ("FastVerticalInterpHistPdf2D", "FastVerticalInterpHistPdf2D2", "FastVerticalInterpHistPdf3D"):
        registry.register(cls, extract_multidimensional, multidimensional_value,
                          integral=multidimensional_integral, density=multidimensional_density)
    return registry
