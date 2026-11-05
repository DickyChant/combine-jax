"""Native joint normalization for ModZ/TwoExp times Chebychev shapes."""
from dataclasses import asdict

from .graph import UnsupportedModelError


def extract(root, specifications):
    children = list(root.pdfList())
    if len(children) != 2 or any(kind != 'all' for kind, _ in specifications): return None
    bases = [c for c in children if c.ClassName() in ('RooModZPdf', 'RooSumTwoExpPdf')]
    polynomials = [c for c in children if c.ClassName() == 'RooChebychev']
    if len(bases) != 1 or len(polynomials) != 1: return None
    base, polynomial = bases[0], polynomials[0]
    from .hmumu import extract_modz, extract_twoexp
    from .chebyshev import extract_chebyshev, arguments
    from .romberg import extract_config, RombergConfig
    _, cheby = extract_chebyshev(polynomial)
    axis, coefficients = arguments(polynomial)
    if base.ClassName() == 'RooModZPdf':
        _, attrs = extract_modz(base)
        names = (*attrs['fields'][1:], *attrs['coefficients'])
        defaults = (0., 0., 2., 91.2, 2.5, *([0.]*len(attrs['coefficients'])))
        shape = 'modz'
    else:
        _, attrs = extract_twoexp(base)
        names = (*attrs['fields'][1:4], attrs['fields'][4] if attrs['offset'] else None)
        defaults = (0., 0., 0., 91.2 if attrs['offset'] else 0.)
        shape = 'twoexp'
    if attrs['fields'][0] != cheby['observable_name']: return None
    servers = {str(c.GetName()):c for c in base.servers()}
    if any(c.dependsOn(axis) for c in coefficients) or any(servers[n].dependsOn(axis) for n in names if n):
        raise UnsupportedModelError('Joint numerical shape coefficients depend on its integration axis')
    config = extract_config(root)
    if config != asdict(RombergConfig()) or config != extract_config(base) or config != extract_config(polynomial):
        raise UnsupportedModelError('Joint product custom numerical configuration is not yet certified')
    return dict(axis=str(axis.GetName()), bounds=(float(axis.getMin()), float(axis.getMax())),
        coefs=(*names, *cheby['coefs']), defaults=(*defaults, *([0.]*len(cheby['coefs']))),
        shape=shape, shape_parameter_count=len(names),
        references=(attrs['reference_range'], cheby['reference_range']), integration_config=config)


def density(node, context, axes):
    import jax.numpy as jnp
    from .romberg import RombergConfig
    from .romberg_jax import integrate
    from .joint_product_math import value
    attrs = node.attrs['joint_numeric_product']
    if axes != (attrs['axis'],) or context.bounds(attrs['axis']) != tuple(attrs['bounds']):
        raise UnsupportedModelError(f'{node.name}: joint numerical product axis/range changed')
    parameters = jnp.stack([context.value(n) if n else jnp.asarray(v)
        for n, v in zip(attrs['coefs'], attrs['defaults'], strict=True)])
    if parameters.ndim != 1: raise UnsupportedModelError('Conditional joint numerical shape parameters')
    function = lambda p, x: value(x, p, attrs['shape'], attrs['shape_parameter_count'], attrs['references'], jnp)
    denominator = integrate(function, parameters, *attrs['bounds'], RombergConfig(**attrs['integration_config']))
    return context.value(node.name)/denominator
