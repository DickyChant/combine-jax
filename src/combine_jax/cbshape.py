"""Explicit-version RooCBShape extraction and JAX evaluation."""
import jax.numpy as jnp
from jax.scipy.special import erf

from . import cbshape_math
from .graph import UnsupportedModelError
from .root_io import load_root


def check_convention(convention):
    if convention not in ('native','root630','root640'):
        raise ValueError(f'Unknown RooCBShape convention {convention!r}')


def reject_untranslated_cache_routes(nodes, roots):
    """Protect native-default NLL assembly from VectorizedCBShape substitution.

    ADDNLL_RECURSIVE flattens RooAddPdf, whereas SimpleProdPdf/RooProdPdf and
    RooMultiPdf (ADDNLL_MULTINLL is off by default) use the generic PDF cache.
    Scalar RooCBShape conversion remains valid inside those generic PDFs.
    This guard is deliberately explicit until the distinct optimized evaluator
    has its own tested implementation; it must not silently use scalar math.
    """
    visited = set()

    def visit(name):
        if name in visited:
            return
        visited.add(name)
        node = nodes[name]
        cls, inputs, attrs = ((node.class_name, node.inputs, node.attrs)
                             if hasattr(node, 'class_name') else node)
        if cls == 'RooCBShape':
            raise UnsupportedModelError(
                f'{name}: native default VectorizedCBShape cache differs from '
                'scalar RooCBShape (signed tails and +10 sigma cutoff); '
                'this direct cached route is not yet translated')
        if cls == 'RooAddPdf':
            for child in attrs['pdfs']:
                visit(child)
        elif cls == 'RooRealSumPdf':
            for child in attrs['functions']:
                visit(child)
        elif cls == 'RooProduct':
            for child in inputs:
                visit(child)

    for name in roots:
        visit(name)


def extract_cbshape(root, *, convention='native'):
    check_convention(convention)
    if root.getForceNumInt() or root.normRange():
        raise UnsupportedModelError('CBShape forced numerical or named-range normalization')
    if convention=='native':
        version=int(load_root().gROOT.GetVersionInt())//100
        if version not in (630,640):
            raise UnsupportedModelError(f'RooCBShape ROOT {version//100}.{version%100:02d} convention is not certified; choose an explicit tested convention')
        convention='root'+str(version)
    # Semantic accessors read the members actually used by evaluate(). Legacy
    # serialized proxy-list names need not be the current constructor's labels.
    if hasattr(root,'getM'):
        fields=tuple(str(getattr(root,n)().GetName()) for n in ('getM','getM0','getSigma','getAlpha','getN'))
    else:
        proxies={str(root.getProxy(i).GetName()):root.getProxy(i) for i in range(root.numProxies())}
        if set(proxies)!={'m','m0','sigma','alpha','n'}:
            raise UnsupportedModelError('Unrecognized RooCBShape proxy layout without semantic accessors')
        fields=tuple(str(proxies[n].arg().GetName()) for n in ('m','m0','sigma','alpha','n'))
    return fields,dict(convention=convention)


def cbshape_value(node, context):
    return cbshape_math.value(*context.values(node.inputs),jnp,legacy=node.attrs['convention']=='root630')


def cbshape_integral(node, context, axes):
    if axes!=(node.inputs[0],) or any(context.depends_on(n,axes[0]) for n in node.inputs[1:]):
        raise UnsupportedModelError(f'{node.name}: CBShape integral requires independent shape parameters')
    return cbshape_math.integral(*context.bounds(axes[0]),*context.values(node.inputs[1:]),jnp,erf,
                                legacy=node.attrs['convention']=='root630')
