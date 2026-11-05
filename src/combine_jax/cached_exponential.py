"""Lower Combine's direct exponential cache without changing scalar RooFit PDFs.

With ADDNLL_GAUSSNLL=1, makeCachingPdf uses VectorizedExponential when there is
one leaf parameter. That evaluator uses the leaf itself as lambda, even when
RooExponential::coefficient() is a formula. ADDNLL_RECURSIVE traverses RooAddPdf;
generic PDF products and MultiPdf alternatives keep their ordinary semantics.
"""
from dataclasses import replace
import hashlib
from types import MappingProxyType

from .graph import UnsupportedModelError


def lower_cached_exponentials(nodes, channels):
    """Return immutable aliases for changed cache routes, channels and audit.

    Aliases are essential: one PDF can also occur inside a generic PDF cache,
    where RooFit still evaluates its actual coefficient. Original graph nodes
    are retained unchanged. No native objects enter this graph transformation.
    """
    result = dict(nodes)
    cache = {}
    audit = []

    def alias(node, inputs, attrs, axes):
        key = node.name + '\0' + '\0'.join(axes)
        name = '__combine_jax_cached_' + hashlib.sha256(key.encode()).hexdigest()[:24]
        if name in result:
            raise UnsupportedModelError(f'{node.name}: generated cache alias collides with a graph node')
        result[name] = replace(node, name=name, inputs=tuple(inputs),
            attrs=MappingProxyType({**attrs, 'native_source_name': node.name}))
        return name

    def visit(name, axes):
        key = name, axes
        if key in cache:
            return cache[key]
        node = nodes[name]
        attrs = dict(node.attrs)
        answer = name
        if node.class_name == 'RooExponential':
            parameters = attrs.get('cache_parameters')
            if parameters is None:
                raise UnsupportedModelError(f'{name}: direct exponential cache requires a direct real observable')
            parameters = [p for p in parameters if p[0] not in axes]
            if len(parameters) == 1:
                parameter, cls = parameters[0]
                if cls != 'RooRealVar' or axes != (node.inputs[0],):
                    raise UnsupportedModelError(f'{name}: unvalidated optimized exponential observable/parameter binding')
                if parameter != node.inputs[1] or attrs['sign'] != 1.:
                    attrs.update(sign=1., cache_evaluator='VectorizedExponential',
                        original_coefficient=node.inputs[1], original_sign=node.attrs['sign'])
                    answer = alias(node, (node.inputs[0], parameter), attrs, axes)
                    audit.append(dict(pdf=name, alias=answer, observable=axes[0],
                        coefficient=node.inputs[1], coefficient_sign=node.attrs['sign'],
                        cached_parameter=parameter, evaluator='VectorizedExponential'))
        elif node.class_name in ('RooAddPdf', 'RooRealSumPdf', 'RooProduct'):
            field = {'RooAddPdf':'pdfs', 'RooRealSumPdf':'functions'}.get(node.class_name)
            children = tuple(attrs[field]) if field else node.inputs
            changed = {child:visit(child, axes) for child in children}
            if any(old != new for old, new in changed.items()):
                if field:
                    attrs[field] = tuple(changed[c] for c in children)
                answer = alias(node, tuple(changed.get(c, c) for c in node.inputs), attrs, axes)
        cache[key] = answer
        return answer

    updated = tuple(replace(channel, pdf=visit(channel.pdf, tuple(sorted(channel.observables))))
                    if not channel.masked else channel for channel in channels)
    return MappingProxyType(result), updated, tuple(audit)
