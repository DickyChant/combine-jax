"""Fold and batch scalar dependency DAGs into a framework-neutral numeric program.

This imports rates and observable-independent shape parameters. It is not the
full likelihood importer: observation grids, PDFs and constraints are separate
operators. No original RooFit nodes are interpreted at runtime.
"""
from collections import Counter, defaultdict
import math

import numpy as np

from ..graph import UnsupportedModelError
from .bundle import TensorBundle


SCALAR_CLASSES = frozenset(('RooRealVar', 'RooConstVar', 'RooFormulaVar', 'RooProduct',
    'RooAddition', 'ProcessNormalization', 'AsymPow', 'RooRecursiveFraction', 'RooSpline1D'))
UNARY = frozenset(('neg', 'not', 'abs', 'sqrt', 'exp', 'log', 'log10', 'sin', 'cos',
                   'tan', 'tanh', 'erf', 'erfc'))
BINARY = frozenset(('add', 'sub', 'mul', 'div', 'pow', 'min', 'max', 'lt', 'le',
                    'gt', 'ge', 'eq', 'ne', 'and', 'or'))
OPERATIONS = UNARY | BINARY | {'sum', 'product', 'asymmetric_log'}


def operation(kind, operands, mask, xp=np, special=None):
    """Vector operations shared by the NumPy verifier and the JAX adapter.

    All model numbers, gathers and padding masks are runtime buffers. Only the
    operation kind is static. Padding is neutralized before a reduction.
    """
    a = operands[..., 0]
    if kind == 'sum':
        return xp.sum(xp.where(mask, operands, 0.), axis=-1)
    if kind == 'product':
        return xp.prod(xp.where(mask, operands, 1.), axis=-1)
    if kind == 'asymmetric_log':
        low, high = operands[..., 1], operands[..., 2]
        x = 2*a
        alpha = x*(x*x*(3*x*x-10)+15)/8
        middle = (high-low)/2 + alpha*(high+low)/2
        return a*xp.where(xp.abs(a) >= .5, xp.where(a >= 0, high, -low), middle)
    if kind == 'neg':
        return -a
    if kind == 'not':
        return xp.logical_not(a)
    if kind in ('erf', 'erfc'):
        if special is None:
            from scipy import special
        return getattr(special, kind)(a)
    if kind in UNARY:
        return getattr(xp, kind)(a)
    b = operands[..., 1]
    if kind == 'add': return a+b
    if kind == 'sub': return a-b
    if kind == 'mul': return a*b
    if kind == 'div': return a/b
    if kind == 'pow': return a**b
    functions = dict(min='minimum', max='maximum', lt='less', le='less_equal',
        gt='greater', ge='greater_equal', eq='equal', ne='not_equal',
        **{'and':'logical_and', 'or':'logical_or'})
    if kind not in functions:
        raise UnsupportedModelError(f'Unknown tensor scalar operation {kind}')
    return getattr(xp, functions[kind])(a, b)


def dependency_flags(nodes, parameters, *, observables=(), profiles=()):
    """Classify dependency direction correctly: a node depends on its servers.

    Profiled nuisances are dynamic even though they are outside the AD vector.
    Observable dependencies may only fold after binding an exact evaluation grid.
    Fixed-bound AD entries are immutable configuration for this import.
    """
    floating = {p.name for p in parameters if p.minimum != p.maximum}
    observables, profiles = set(observables), set(profiles)
    result, visiting = {}, set()

    def visit(name):
        if name in visiting:
            raise ValueError(f'Cycle at {name}')
        if name not in result:
            visiting.add(name)
            cls, children, _ = nodes[name]
            own = (name in floating, name in observables, name in profiles)
            for child in children:
                own = tuple(a or b for a,b in zip(own, visit(child), strict=True))
            result[name] = own
            visiting.remove(name)
        return result[name]

    for name in nodes:
        visit(name)
    return result


class _Builder:
    def __init__(self, nodes, parameters, flags, fixed_value):
        self.nodes, self.flags, self.fixed_value = nodes, flags, fixed_value
        self.parameters = {p.name:(i,p) for i,p in enumerate(parameters)}
        self.values, self.depths, self.instructions = [], [], {}
        self.constants, self.expressions, self.outputs, self.parameter_slots = {}, {}, {}, {}
        self.folded = []
        self.constant(0.)  # Safe reduction padding input.

    def allocate(self, value, depth):
        slot = len(self.values)
        self.values.append(float(value))
        self.depths.append(depth)
        return slot

    def constant(self, value):
        value = float(value)
        if not math.isfinite(value):
            raise UnsupportedModelError('Nonfinite folded scalar value')
        key = np.float64(value).tobytes()
        if key not in self.constants:
            self.constants[key] = self.allocate(value, 0)
        return self.constants[key]

    def is_constant(self, slot):
        return self.depths[slot] == 0 and slot not in self.parameter_slots

    def emit(self, kind, inputs):
        inputs = tuple(inputs)
        if kind not in OPERATIONS:
            raise UnsupportedModelError(f'Unsupported scalar opcode {kind}')
        if not inputs:
            if kind not in ('sum', 'product'):
                raise ValueError(f'{kind} needs operands')
            return self.constant(0. if kind == 'sum' else 1.)
        if all(self.is_constant(i) for i in inputs):
            values = np.asarray([[self.values[i] for i in inputs]])
            with np.errstate(all='ignore'):
                value = operation(kind, values, np.ones(values.shape, dtype=bool))[0]
            return self.constant(value)
        key = kind, inputs
        if key not in self.expressions:
            depth = 1+max(self.depths[i] for i in inputs)
            slot = self.allocate(0., depth)
            self.instructions[slot] = (kind, inputs)
            self.expressions[key] = slot
        return self.expressions[key]

    def formula(self, tree, arguments):
        kind = tree[0]
        if kind == 'constant': return self.constant(tree[1])
        if kind == 'argument': return arguments[tree[1]]
        binary = {'+':'add', '-':'sub', '*':'mul', '/':'div', '**':'pow',
                  '<':'lt', '<=':'le', '>':'gt', '>=':'ge', '==':'eq', '!=':'ne'}
        if kind in ('binary', 'compare'):
            return self.emit(binary[tree[1]], [self.formula(t, arguments) for t in tree[2:]])
        if kind == 'unary':
            a = self.formula(tree[2], arguments)
            return a if tree[1] == '+' else self.emit({'-':'neg', 'not':'not'}[tree[1]], [a])
        if kind == 'call':
            op = {'fabs':'abs', 'pow':'pow'}.get(tree[1], tree[1])
            return self.emit(op, [self.formula(t, arguments) for t in tree[2]])
        if kind == 'boolean':
            result = self.formula(tree[2][0], arguments)
            for term in tree[2][1:]:
                result = self.emit(tree[1], [result, self.formula(term, arguments)])
            return result
        if kind == 'conditional':
            condition = self.formula(tree[1], arguments)
            if self.is_constant(condition):
                return self.formula(tree[2] if self.values[condition] else tree[3], arguments)
            raise UnsupportedModelError('Dynamic conditional scalar formula needs masked branch lowering; eager select would corrupt inactive-branch derivatives')
        raise UnsupportedModelError(f'Unsupported scalar formula AST {kind}')

    def node(self, name):
        if name in self.outputs:
            return self.outputs[name]
        cls, children, attrs = self.nodes[name]
        if cls not in SCALAR_CLASSES:
            raise UnsupportedModelError(f'{name} ({cls}): not a scalar tensor import class')
        floating, observable, profiled = self.flags[name]
        if observable or profiled:
            raise UnsupportedModelError(f'{name}: scalar program cannot bind observable/profile dependencies')
        if not floating:
            value = self.fixed_value(name)
            result = self.constant(value)
            self.folded.append(name)
        elif name in self.parameters:
            index, parameter = self.parameters[name]
            if parameter.minimum == parameter.maximum:
                raise AssertionError('Fixed parameter escaped dependency folding')
            result = self.allocate(parameter.value, 0)
            self.parameter_slots[result] = index
        else:
            values = [self.node(c) for c in children]
            if cls == 'RooFormulaVar':
                result = self.formula(attrs['ast'], values)
            elif cls in ('RooProduct', 'RooAddition'):
                result = self.emit('product' if cls == 'RooProduct' else 'sum', values)
            elif cls == 'RooRecursiveFraction':
                one = self.constant(1.)
                result = self.emit('product', [values[0], *[self.emit('sub', [one,v]) for v in values[1:]]])
            elif cls == 'AsymPow':
                low, high, theta = values
                shift = self.emit('asymmetric_log', [theta, self.emit('log', [low]), self.emit('log', [high])])
                result = self.emit('exp', [shift])
            elif cls == 'ProcessNormalization':
                exponent = [self.emit('mul', [self.node(n), self.constant(k)])
                            for n,k in zip(attrs['symmetric'], attrs['log_kappa'], strict=True)]
                exponent += [self.emit('asymmetric_log', [self.node(n), self.constant(lo), self.constant(hi)])
                             for n,(lo,hi) in zip(attrs['asymmetric'], attrs['log_asymmetric'], strict=True)]
                exp = self.emit('exp', [self.emit('sum', exponent)])
                result = self.emit('product', [self.constant(attrs['nominal']), exp,
                                               *[self.node(n) for n in attrs['other']]])
            else:
                raise UnsupportedModelError(f'{name} ({cls}): dynamic scalar tensor lowering not implemented')
        self.outputs[name] = result
        return result


def import_scalars(nodes, parameters, roots, *, fixed_value, observables=(), profiles=(), provenance=None):
    """Compile explicit scalar roots; fixed_value is an import-time native reader.

    It is called only for certified scalar classes with no floating, observable
    or profiled dependency. Numeric constants are stored in buffers, not closures.
    Requested roots are preserved in output order, including aliases.
    """
    roots = tuple(roots)
    flags = dependency_flags(nodes, parameters, observables=observables, profiles=profiles)
    certified = set()
    def certify(name):
        if name not in certified:
            cls, children, _ = nodes[name]
            if cls not in SCALAR_CLASSES:
                raise UnsupportedModelError(f'{name} ({cls}): unsupported dependency in scalar import')
            for child in children:
                certify(child)
            certified.add(name)
    for name in roots:
        certify(name)
    builder = _Builder(nodes, parameters, flags, fixed_value)
    root_slots = []
    for name in roots:
        try:
            root_slots.append(builder.node(name))
        except UnsupportedModelError as exc:
            raise UnsupportedModelError(f'Scalar root {name}: {exc}') from exc
    batches = defaultdict(list)
    for slot, (kind, inputs) in builder.instructions.items():
        batches[(builder.depths[slot], kind)].append((slot, inputs))
    arrays = dict(initial_values=np.array(builder.values, dtype=np.float64),
        parameter_indices=np.array(list(builder.parameter_slots.values()), dtype=np.int32),
        parameter_slots=np.array(list(builder.parameter_slots), dtype=np.int32),
        parameter_initial=np.array([p.value for p in parameters], dtype=np.float64),
        parameter_fixed=np.array([p.minimum == p.maximum for p in parameters], dtype=bool),
        output_slots=np.array(root_slots, dtype=np.int32))
    schedule = []
    for i, ((depth, kind), entries) in enumerate(sorted(batches.items())):
        width = max(len(inputs) for _,inputs in entries)
        inputs = np.zeros((len(entries), width), dtype=np.int32)
        mask = np.zeros(inputs.shape, dtype=bool)
        for row, (_, children) in enumerate(entries):
            inputs[row, :len(children)] = children
            mask[row, :len(children)] = True
        iname, mname, oname = f'inputs_{i}', f'mask_{i}', f'outputs_{i}'
        arrays.update({iname:inputs, mname:mask, oname:np.array([s for s,_ in entries], dtype=np.int32)})
        schedule.append(dict(operation=kind, depth=depth, inputs=iname, mask=mname, outputs=oname))
    metadata = dict(schema='combine-jax.tensor.v1', kind='scalar_program', schedule=schedule,
        parameter_names=[p.name for p in parameters], output_names=list(roots),
        folded_nodes=builder.folded,
        folded_classes=dict(sorted(Counter(nodes[n][0] for n in builder.folded).items())),
        dynamic_instructions=len(builder.instructions), batches=len(schedule),
        maximum_depth=max(builder.depths), provenance=provenance or {},
        scope='Scalar rates/shape parameters only; does not evaluate channel PDFs or constraints. Fixed-bound entries must retain imported values. Dynamic conditionals and floating splines require subsequent lowering.')
    return TensorBundle(metadata, arrays)


def validate_program(bundle):
    """Check topology and schemas before any backend tracing or scatter/gather."""
    a, m = bundle.arrays, bundle.metadata
    if m['kind'] != 'scalar_program':
        raise ValueError('Expected scalar program')
    n, p = len(a['initial_values']), len(m['parameter_names'])
    for name in ('initial_values', 'parameter_initial'):
        if a[name].dtype != np.float64 or a[name].ndim != 1 or not np.all(np.isfinite(a[name])):
            raise ValueError(f'Invalid float64 buffer {name}')
    if a['parameter_initial'].shape != (p,) or a['parameter_fixed'].shape != (p,) or a['parameter_fixed'].dtype != bool:
        raise ValueError('Parameter metadata mismatch')
    indices, slots = a['parameter_indices'], a['parameter_slots']
    if indices.dtype != np.int32 or indices.shape != slots.shape or indices.ndim != 1:
        raise ValueError('Invalid parameter gather shape')
    if np.any(indices < 0) or np.any(indices >= p) or len(set(indices.tolist())) != len(indices):
        raise ValueError('Invalid parameter indices')
    if np.any(a['parameter_fixed'][indices]):
        raise ValueError('Fixed-bound parameter remained dynamic')
    def index_buffer(name):
        values = a[name]
        if values.dtype != np.int32 or np.any(values < 0) or np.any(values >= n):
            raise ValueError(f'Invalid slot indices {name}')
        return values
    index_buffer('parameter_slots')
    if len(set(slots.tolist())) != len(slots):
        raise ValueError('Duplicate parameter slots')
    produced = set(slots.tolist())
    all_outputs = set()
    for batch in m['schedule']:
        op = batch['operation']
        if op not in OPERATIONS:
            raise ValueError(f'Unknown opcode {op}')
        inputs, outputs, mask = index_buffer(batch['inputs']), index_buffer(batch['outputs']), a[batch['mask']]
        if inputs.ndim != 2 or mask.shape != inputs.shape or mask.dtype != bool or outputs.shape != (len(inputs),):
            raise ValueError('Invalid operation batch shape')
        arity = 1 if op in UNARY else 2 if op in BINARY else 3 if op == 'asymmetric_log' else None
        if (arity is not None and (inputs.shape[1] != arity or not np.all(mask))) or inputs.shape[1] < 1:
            raise ValueError(f'Invalid arity for {op}')
        if len(set(outputs.tolist())) != len(outputs) or set(outputs.tolist()) & (produced | all_outputs):
            raise ValueError('Duplicate output slots')
        all_outputs.update(outputs.tolist())
    # Every instruction may read constants, parameters or earlier instructions.
    available = set(range(n))-all_outputs
    for batch in m['schedule']:
        if not set(a[batch['inputs']].ravel().tolist()) <= available:
            raise ValueError('Instruction reads an unavailable/future slot')
        available.update(a[batch['outputs']].tolist())
    if index_buffer('output_slots').shape != (len(m['output_names']),):
        raise ValueError('Output names/slots mismatch')


def evaluate_numpy(bundle, theta):
    validate_program(bundle)
    a = bundle.arrays
    theta = np.asarray(theta, dtype=np.float64)
    if theta.shape != a['parameter_initial'].shape:
        raise ValueError('Parameter vector shape mismatch')
    if np.any(theta[a['parameter_fixed']] != a['parameter_initial'][a['parameter_fixed']]):
        raise ValueError('Fixed-bound parameter changed: rebuild the import')
    values = a['initial_values'].copy()
    values[a['parameter_slots']] = theta[a['parameter_indices']]
    for batch in bundle.metadata['schedule']:
        values[a[batch['outputs']]] = operation(batch['operation'], values[a[batch['inputs']]], a[batch['mask']])
    return values[a['output_slots']]
