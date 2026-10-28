"""Optional PyTorch execution of neutral complete-likelihood buffers.

The importer and JAX backend are unchanged. This adapter uses the shared scalar,
constraint and template equations, functional indexed writes, and device-side
reductions. Analytic special functions have a separately checked PyTorch port.
"""
import math

from ..graph import UnsupportedModelError
from .constraints import constraint_batch
from .counting import channel_values
from .model import unpack_model
from .scalar import operation
from .templates import morph_batch, template_terms
from .binned_channels import operation as binned_operation, nll_batch, unpack_binned
from .fast_histograms import normalized_batch
from .analytic_shapes import ANALYTIC_OPERATIONS, GRID_ANALYTIC_OPERATIONS
from .grid_expressions import GRID_OPERATIONS
from .torch_analytic import analytic_operation


class _ArrayAPI:
    """Small NumPy-style surface for the existing framework-neutral equations."""

    def __init__(self, torch):
        self.torch = torch
        self.pi, self.inf, self.nan = math.pi, math.inf, math.nan
        self.float64 = torch.float64

    def __getattr__(self, name):
        aliases = {'equal': 'eq', 'not_equal': 'ne', 'less': 'lt', 'less_equal': 'le',
                   'greater': 'gt', 'greater_equal': 'ge', 'power': 'pow'}
        return getattr(self.torch, aliases.get(name, name))

    def sum(self, value, axis=None, keepdims=False):
        return self.torch.sum(value, dim=axis, keepdim=keepdims)

    def prod(self, value, axis=None, keepdims=False):
        # PyTorch 2.11 prod_backward reads the number of zeros with .item(),
        # which synchronizes the host and fails CUDA-graph capture. A balanced
        # multiplication tree is a polynomial at zeros as well, so ordinary AD
        # gives correct first and higher derivatives without data-dependent reads.
        ndim = value.ndim
        if ndim == 0:
            if axis not in (None, 0, -1):
                raise ValueError('Product axis exceeds scalar dimensions')
            return value
        if axis is None:
            work = value.reshape(-1)
        else:
            if not -ndim <= axis < ndim:
                raise ValueError('Product axis exceeds tensor dimensions')
            axis = axis % ndim
            work = self.torch.movedim(value, axis, -1)
        if work.shape[-1] == 0:
            result = work.sum(dim=-1)+1.
        else:
            while work.shape[-1] > 1:
                pairs = work.shape[-1]//2
                reduced = work[..., :2*pairs:2]*work[..., 1:2*pairs:2]
                if work.shape[-1] % 2:
                    reduced = self.torch.cat((reduced, work[..., -1:]), dim=-1)
                work = reduced
            result = work[..., 0]
        if keepdims:
            result = result.reshape((1,)*ndim) if axis is None else result.unsqueeze(axis)
        return result

    def concatenate(self, values, axis=0):
        return self.torch.cat(values, dim=axis)

    def minimum(self, a, b):
        if not self.torch.is_tensor(a):
            a = b.new_full((), a)
        if not self.torch.is_tensor(b):
            b = a.new_full((), b)
        return self.torch.minimum(a, b)

    def maximum(self, a, b):
        if not self.torch.is_tensor(a):
            a = b.new_full((), a)
        if not self.torch.is_tensor(b):
            b = a.new_full((), b)
        return self.torch.maximum(a, b)

    def sqrt(self, value):
        return self.torch.sqrt(value) if self.torch.is_tensor(value) else math.sqrt(value)

    def copysign(self, magnitude, sign):
        if not self.torch.is_tensor(magnitude):
            magnitude = self.torch.full_like(sign, magnitude)
        return self.torch.copysign(magnitude, sign)

    def take_along_axis(self, values, indices, axis):
        return self.torch.take_along_dim(values, indices, dim=axis)

    def clip(self, values, lower, upper):
        # Match NumPy/JAX min/max derivatives at a tie, unlike torch.clamp.
        return self.minimum(self.maximum(values, lower), upper)

    def asarray(self, values, dtype):
        return values.to(dtype=dtype)


def to_torch_arrays(model, *, device='cpu'):
    """Transfer numeric buffers once; integer gathers use PyTorch's int64 indices."""
    import torch
    result = {}
    for name, array in model.arrays.items():
        if name.startswith('parameter_info__'):
            continue
        dtype = torch.int64 if array.dtype.kind == 'i' else torch.bool if array.dtype.kind == 'b' else torch.float64
        result[name] = torch.tensor(array, dtype=dtype, device=device)
    return result


def make_torch_function(model):
    """Return nll(theta, arrays), with no runtime host reads or tensor constants."""
    import torch
    if model.metadata['kind'] not in ('counting_likelihood', 'template_likelihood', 'binned_likelihood'):
        raise UnsupportedModelError('Unsupported PyTorch likelihood kind')
    components = unpack_model(model)
    has_templates = 'templates' in components
    has_binned = 'binned' in components
    schedules = {key: tuple(value.metadata['schedule']) for key, value in components.items()}
    fields = {key: tuple(value.arrays) for key, value in components.items()}
    if has_binned:
        parts = unpack_binned(components['binned'])
        binned_fields = {key: tuple(value.arrays) for key, value in parts.items()}
        histogram_schedule = tuple(parts['histograms'].metadata['schedule'])
        density_schedule = tuple(components['binned'].metadata['program'])
        nll_schedule = tuple(components['binned'].metadata['nll_program'])
    for batch in schedules['constraints']:
        if batch['normalization'] == 'poisson_mean' and batch['observation_dynamic']:
            raise UnsupportedModelError('PyTorch gammainc derivative in a floating shape argument is unsupported')
    xp = _ArrayAPI(torch)

    def put(values, indices, updates):
        # Imported destinations are unique. Never mutate a tensor retained by
        # autograd or by a previous CUDA-graph replay.
        return torch.index_copy(values, 0, indices.reshape(-1),
                                updates.reshape(indices.numel(), *values.shape[1:]))

    def density(op, values, scalars, b):
        if op in ANALYTIC_OPERATIONS:
            return analytic_operation(op, scalars, b, xp)
        if op in GRID_ANALYTIC_OPERATIONS:
            parameters = values[b['children']]
            if 'child_mask' in b:
                parameters = torch.where(b['child_mask'][:,:,None], parameters, 0.)
            return analytic_operation(op, scalars, b, xp, parameters=parameters)
        if op in GRID_OPERATIONS:
            table = values[b['children']]
            active = table[:,0,:] != 0
            arguments = torch.where(active[:,:,None], torch.swapaxes(table[:,1:,:], 1, 2), 1.)
            kind = op.removeprefix('gridexpr_')
            if kind == 'select':
                result = torch.where(arguments[...,0] != 0, arguments[...,1], arguments[...,2])
            else:
                result = operation(kind, arguments, b['operand_mask'][:,None,:], xp, torch.special)
            return torch.where(active, result, 0.).to(torch.float64)
        return binned_operation(op, values, scalars, b, xp)

    def binned(scalars, a):
        parts = {key: {name: a[key+'__'+name] for name in names} for key, names in binned_fields.items()}
        h = parts['histograms']
        bins = h['initial_bins']
        for batch in histogram_schedule:
            b = {key: h[name] for key, name in batch['buffers'].items()}
            bins = put(bins, b['outputs'], normalized_batch(batch['logarithmic'], scalars[b['inputs']], b, xp))
        q = parts['queries']
        queries = put(q['initial_values'], q['gather_outputs'], torch.where(q['gather_valid'], bins[q['gather_indices']], 0.))
        queries = torch.index_add(queries, 0, q['integral_outputs'], bins[q['integral_indices']]*q['integral_weights'])
        values = put(a['initial_values'], a['fast_outputs'], queries[a['fast_inputs']])
        for batch in density_schedule:
            b = {key: a[name] for key, name in batch['buffers'].items()}
            values = put(values, b['outputs'], density(batch['operation'], values, scalars, b))
        result = a['initial_nll']
        for batch in nll_schedule:
            b = {key: a[name] for key, name in batch['buffers'].items()}
            result = put(result, b['outputs'], nll_batch(values, scalars, b, xp))
        return result

    def nll(theta, arrays):
        if theta.dtype != torch.float64 or theta.shape != arrays['scalars__parameter_initial'].shape:
            raise ValueError('PyTorch requires the float64 model parameter vector')
        buffers = {key: {name: arrays[key+'__'+name] for name in names} for key, names in fields.items()}
        a = buffers['scalars']
        values = put(a['initial_values'], a['parameter_slots'], theta[a['parameter_indices']])
        for batch in schedules['scalars']:
            result = operation(batch['operation'], values[a[batch['inputs']]], a[batch['mask']], xp, torch.special)
            values = put(values, a[batch['outputs']], result)
        valid = torch.all(torch.where(a['parameter_fixed'], theta == a['parameter_initial'], True))
        values = torch.where(valid, values[a['output_slots']], math.nan)
        a = buffers['channels']
        channels = put(a['fixed_values'], a['dynamic_positions'], channel_values(values, a, xp))
        result = channels.sum()+arrays['offset']
        profiles = theta.new_empty((0,))
        if has_templates:
            a = buffers['templates']
            histograms = a['histogram_values']
            for batch in schedules['templates'][:-1]:
                b = {key: a[name] for key, name in batch['buffers'].items()}
                updates = morph_batch(batch['mode'], values[a[batch['inputs']]], b, xp)
                histograms = put(histograms, a[batch['outputs']], updates)
            terms, profiles, _ = template_terms(histograms, values, a, xp)
            result = result+terms.sum()
        if has_binned:
            result = result+binned(values, buffers['binned']).sum()
        a = buffers['constraints']
        combined = torch.cat((values, profiles))
        constraints = torch.zeros_like(a['output_template'])
        for batch in schedules['constraints']:
            arguments = None if batch['inputs'] is None else combined[a[batch['inputs']]]
            b = {key: a[name] for key, name in batch['buffers'].items()}
            penalties = constraint_batch(batch['operation'], batch['normalization'],
                batch['observation_dynamic'], arguments, b, xp, torch.special)
            constraints = put(constraints, a[batch['outputs']], penalties)
        return result+constraints.sum()

    return nll
