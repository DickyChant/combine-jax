"""Explicit Combine v11 fit configuration at the ROOT import boundary.

The serialized result is plain metadata and NumPy arrays. No ROOT or numerical
backend is imported here; workspace objects are used only by boundary helpers.
"""
from contextlib import contextmanager

import numpy as np

from .bundle import TensorBundle


POLICY = 'combine_v11_default'


def workspace_fit_state(workspace):
    """Capture CLI-relevant state before extraction can move observables."""
    config = workspace.obj('ModelConfig')
    if not config:
        raise ValueError('Missing ModelConfig')
    nuisances = config.GetNuisanceParameters()
    observables = config.GetObservables()
    optimized = sorted(v.GetName() for v in nuisances or ()
        if v.InheritsFrom('RooRealVar') and v.getAttribute('optimizeBounds'))
    return dict(policy=POLICY, runtime_flags=dict(OPTIMIZE_BOUNDS=1, UNBOUND_GAUSSIANS=0),
        nuisance_bounds_removed=optimized,
        fixed_real_values={v.GetName():float(v.getVal()) for v in workspace.allVars()
            if v.isConstant() or (v.getMin()==v.getMax() and v.GetName() not in optimized)},
        category_indices={v.GetName():int(v.getIndex()) for v in workspace.allCats()
            if not observables or not observables.find(v.GetName())},
        mass_argument=float(workspace.var('MH').getVal()) if workspace.var('MH') else None)


@contextmanager
def combine_v11_fit_context(workspace):
    """Use native default effective ranges during extraction, then restore them.

Removal must precede extraction: bounds can enter constraint normalizations.
The installed FitterAlgoBase::optimizeBounds policy is tested independently.
Fixed parameters remain fixed even when their RooRealVar ranges are removed.
"""
    state = workspace_fit_state(workspace)
    ranges = {n:(float(workspace.var(n).getMin()), float(workspace.var(n).getMax()))
        for n in state['nuisance_bounds_removed']}
    try:
        for n in ranges:
            workspace.var(n).removeMin(); workspace.var(n).removeMax()
        yield state, ranges
    finally:
        for n, (lower, upper) in ranges.items():
            var = workspace.var(n)
            if np.isfinite(lower):var.setMin(lower)
            else:var.removeMin()
            if np.isfinite(upper):var.setMax(upper)
            else:var.removeMax()


def attach_fit_policy(model, state, stored_ranges):
    """Retain stored bounds alongside the effective bounds used by the fitter."""
    from .parameter_info import model_parameter_info
    info = model_parameter_info(model)
    if 'fit_policy' in info.metadata:
        raise ValueError('Model already has a fit policy')
    lower = info.arrays['lower'].copy(); upper = info.arrays['upper'].copy()
    for i, name in enumerate(info.metadata['names']):
        # The equal-bound encoding of imported constants must remain immutable.
        if name in stored_ranges and lower[i] != upper[i]:
            lower[i], upper[i] = stored_ranges[name]
    lower.flags.writeable=False;upper.flags.writeable=False
    result = TensorBundle._from_frozen({**model.metadata, 'parameter_info':{**info.metadata, 'fit_policy':state}},
        {**model.arrays, 'parameter_info__stored_lower':lower, 'parameter_info__stored_upper':upper})
    model_parameter_info(result)
    return result


def check_native_fit_state(workspace, model):
    """Fail before timing if native defaults and imported fit state differ."""
    from .parameter_info import model_parameter_info
    info = model_parameter_info(model)
    expected = info.metadata.get('fit_policy')
    if expected is None:
        raise ValueError('Model lacks explicit Combine v11 fit policy; reimport before CLI fit timing')
    actual = workspace_fit_state(workspace)
    if actual != expected:
        keys = [key for key in expected if actual.get(key) != expected[key]]
        raise ValueError(f'Native/imported fit state differs: {keys}')
    optimized = set(actual['nuisance_bounds_removed'])
    for i, name in enumerate(info.metadata['names']):
        var = workspace.var(name)
        value = float(var.getVal())
        stored = (value,value) if var.isConstant() else (float(var.getMin()),float(var.getMax()))
        effective = (-np.inf,np.inf) if name in optimized and not var.isConstant() else stored
        if value != info.arrays['initial'][i] or stored != (info.arrays['stored_lower'][i],info.arrays['stored_upper'][i]):
            raise ValueError(f'Native/imported start or stored bounds differ: {name}')
        if effective != (info.arrays['lower'][i],info.arrays['upper'][i]):
            raise ValueError(f'Native/imported effective bounds differ: {name}')
    return actual


def mass_cli_arguments(state):
    """Preserve the imported mass explicitly; Combine's default is 120."""
    mass = state['mass_argument']
    if mass is None:
        return []
    if not np.isfinite(mass) or float(np.float32(mass)) != mass:
        raise ValueError('Imported MH is not exactly representable by the native float mass option')
    return ['--mass', repr(mass)]
