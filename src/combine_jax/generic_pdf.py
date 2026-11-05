"""Allowlisted GenericPdf values and native numerical normalization."""
import math

import jax.numpy as jnp

from .formula import extract_formula, formula_value, evaluate_formula
from .graph import UnsupportedModelError
from .romberg import extract_config, RombergConfig
from .romberg_jax import integrate
from .root_io import load_root


def extract_generic(root):
    if root.normRange():
        raise UnsupportedModelError('GenericPdf named normalization range is not translated')
    inputs,attrs=extract_formula(root)
    attrs['integration_config']=extract_config(root)
    attrs['fixed_normalizations']={}
    # A shape with only one floating real leaf has no floating shape parameter
    # when that leaf is its integration axis. Fold the native integral itself,
    # retaining the range and stack provenance, rather than a fitted surrogate.
    variables=[v for v in root.getVariables() if v.InheritsFrom('RooRealVar') and not v.isConstant()]
    if len(variables)==1:
        axis=variables[0];low,high=float(axis.getMin()),float(axis.getMax())
        if math.isfinite(low+high) and high>low:
            R=load_root();normalization=root.createIntegral(R.RooArgSet(axis))
            value=float(normalization.getVal())
            if not math.isfinite(value) or value<=0:
                raise UnsupportedModelError('Nonfinite/nonpositive fixed GenericPdf normalization')
            attrs['fixed_normalizations'][str(axis.GetName())]=dict(bounds=(low,high),value=value,
                root_version=str(R.gROOT.GetVersion()),method='native integral with other leaves fixed')
    return inputs,attrs


def generic_integral(node,context,axes):
    if len(axes)!=1:
        raise UnsupportedModelError(f'{node.name}: only one-dimensional GenericPdf normalization is translated')
    axis=axes[0];low,high=context.bounds(axis)
    fixed=node.attrs['fixed_normalizations'].get(axis)
    if fixed and (low,high)==tuple(fixed['bounds']):
        return jnp.asarray(fixed['value'],dtype=jnp.float64)
    if axis not in node.inputs or any(context.depends_on(n,axis) for n in node.inputs if n!=axis):
        raise UnsupportedModelError(f'{node.name}: numerical GenericPdf needs a direct axis and independent scalar arguments')
    index=node.inputs.index(axis);names=tuple(n for n in node.inputs if n!=axis)
    parameters=jnp.stack(context.values(names)) if names else jnp.empty(0,dtype=jnp.float64)
    if parameters.ndim!=1:
        raise UnsupportedModelError(f'{node.name}: conditional GenericPdf normalization needs batched lowering')
    def function(p,x):
        values=[p[i] for i in range(len(names))];values.insert(index,x)
        return jnp.broadcast_to(evaluate_formula(node.attrs['ast'],values),jnp.shape(x))
    return integrate(function,parameters,low,high,RombergConfig(**node.attrs['integration_config']))


generic_value=formula_value
