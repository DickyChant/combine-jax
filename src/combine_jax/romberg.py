"""Framework-neutral numerical normalization with RooIntegrator1D conventions.

ROOT's trapezoidal refinements, five-estimate extrapolation, error tests and
raw-estimate fallback are retained. This is the native numerical procedure,
not a replacement fixed integration grid. See ROOT 6.30 RooRombergIntegrator.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

from .graph import UnsupportedModelError


@dataclass(frozen=True)
class RombergConfig:
    max_steps: int = 20
    min_steps_zero: int = 999
    fixed_steps: int = 0
    extrapolate: bool = True
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7

    def __post_init__(self):
        if not (1 <= self.max_steps <= 30 and 0 <= self.fixed_steps <= self.max_steps
                and self.min_steps_zero >= 1 and self.eps_abs > 0 and self.eps_rel > 0
                and math.isfinite(self.eps_abs+self.eps_rel)):
            raise UnsupportedModelError('Invalid or unsupported native Romberg configuration')


def extract_config(pdf):
    config = pdf.getIntegratorConfig()
    config = config if config else pdf.defaultIntegratorConfig()
    if str(config.method1D().getLabel()) != 'RooIntegrator1D':
        raise UnsupportedModelError('Only native RooIntegrator1D numerical normalization is translated')
    section = config.getConfigSection('RooIntegrator1D')
    if str(section.getCatLabel('sumRule')) != 'Trapezoid':
        raise UnsupportedModelError('Native Romberg midpoint summation is not translated')
    rule = str(section.getCatLabel('extrapolation'))
    if rule not in ('None', 'Wynn-Epsilon'):
        raise UnsupportedModelError('Unknown native Romberg extrapolation')
    return asdict(RombergConfig(max_steps=int(section.getRealValue('maxSteps')),
        min_steps_zero=int(section.getRealValue('minSteps')),fixed_steps=int(section.getRealValue('fixSteps')),
        extrapolate=rule=='Wynn-Epsilon',eps_abs=float(config.epsAbs()),eps_rel=float(config.epsRel())))


def extrapolate(history):
    """Neville extrapolation at zero from the five latest squared step sizes.

    All step sizes share a power-of-four factor, so the normalized abscissae
    below preserve the native arithmetic ratios without a large runtime table.
    """
    h = (1., .25, .0625, .015625, .00390625)
    c = [history[i] for i in range(5)]
    d = list(c)
    result = history[-1]
    error = history[-1]*0.
    for order in range(1,5):
        divided = [(c[i+1]-d[i])/(h[i]-h[i+order]) for i in range(5-order)]
        c = [h[i]*v for i,v in enumerate(divided)]
        d = [h[i+order]*v for i,v in enumerate(divided)]
        error = d[-1]
        result = result+error
    return result,error


def integrate_numpy(function, parameters, low, high, config=RombergConfig()):
    """Return (native integral, refinement count, return mode, converged)."""
    width = high-low
    if not math.isfinite(low+high) or width <= 0:
        raise UnsupportedModelError('Romberg normalization needs finite positive bounds')
    history = np.zeros(5,dtype=np.float64)
    for level in range(1,config.max_steps+1):
        if level==1:
            raw=.5*width*np.sum(function(parameters,np.array([low,high],dtype=np.float64)))
        else:
            count=1 << (level-2); total=0.
            for start in range(0,count,64):
                x=low+(np.arange(start,min(start+64,count),dtype=np.float64)+.5)*(width/count)
                total+=np.sum(function(parameters,x))
            raw=.5*(history[-1]+width*total/count)
        history=np.r_[history[1:],raw]
        # Retain the native minSteps zero-convergence condition, including its
        # comparison to the current estimate (normally disabled by minSteps=999).
        if level>=config.min_steps_zero and raw<config.eps_abs/width:
            return 0.,level,2,True
        if config.fixed_steps:
            if level==config.fixed_steps:return raw,level,0,True
        elif level>=5:
            value,error=extrapolate(history) if config.extrapolate else (raw,raw-history[-2])
            if abs(error)<=max(config.eps_abs,config.eps_rel*abs(value)):
                return value,level,int(config.extrapolate),True
    return raw,config.max_steps,0,False
