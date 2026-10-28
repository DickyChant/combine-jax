"""Analytic derivatives and an off-centre kink test finite-step locality."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from fd_reference import estimates, first_consistent_triple


def probes(function, point, steps):
    point = np.longdouble(point)
    result = []
    for step in steps:
        h = np.longdouble(step)
        deltas = {k:function(point+k*h)-function(point) for k in (-2,-1,1,2)}
        result.append(dict(step=float(h),estimates=estimates(deltas,h)))
    return result


def test_quartic_local_convergence_and_derivative():
    function = lambda x: 3*x+2*x*x+x**4
    point = np.longdouble('.7')
    result = probes(function,point,(.1,.05,.02,.01,.005,.002,.001,.0005,.0002,.0001))
    selected = first_consistent_triple(result)
    assert selected and selected["first_probe_index"] > 0
    derivative = float(3+4*point+4*point**3)
    first = selected["first_probe_index"]
    assert all(abs(v-derivative)/abs(derivative) < 1e-5
               for p in result[first:first+3] for v in p["estimates"].values())
    assert all(abs(p["estimates"]["central_four"]-derivative) < 1e-11 for p in result)


def test_central_plateau_does_not_override_directional_inconsistency():
    boundary = np.longdouble('1e-7')
    function = lambda x: x+np.longdouble('.02')*max(np.longdouble(0),x-boundary)
    result = probes(function,0,(.01,.005,.002,.001,.0005,.0002,.0001,5e-5,2e-5,
                                1e-5,5e-6,2e-6,1e-6,5e-7,2e-7,1e-7,5e-8,2e-8,1e-8,5e-9))
    coarse = [p["estimates"]["central_four"] for p in result[:3]]
    assert (max(coarse)-min(coarse))/min(coarse) < 1e-5
    assert all(abs(v-1.) > .005 for v in coarse)
    selected = first_consistent_triple(result)
    assert selected and selected["first_probe_index"] > 2
    first = selected["first_probe_index"]
    assert all(abs(v-1.) < 1e-12 for p in result[first:first+3] for v in p["estimates"].values())


def test_uncertain_and_nonfinite_estimates_cannot_pass():
    uncertain = [dict(estimates=dict(a=1.,b=1.01,c=.99,d=1.)) for _ in range(3)]
    assert first_consistent_triple(uncertain) is None
    nonfinite = [dict(estimates=dict(a=1.,b=1.,c=1.,d=np.nan)) for _ in range(3)]
    assert first_consistent_triple(nonfinite) is None
