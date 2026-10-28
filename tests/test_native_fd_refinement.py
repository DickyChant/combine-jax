import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fd_refinement import refine_coordinate,select_reference,REFINEMENT_FRACTIONS
from test_native_fd_scan import Variable


@pytest.mark.parametrize('coordinate,lower,upper',[(.3,-math.inf,math.inf),(0.,0.,1.),(1.,0.,1.)])
def test_exhaustive_native_refinement_and_restoration(coordinate,lower,upper):
    v=Variable(coordinate)
    def value():
        assert lower<=v.getVal()<=upper
        return math.exp(4*v.getVal())-math.exp(4*coordinate)
    r=refine_coordinate(v,value,scale=1.,lower=lower,upper=upper,zero=0.)
    assert r['status']=='converged' and len(r['probes'])==len(REFINEMENT_FRACTIONS)
    k=r['selected_first_probe_index']
    for p in r['probes'][k:k+3]:assert p['derivative']==pytest.approx(4*math.exp(4*coordinate),rel=1e-5)
    assert v.getVal()==coordinate


def test_full_spread_and_distinct_steps_not_just_adjacent_agreement():
    probes=[dict(status='valid',stencil='central',direction=1.,step=h,derivative=d)
            for h,d in zip((.1,.05,.02),(1.,1.000009,1.000018))]
    assert select_reference(probes)['status']=='unresolved'
    for p in probes:p['derivative']=1.;p['step']=.1
    assert select_reference(probes)['status']=='unresolved'


def test_uncertain_small_derivative_remains_uncertified():
    v=Variable(0.)
    r=refine_coordinate(v,lambda:1e-7*v.getVal(),scale=1.,lower=-1.,upper=1.,zero=0.)
    assert r['status']=='roundoff_sensitive'
    assert refine_coordinate(v,lambda:0.,scale=1.,lower=0.,upper=0.,zero=0.)['status']=='fixed_bounds'
