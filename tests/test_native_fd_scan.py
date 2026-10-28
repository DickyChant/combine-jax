import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fd_scan import scan_coordinate
from capture_asimov_native_reference import checked_native_value


class Variable:
    def __init__(self,value):self.value=value
    def getVal(self):return self.value
    def setVal(self,value):self.value=value


@pytest.mark.parametrize('coordinate,lower,upper',[(.7,-math.inf,math.inf),(0.,0.,1.),(1.,0.,1.)])
def test_native_scan_central_and_boundary(coordinate,lower,upper):
    variable=Variable(coordinate)
    value=lambda: math.exp(variable.getVal())-math.exp(coordinate)
    result=scan_coordinate(variable,value,scale=1.,lower=lower,upper=upper,zero=0.)
    assert result['status']=='converged' and variable.getVal()==coordinate
    for probe in result['probes'][-3:]:
        assert probe['derivative']==pytest.approx(math.exp(coordinate),rel=1e-7)


def test_tiny_derivative_and_fixed_coordinate_are_not_certified():
    v=Variable(0.)
    assert scan_coordinate(v,lambda:0.,scale=1.,lower=-1.,upper=1.,zero=0.)['status']=='roundoff_sensitive'
    assert scan_coordinate(v,lambda:0.,scale=1.,lower=0.,upper=0.,zero=0.)['status']=='fixed_bounds'


@pytest.mark.parametrize('value,errors',[(9e9,0),(9e9+10.,0),(float('nan'),0),(1.,1)])
def test_native_finite_penalties_and_errors_are_rejected(value,errors):
    root=SimpleNamespace(RooAbsReal=SimpleNamespace(clearEvalErrorLog=lambda:None,numEvalErrors=lambda:errors))
    with pytest.raises(ValueError,match='Invalid native NLL'):
        checked_native_value(root,SimpleNamespace(getVal=lambda:value))
