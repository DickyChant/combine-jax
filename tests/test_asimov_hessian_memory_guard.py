from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_asimov_hessian_pilot import memory_guard


@pytest.mark.parametrize('total,used,expected',[(80,1,56),(96,1,67.2),(96,60,27),(128,1,68)])
def test_guard_respects_physical_free_and_allocator_headroom(total,used,expected):
    actual=memory_guard(dict(device_total_mib=total*1024,device_baseline_mib=used*1024,error=None))
    assert actual==pytest.approx(expected)
    assert actual<.75*total and actual<.8*(total-used)


@pytest.mark.parametrize('total,used,error',[(0,0,None),(80,80,None),(80,-1,None),(float('nan'),0,None),(80,0,'NVML failed')])
def test_invalid_device_memory_is_not_an_allocation_budget(total,used,error):
    with pytest.raises(ValueError):memory_guard(dict(device_total_mib=total,device_baseline_mib=used,error=error))
