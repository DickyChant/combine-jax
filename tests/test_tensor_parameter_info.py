from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from combine_jax.graph import Parameter
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import pack_parameter_info,validate_parameter_info,fit_coordinates


def fixture():
    return pack_parameter_info([Parameter('r',1.,0.,10.,.2,'poi'),Parameter('nuisance',-.5,-np.inf,np.inf,1.,'nuisance'),
        Parameter('fixed',3.,3.,3.,0.,'other'),Parameter('flat',12.,-20.,20.,0.,'nuisance')])


def test_neutral_metadata_and_physical_scaled_coordinates(tmp_path):
    info=fixture();info.save(tmp_path/'parameters');info=TensorBundle.load(tmp_path/'parameters')
    fit=fit_coordinates(info,fixed={'nuisance':.7})
    np.testing.assert_array_equal(fit['reference'],[1.,.7,3.,12.]);np.testing.assert_array_equal(fit['free_indices'],[0,3])
    np.testing.assert_allclose(fit['scale'],[.2,1.2]);np.testing.assert_allclose(fit['lower'],[-5.,-32/1.2]);np.testing.assert_allclose(fit['upper'],[45.,8/1.2])
    np.testing.assert_array_equal(fit['physical_lower'],[0.,-20.]);np.testing.assert_array_equal(fit['physical_upper'],[10.,20.])
    assert info.metadata['roles']==['poi','nuisance','other','nuisance'] and info.arrays['error'][-1]==0.
    script='''import sys
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import fit_coordinates
m=TensorBundle.load(sys.argv[1]);fit_coordinates(m)
assert not any(n in sys.modules for n in ('jax','ROOT','torch'))
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path/'parameters')],check=True,capture_output=True)


def test_invalid_metadata_and_fixed_bounds_are_rejected():
    info=fixture()
    for field,value in [('lower',np.array([2.,-np.inf,3.,-20.])),('upper',np.array([0.,np.inf,3.,20.])),('error',np.array([.2,-1.,0.,0.]))]:
        with pytest.raises(ValueError):validate_parameter_info(TensorBundle(info.metadata,{**info.arrays,field:value}))
    for kw in ({'fixed':{'missing':0.}},{'fixed':{'r':-1.}},{'fixed':{'fixed':2.}},{'initial':[1.,0.,2.,12.]},{'initial':[1.,np.nan,3.,12.]}):
        with pytest.raises(ValueError):fit_coordinates(info,**kw)
    all_fixed=fit_coordinates(info,fixed=dict(zip(info.metadata['names'],info.arrays['initial'],strict=True)))
    assert all_fixed['free_indices'].shape==(0,) and all_fixed['scale'].shape==(0,)
