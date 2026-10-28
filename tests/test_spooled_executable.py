"""Exercise exact kernel reloads, changed inputs and corrupted-artifact rejection."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from spooled_executable import SpooledExecutable
from combine_jax.local import release_compilation_cache


@pytest.mark.parametrize("operation", ("value", "value_and_grad", "hessian"))
def test_spooled_polynomial_matches_analytic_results_after_release(tmp_path, operation):
    def objective(point):
        x,y=point
        return 2*x*x+3*x*y+5*y*y+7*x+11*y
    function={"value":objective,"value_and_grad":jax.value_and_grad(objective),
              "hessian":jax.hessian(objective)}[operation]
    kernel=jax.jit(function)
    compiled=kernel.lower(np.zeros(2)).compile()
    stored,record=SpooledExecutable.write(compiled,tmp_path/(operation+".bin"),backend="cpu")
    assert record["serialized_bytes"]>0
    del kernel,compiled
    release_compilation_cache()
    for x,y in ((0.,0.),(.3,-.8),(-1.,2.)):
        actual=stored(np.array([x,y]))
        expected_value=2*x*x+3*x*y+5*y*y+7*x+11*y
        if operation=="value": expected=expected_value
        elif operation=="value_and_grad": expected=(expected_value,np.array([4*x+3*y+7,3*x+10*y+11]))
        else: expected=np.array([[4.,3.],[3.,10.]])
        for a,e in zip(jax.tree_util.tree_leaves(actual),jax.tree_util.tree_leaves(expected),strict=True):
            assert isinstance(a,np.ndarray)
            np.testing.assert_allclose(a,e,rtol=1e-14,atol=1e-14)
        assert all(v>=0 for v in stored.calls[-1].values())
    assert len(stored.calls)==3


def test_spooled_array_constants_and_integrity_guards(tmp_path):
    weights=np.array([.2,1.5,3.,4.])
    def objective(point):
        return jnp.sum(jnp.asarray(weights)*(point[0]+jnp.arange(4)*point[1])**2)
    kernel=jax.jit(jax.hessian(objective)); compiled=kernel.lower(np.zeros(2)).compile()
    path=tmp_path/"array.bin"
    stored,_=SpooledExecutable.write(compiled,path,backend="cpu")
    expected=np.asarray(compiled(np.array([.3,-.2])))
    with pytest.raises(FileExistsError):
        SpooledExecutable.write(compiled,path,backend="cpu")
    del kernel,compiled
    release_compilation_cache()
    np.testing.assert_array_equal(stored(np.array([.3,-.2])),expected)
    original=path.read_bytes();path.write_bytes(bytes([original[0]^1])+original[1:])
    with pytest.raises(ValueError,match="integrity"):
        stored(np.zeros(2))
