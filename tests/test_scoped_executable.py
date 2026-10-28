"""Exact complete-matrix/batch evaluation with bounded executable lifetimes."""

from pathlib import Path
import sys
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from scoped_executable import LoadedExecutable, hessian_with_scoped_loads, value_gradient_points_with_scoped_loads
from spooled_executable import SpooledExecutable
from combine_jax.local import release_compilation_cache


def evaluator(tmp_path, operation, chunk=1):
    functions = [lambda p: 2*p[0]**2+3*p[0]*p[1]+5*p[1]**2+7*p[0],
                 lambda p: jnp.exp(p[0])+p[0]*p[1]+.2*p[1]**4,
                 lambda p: .3*p[0]*p[1]*p[2],
                 lambda p: jnp.asarray(3.)]
    indices = [np.array([0, 2]), np.array([1, 2]), np.array([0, 1, 2]),
               np.array([], dtype=int)]
    terms = []
    for i, (function, coordinates) in enumerate(zip(functions, indices, strict=True)):
        term = SimpleNamespace(indices=coordinates)
        if operation == "value_and_grad":
            kernel = jax.jit(jax.value_and_grad(function))
            arguments = (np.zeros(len(coordinates)),)
            name = operation
        elif len(coordinates):
            if chunk is None:
                kernel = jax.jit(jax.hessian(function))
                arguments = (np.zeros(len(coordinates)),)
                name = "hessian"
            else:
                gradient = jax.grad(function)
                kernel = jax.jit(lambda p, v, gradient=gradient:
                    jax.vmap(lambda d: jax.jvp(gradient, (p,), (d,))[1])(v))
                arguments = (np.zeros(len(coordinates)), np.zeros((chunk, len(coordinates))))
                name = "hessian_columns"
        else:
            terms.append(term)
            continue
        compiled = kernel.lower(*arguments).compile()
        stored, _ = SpooledExecutable.write(compiled, tmp_path/f"{operation}_{chunk}_{i}.bin", backend="cpu")
        setattr(term, name, stored)
        terms.append(term)
        del compiled, kernel
        release_compilation_cache()

    def point(theta):
        theta = np.asarray(theta, dtype=np.float64)
        if theta.shape != (3,):
            raise ValueError("Wrong point shape")
        return theta

    return SimpleNamespace(terms=terms, _point=point, hessian_chunk_size=chunk,
                           likelihood=SimpleNamespace(offset=1.25))


def expected(point):
    x, y, z = point
    value = 4.25+2*x*x+3*x*z+5*z*z+7*x+np.exp(y)+y*z+.2*z**4+.3*x*y*z
    gradient = np.array([4*x+3*z+7+.3*y*z, np.exp(y)+z+.3*x*z, 3*x+10*z+y+.8*z**3+.3*x*y])
    hessian = np.array([[4., .3*z, 3.+.3*y], [.3*z, np.exp(y), 1.+.3*x],
                        [3.+.3*y, 1.+.3*x, 10.+2.4*z*z]])
    return value, gradient, hessian


POINTS = np.array([[0., 0., 0.], [.3, -.8, .2], [-1., 2., -.5]])


@pytest.mark.parametrize("chunk", [None, 1, 2])
def test_full_hessian_keeps_every_entry_and_loads_once_per_term(tmp_path, chunk):
    model = evaluator(tmp_path, "hessian", chunk)
    for point in POINTS:
        np.testing.assert_allclose(hessian_with_scoped_loads(model, point), expected(point)[2],
                                   rtol=1e-14, atol=1e-14)
    for term in model.terms[:3]:
        stored = term.hessian if chunk is None else term.hessian_columns
        calls = 1 if chunk is None else int(np.ceil(len(term.indices)/chunk))
        assert len(stored.scoped_loads) == len(POINTS)
        assert all(load["calls"] == calls for load in stored.scoped_loads)
        assert len(stored.calls) == len(POINTS)*calls
        assert sum(c["deserialize_seconds"] > 0 for c in stored.calls) == len(POINTS)
        assert not stored._scoped_active


def test_reference_batch_includes_constant_terms_and_shared_coordinates(tmp_path):
    model = evaluator(tmp_path, "value_and_grad")
    values, gradients = value_gradient_points_with_scoped_loads(model, POINTS)
    np.testing.assert_allclose(values, [expected(p)[0] for p in POINTS], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(gradients, [expected(p)[1] for p in POINTS], rtol=1e-14, atol=1e-14)
    for term in model.terms:
        assert len(term.value_and_grad.scoped_loads) == 1
        assert len(term.value_and_grad.calls) == len(POINTS)
    with pytest.raises(ValueError, match="nonempty"):
        value_gradient_points_with_scoped_loads(model, np.empty((0, 3)))
    with pytest.raises(ValueError, match="shape"):
        value_gradient_points_with_scoped_loads(model, np.zeros((2, 4)))


def test_cleanup_on_exception_and_nested_scope_rejection(tmp_path):
    model = evaluator(tmp_path, "value_and_grad")
    stored = model.terms[0].value_and_grad
    with pytest.raises(RuntimeError, match="deliberate"):
        with LoadedExecutable(stored) as loaded:
            value, gradient = loaded(np.zeros(2))
            assert value.flags.owndata and gradient.flags.owndata
            with pytest.raises(RuntimeError, match="nested"):
                with LoadedExecutable(stored):
                    pass
            raise RuntimeError("deliberate")
    assert loaded.compiled is None and not stored._scoped_active
    assert stored.scoped_loads[-1]["exception_type"] == "RuntimeError"
    assert len(stored.calls) == 1
    with pytest.raises(RuntimeError, match="not open"):
        loaded(np.zeros(2))
    with pytest.raises(RuntimeError, match="reused"):
        with loaded:
            pass
    with LoadedExecutable(stored) as fresh:
        fresh(np.zeros(2))
    assert len(stored.calls) == 2


def test_integrity_check_precedes_loading(tmp_path):
    model = evaluator(tmp_path, "value_and_grad")
    stored = model.terms[0].value_and_grad
    stored.path.write_bytes(stored.path.read_bytes()+b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        with LoadedExecutable(stored):
            pass
    assert not stored.calls
    with pytest.raises(TypeError, match="process-created"):
        LoadedExecutable(object())
