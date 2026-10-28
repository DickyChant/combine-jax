"""Compare coordinate-aware AD covariance with installed Minuit on bounded toys.

This diagnostic uses iminuit 2.33's private serialized state to retain the actual
internal coordinates. Ext2int can move a near-bound point and is not substituted.
No HHVV likelihood, fit output, or acceptance policy is changed.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
import traceback

import numpy as np


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_case(kind):
    import jax
    import jax.numpy as jnp
    from iminuit import Minuit

    linear = kind == "linear_double_lower"
    upper = "upper" in kind
    limited = kind != "unbounded"
    double = "double" in kind
    low, high = (0., 10.) if double else ((None, 10.) if upper else (0., None))
    if not limited:
        low = high = None
    mean = 11. if upper else -1. if limited else 1.4

    def objective(point):
        x, y, z = point
        first = x if linear else .5*(x-mean)**2
        return first+.5*(y-.3*x)**2+.5*(z-4.)**2

    native = Minuit(lambda x, y, z: float(objective((x, y, z))),
                    x=9.8 if upper else .2, y=0., z=4.)
    native.errordef = .5
    native.tol = 1e-8
    native.strategy = 1
    native.limits["x"] = (low, high)
    native.fixed["z"] = True
    native.migrad(ncall=10000, iterate=1, use_simplex=False)
    native.hesse()
    fmin = native.fmin
    flags = {key: bool(getattr(fmin, key)) for key in (
        "is_valid", "has_accurate_covar", "has_posdef_covar", "has_made_posdef_covar",
        "is_above_max_edm", "hesse_failed", "has_reached_call_limit")}
    assert all(flags[k] for k in ("is_valid", "has_accurate_covar", "has_posdef_covar"))
    assert not any(flags[k] for k in ("has_made_posdef_covar", "is_above_max_edm",
                                    "hesse_failed", "has_reached_call_limit"))
    trafo = native._last_state.trafo
    assert [trafo.ext_of_int(i) for i in range(trafo.variable_parameters)] == [0, 1]
    saved_state = native._fmin._src.__getstate__()
    assert len(saved_state) == 6 and saved_state[2]
    internal = np.asarray(saved_state[2][-1].vec, dtype=np.float64).copy()
    point = np.asarray(native.values, dtype=np.float64)
    np.testing.assert_array_equal(
        [trafo.int2ext(i, float(v)) for i, v in enumerate(internal)], point[:2])

    def transform(u):
        if double:
            x = low+.5*(high-low)*(jnp.sin(u[0])+1.)
        elif limited and upper:
            x = high+1.-jnp.sqrt(1.+u[0]**2)
        elif limited:
            x = low-1.+jnp.sqrt(1.+u[0]**2)
        else:
            x = u[0]
        return jnp.array([x, u[1], 4.], dtype=jnp.float64)

    u = jnp.asarray(internal)
    external = np.asarray(transform(u))
    np.testing.assert_allclose(external, point, atol=1e-12, rtol=0.)
    gradient = np.asarray(jax.grad(objective)(jnp.asarray(point)))
    hessian = np.asarray(jax.hessian(objective)(jnp.asarray(point)))
    jacobian = np.asarray(jax.jacfwd(transform)(u))
    second = np.asarray(jax.jacfwd(jax.jacrev(transform))(u))
    curvature = np.asarray(jax.hessian(lambda v: objective(transform(v)))(u))
    extra = np.einsum("i,ijk->jk", gradient, second)
    chain = jacobian.T@hessian@jacobian+extra
    np.testing.assert_allclose(curvature, chain, atol=1e-10, rtol=1e-10)
    np.linalg.cholesky(curvature)
    inverse = np.linalg.inv(curvature)
    np.testing.assert_allclose(curvature@inverse, np.eye(2), atol=1e-10, rtol=0.)
    covariance = jacobian@inverse@jacobian.T
    native_covariance = np.asarray(native.covariance)
    errors = []
    for i, sigma in enumerate(np.sqrt(np.diag(inverse))):
        plus, minus = internal.copy(), internal.copy()
        plus[i] += sigma
        minus[i] -= sigma
        up = float(transform(jnp.asarray(plus))[i]-external[i])
        down = float(transform(jnp.asarray(minus))[i]-external[i])
        if i == 0 and double and sigma > 1.:
            up = high-low
        errors.append(.5*(abs(up)+abs(down)))
    errors = np.asarray(errors)
    native_errors = np.asarray(native.errors)[:2]
    error_difference = float(np.max(abs(errors-native_errors)/native_errors))
    normalizer = np.maximum(np.sqrt(np.outer(np.diag(native_covariance),
                                             np.diag(native_covariance))), 1e-8)
    covariance_difference = float(np.max(abs(covariance-native_covariance)/normalizer))
    raw = hessian[:2, :2]
    raw_rank = int(np.linalg.matrix_rank(raw))
    raw_sigmas = np.sqrt(np.diag(np.linalg.inv(raw))) if raw_rank == 2 else None
    recovered = np.asarray([trafo.ext2int(i, float(point[i])) for i in range(2)])
    inverse_roundtrip = np.asarray([trafo.int2ext(i, float(v)) for i, v in enumerate(recovered)])
    return dict(case=kind, status="passed" if error_difference < .01 and covariance_difference < .01 else "failed",
        native_flags=flags, native_edm=float(fmin.edm), native_nfcn=native.nfcn,
        bounds=[low, high], fixed={"z": 4.}, values=point.tolist(),
        native_actual_internal_coordinates=internal.tolist(),
        clipped_inverse_coordinates=recovered.tolist(),
        clipped_inverse_roundtrip_max_absolute_error=float(abs(inverse_roundtrip-point[:2]).max()),
        jax_transform_max_absolute_error=float(abs(external-point).max()),
        gradient=gradient.tolist(), external_hessian=hessian.tolist(),
        transformation_jacobian=jacobian.tolist(), transformation_second_derivatives=second.tolist(),
        gradient_chain_rule_term=extra.tolist(), internal_hessian=curvature.tolist(),
        internal_covariance=inverse.tolist(), external_covariance=covariance.tolist(),
        native_covariance=native_covariance.tolist(),
        ad_transformed_error_fields=errors.tolist(), native_error_fields=native_errors.tolist(),
        ad_external_covariance_sigmas=np.sqrt(np.diag(covariance))[:2].tolist(),
        native_external_covariance_sigmas=np.sqrt(np.diag(native_covariance))[:2].tolist(),
        error_max_relative_difference=error_difference,
        covariance_max_normalized_difference=covariance_difference,
        raw_external_hessian_rank=raw_rank,
        raw_external_inverse_sigmas=raw_sigmas.tolist() if raw_sigmas is not None else None,
        raw_sigma_vs_native_error_max_relative_difference=float(
            np.max(abs(raw_sigmas-native_errors)/native_errors)) if raw_sigmas is not None else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.parent != Path("reports") or args.output.suffix != ".json" or args.output.exists():
        parser.error("Use a fresh reports/*.json output")
    assert version("iminuit") == "2.33.0", "Private state layout is version-specific"
    assert os.environ["JAX_PLATFORMS"] == "cpu" and os.environ["JAX_ENABLE_X64"].lower() == "true"
    import jax
    import jaxlib
    import iminuit._core
    assert jax.config.jax_enable_x64 and jax.default_backend() == "cpu"
    started = time.perf_counter()
    report = dict(status="running", created_utc=datetime.now(timezone.utc).isoformat(),
        iminuit_version=version("iminuit"), jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        float64=True, cpu_affinity=sorted(os.sched_getaffinity(0)),
        sources={__file__: digest(__file__), iminuit._core.__file__: digest(iminuit._core.__file__)},
        external_sources=[
            "https://root.cern.ch/doc/v636/MnUserTransformation_8cxx_source.html",
            "https://root.cern.ch/doc/v636/SinParameterTransformation_8cxx_source.html"],
        tolerance=dict(error_relative=.01, covariance_normalized=.01), cases=[],
        scope="Synthetic Minuit/AD diagnostic, not Combine or HHVV validation. Actual installed "
              "Minuit internal coordinates are preserved. Full chain-rule curvature is inverted "
              "without repair, mapped to external covariance, and converted separately to the "
              "nonlinear Minuit error convention. No active fit or acceptance gate is changed.")
    try:
        for kind in ("unbounded", "double_lower", "double_upper", "single_lower", "single_upper",
                     "linear_double_lower"):
            result = run_case(kind)
            report["cases"].append(result)
            print(kind, result["status"], "error relative", result["error_max_relative_difference"],
                  "covariance normalized", result["covariance_max_normalized_difference"], flush=True)
        report["status"] = "passed" if all(c["status"] == "passed" for c in report["cases"]) else "failed"
    except BaseException:
        report.update(status="failed", exception=traceback.format_exc())
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        with args.output.open("x") as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+"\n")
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
