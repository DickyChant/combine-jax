"""Verify exact-bound coordinate values and raw AD curvature on CPU or GPU."""
import argparse
import os
from pathlib import Path

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu','gpu'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jax.numpy as jnp
    import numpy as np
    assert jax.config.x64_enabled and all(d.platform == args.backend for d in jax.devices())
    print('verified devices', jax.devices(), flush=True)
    from combine_jax.graph import Parameter
    from combine_jax.tensor.parameter_info import pack_parameter_info, fit_coordinates, COORDINATE_FIELDS
    from combine_jax.tensor.fitting import make_fit_function, physical_parameters
    info = pack_parameter_info([Parameter('rate',1.,0.,10.,.1,'poi'),
                               Parameter('shape',2.,-10.,10.,1.,'nuisance')])
    raw = fit_coordinates(info)
    coordinates = jax.device_put({key:raw[key] for key in COORDINATE_FIELDS})
    point = jnp.array([-10.,-1.3])
    def objective(x, arrays):
        delta = x[1]-.7
        return jnp.where(x[0]>=0, .5*x[0]**2+x[0]+.5*delta**2+.1*x[0]*delta, jnp.nan)
    monitor = Resources(args.backend == 'gpu', args.output)
    try:
        uncorrected = jax.jit(lambda z, origin, scale: origin+z*scale)(
            point, coordinates['reference'], coordinates['scale'])
        corrected = jax.jit(physical_parameters)(point, coordinates)
        jacobian = jax.jit(jax.jacfwd(physical_parameters))(point, coordinates)
        result = jax.device_get(jax.jit(make_fit_function(objective, max_iterations=200,
            gradient_tolerance=1e-9))(coordinates, {}))
        assert float(corrected[0]) == 0.
        np.testing.assert_array_equal(jacobian, np.diag([.1,1.]))
        assert result['status'] == result['covariance_status'] == 0
        assert result['iteration'] < 200 and float(result['values'][0]) == 0.
        np.testing.assert_allclose(result['values'], [0.,.7], atol=1e-9, rtol=0)
        hessian = np.array([[1.,.1],[.1,1.]])
        np.testing.assert_allclose(result['hessian'], hessian, atol=1e-14, rtol=0)
        np.testing.assert_allclose(result['covariance'], np.linalg.inv(hessian), atol=1e-14, rtol=0)
        report = dict(provenance(), status='passed', script_sha256=sha(__file__), backend=args.backend,
            jax_version=jax.__version__, float64=True, devices=[d.device_kind for d in jax.devices()],
            raw_mapped_point=np.array(uncorrected), corrected_point=np.array(corrected),
            coordinate_jacobian=np.array(jacobian), result=result,
            expected_minimum=[0.,.7], expected_hessian=hessian,
            scope='Known constrained-quadratic regression, exact physical bound and affine AD derivatives. This establishes a coordinate-roundoff fix, not HHVV convergence or native uncertainty acceptance.')
    finally:
        monitor.close()
    report['resources'] = monitor.snapshot()
    save_json(args.output, report)
    print(report['status'], int(result['iteration']), result['values'], flush=True)


if __name__ == '__main__':
    main()
