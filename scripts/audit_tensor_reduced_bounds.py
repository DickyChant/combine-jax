"""Independently check reduced-space optimizer fits from the isolated worktree."""
import argparse
import json
from pathlib import Path
import subprocess

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
from combine_jax.tensor.reported_errors import minuit_reported_errors
from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worktree', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and jax.config.x64_enabled
    root = args.worktree
    source = root/'reports/tensor_reduced_bounds_analytic_fits.json'
    report = json.loads(source.read_text())
    assert report['status'] == 'passed' and report['stage'] == 'complete'
    for path, digest in report['sources'].items():
        assert sha(root/path) == digest, path
    tests = Path('reports/tensor_reduced_bounds_tests_v2.log')
    release = Path('reports/tensor_reduced_bounds_release_line_tests.log')
    assert '24 passed, 2 warnings' in tests.read_text()
    assert '4 passed, 2 warnings' in release.read_text()
    summaries = []
    for case in report['cases']:
        assert case['status'] == 'passed' and not case['failed_gates']
        model = TensorBundle.load(root/case['model'])
        for name, digest in case['model_sha256'].items():
            assert sha(root/case['model']/name) == digest
        info = model_parameter_info(model)
        vg = jax.jit(jax.value_and_grad(make_model_function(model)))
        arrays = jax.device_put(dict(model.arrays))
        device, native = case['device'], case['native']
        assert device['status'] == device['covariance_status'] == 0
        assert native['valid'] and native['accurate_covariance']
        points = [('minimum', device['values'], native['nll'], device['projected_gradient'], fit_coordinates(info))]
        assert len(case['profiles']) == 2
        for p in case['profiles']:
            assert p['passed'] and p['device_status'] == 0 and p['native_valid']
            points.append(('profile_r_'+str(p['r']), p['device_values'], p['native_nll'], p['projected_gradient'],
                           fit_coordinates(info, initial=device['values'], fixed={'r':p['r']})))
        checks = []
        for label, point, expected, saved_gradient, c in points:
            point = np.array(point)
            value, gradient = map(np.asarray, vg(point, arrays))
            free = c['free_indices']
            z = (point[free]-c['reference'][free])/c['scale']
            scaled = gradient[free]*c['scale']
            eps = 32*np.finfo(np.float64).eps*(1+abs(z))
            blocked = ((z <= c['lower']+eps)&(scaled > 0))|((z >= c['upper']-eps)&(scaled < 0))
            pg = np.where(blocked, 0., scaled)
            np.testing.assert_allclose(pg, saved_gradient, atol=2e-10, rtol=1e-5)
            norm = float(np.max(abs(pg), initial=0.))
            error = abs(float(value)-expected)
            assert norm <= 1e-7 and error < 1e-6
            checks.append(dict(label=label, projected_gradient_max=norm, native_nll_absolute_error=error))
        hessian, covariance = np.array(device['hessian']), np.array(device['covariance'])
        assert np.linalg.eigvalsh(hessian).min() > 0
        np.testing.assert_allclose(hessian@covariance, np.eye(len(hessian)), atol=1e-8, rtol=0.)
        np.testing.assert_allclose(covariance, native['covariance'], atol=1e-5, rtol=.01)
        errors = np.sqrt(np.diag(covariance))
        np.testing.assert_array_equal(errors, device['errors'])
        reported, status = minuit_reported_errors(np.array(device['values']), errors,
                                                  info.arrays['lower'], info.arrays['upper'])
        np.testing.assert_array_equal(status, 0)
        np.testing.assert_allclose(reported, device['reported_errors'], atol=2e-13, rtol=2e-12)
        relative = abs(reported-native['errors'])/native['errors']
        assert np.max(relative) < .01
        tolerance = np.maximum(.01*abs(np.array(native['values'])), .01*np.array(native['errors']))
        assert np.all(abs(np.array(device['values'])-native['values']) < tolerance)
        summaries.append(dict(case=case['label'], checks=checks,
                              max_reported_error_relative_difference=float(np.max(relative))))
        jax.clear_caches()
    old = {}
    for path in Path('src/combine_jax').glob('*.py'):
        expected = subprocess.check_output(['git', 'show', '3217d9a:'+str(path)])
        assert expected == path.read_bytes() == (root/path).read_bytes()
        old[str(path)] = sha(path)
    save_json(args.output, dict(status='passed_small_cards_only', script_sha256=sha(__file__),
        worktree=str(root), producer=str(source), producer_sha256=sha(source),
        verified_producer_sources=report['sources'],
        test_logs={str(p):sha(p) for p in (tests, release)}, tests=28,
        analytic_cases=200, analytic_description='100 predefined SPD bound quadratics at offsets 0 and 1e7, plus lower/upper bound-release tests.',
        gradient_tolerance=1e-7, strict_poisson_gradient_tolerance=1e-8,
        armijo_roundoff_multiplier=8., cases=summaries,
        old_backend_unchanged_since='3217d9a', old_backend_sources=old,
        scope='Five generic baseline fits and ten profile-point gates pass, including native-objective covariance/error comparisons. Recomputed gradients independently check three analytic minima and six profiles. Existing exact-bound regression retained; Bernstein now needs no roundoff acceptance. No HHVV convergence or performance claim. Main optimizer source remains unchanged for its live CPU run.'))
    print('passed_small_cards_only: 28 tests, 3 analytic baselines and 6 profiles', flush=True)


if __name__ == '__main__':
    main()
