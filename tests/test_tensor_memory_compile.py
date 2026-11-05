"""Compile-only diagnostics and gradient-only EDM fits have explicit outputs."""
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.graph import Parameter
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.parameter_info import pack_parameter_info
import combine_jax.tensor.fitting as fitting
from test_tensor_fitting import numeric_coordinates


@pytest.mark.parametrize('operation,inverse_block', [('hessian',0), ('fit',0), ('fit',2)])
def test_compile_only_never_emits_numerical_or_fit_results(counting_path, tmp_path, operation,inverse_block):
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, Path('tests/cards/counting.txt'))
    model.save(tmp_path / 'model')
    output = tmp_path / 'compile.json'
    command = [sys.executable, 'scripts/benchmark_tensor.py', '--model', str(tmp_path / 'model'),
        '--backend', 'cpu', '--operation', operation, '--compile-only', '--output', str(output)]
    if operation == 'fit':
        command += ['--fit-stopping-criterion', 'edm', '--fit-covariance', 'none', '--fit-edm-validation', 'estimated','--fit-inverse-update-block-size',str(inverse_block)]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'compiled_not_executed' and report['compile_only']
    assert report['compiler_memory']['temp_size_in_bytes'] >= 0
    assert 'first_execute_seconds' not in report['timings']
    assert 'arrays' not in report and not output.with_suffix('.npz').exists()
    assert 'fit_result' not in report and 'first_fit_checkpoint' not in report
    if operation=='fit':assert report['fit_options']['inverse_update_block_size']==inverse_block


@pytest.mark.parametrize('options,message', [
    (['--compile-only', '--mode', 'eager'], 'requires jit'),
    (['--fit-inverse-update-block-size','-1'], 'must be nonnegative'),
    (['--operation','nll','--fit-inverse-update-block-size','2'], 'applies only to fit'),
    (['--fit-stopping-criterion', 'edm', '--fit-covariance', 'none'], 'require --fit-edm-validation estimated')])
def test_invalid_memory_options_fail_before_loading(tmp_path, options, message):
    output = tmp_path / 'rejected.json'
    result = subprocess.run([sys.executable, 'scripts/benchmark_tensor.py', '--model', 'unused',
        '--backend', 'cpu', '--operation', 'fit', '--output', str(output), *options],
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2 and message in result.stderr
    assert not output.exists()


def test_gradient_only_edm_never_traces_hessian_and_marks_errors_unavailable(monkeypatch):
    def forbidden(*args, **kwargs):
        def curvature(*unused):
            raise AssertionError('A gradient-only fit attempted a Hessian')
        return curvature
    monkeypatch.setattr(fitting, 'make_value_gradient_hessian', forbidden)
    info = pack_parameter_info([Parameter('x', 0., -10., 10., .5, 'poi'),
                               Parameter('y', 0., -10., 10., 2., 'nuisance')])
    def objective(x, a):
        delta = x - a
        return delta[0]**2 + .2*delta[0]*delta[1] + 2*delta[1]**2
    function = jax.jit(fitting.make_fit_function(objective, covariance=False, stopping_criterion='edm',
                                               edm_validation='estimated', edm_tolerance=1e-4))
    result = jax.device_get(function(numeric_coordinates(info), jnp.array([.7, -.3])))
    assert result['converged'] and result['corrected_estimated_edm'] < 1e-4
    assert result['hessian_evaluations'] == 0 and not result['edm_curvature_validated']
    assert result['covariance_status'] == 6 and not result['valid_covariance']
    assert np.isnan(result['covariance']).all() and np.isnan(result['reported_errors']).all()
    np.testing.assert_allclose(result['values'], [.7, -.3], atol=.01, rtol=0.)


@pytest.mark.parametrize('covariance', [False, True])
def test_omitting_matrices_preserves_fit_and_requested_curvature(covariance):
    info = pack_parameter_info([Parameter('x', 0., -10., 10., .5, 'poi'),
                               Parameter('y', 0., -10., 10., 2., 'nuisance')])
    def objective(x, a):
        delta = x-a
        return delta[0]**2+.2*delta[0]*delta[1]+2*delta[1]**2
    coordinates = numeric_coordinates(info); data = jnp.array([.7, -.3])
    results = []
    for retain in (True, False):
        fn = jax.jit(fitting.make_fit_function(objective, covariance=covariance, return_matrices=retain,
            stopping_criterion='edm', edm_validation='estimated', edm_tolerance=1e-4))
        results.append(jax.device_get(fn(coordinates, data)))
    full, compact = results
    assert set(full)-set(compact) == {'hessian', 'covariance'}
    for key in compact:
        np.testing.assert_allclose(compact[key], full[key], atol=1e-14, rtol=1e-14, equal_nan=True)
    assert compact['hessian_evaluations'] == int(covariance)
    assert compact['converged'] and compact['valid_covariance'] == covariance


def test_gradient_only_compact_result_has_linear_output_storage():
    count = 128
    info = pack_parameter_info([Parameter('x'+str(i), 0., -10., 10., 1., 'nuisance') for i in range(count)])
    coordinates = numeric_coordinates(info); data = jnp.ones(count, dtype=jnp.float64)
    def objective(x, a): return jnp.sum((x-a)**2)
    sizes = []
    for retain in (True, False):
        fn = jax.jit(fitting.make_fit_function(objective, covariance=False, return_matrices=retain,
            stopping_criterion='edm', edm_validation='estimated'))
        compiled = fn.lower(coordinates, data).compile()  # No fit execution.
        sizes.append(compiled.memory_analysis().output_size_in_bytes)
    assert sizes[0]-sizes[1] >= 8*count*count
    assert sizes[1] < 256*count
