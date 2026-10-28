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


@pytest.mark.parametrize('operation', ['hessian', 'fit'])
def test_compile_only_never_emits_numerical_or_fit_results(counting_path, tmp_path, operation):
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, Path('tests/cards/counting.txt'))
    model.save(tmp_path / 'model')
    output = tmp_path / 'compile.json'
    command = [sys.executable, 'scripts/benchmark_tensor.py', '--model', str(tmp_path / 'model'),
        '--backend', 'cpu', '--operation', operation, '--compile-only', '--output', str(output)]
    if operation == 'fit':
        command += ['--fit-stopping-criterion', 'edm', '--fit-covariance', 'none', '--fit-edm-validation', 'estimated']
    completed = subprocess.run(command, capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'compiled_not_executed' and report['compile_only']
    assert report['compiler_memory']['temp_size_in_bytes'] >= 0
    assert 'first_execute_seconds' not in report['timings']
    assert 'arrays' not in report and not output.with_suffix('.npz').exists()
    assert 'fit_result' not in report and 'first_fit_checkpoint' not in report


@pytest.mark.parametrize('options,message', [
    (['--compile-only', '--mode', 'eager'], 'requires jit'),
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
