"""Replay the saved A100 Asimov fit comparison; never run another fit."""
import json
from pathlib import Path

from audit_tensor_edm_comparison import compare
from audit_tensor_edm_timing import gpu_memory_summary
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    directory = Path('build/asimov_a100_tensor_first_results')
    original = directory / 'gpu_tensor_edm_gpu_fit_native_point.json'
    raw_fit = directory / 'gpu_tensor_edm_gpu_fit.json'
    point = json.loads(original.read_text())
    assert point['dataset'] == 'asimov_prefit' and point['fit_sha256'] == sha(raw_fit)
    assert point['status'] == 'passed_point_only'
    fit_path = Path('reports/gpu_tensor_asimov_a100_completed_gpu_fit.json')
    native_path = Path('reports/gpu_tensor_asimov_a100_completed_native_fit.json')
    fit = json.loads(fit_path.read_text())
    assert fit['collection']['original_report_sha256'] == sha(raw_fit)
    assert point['arrays_sha256'] == fit['arrays_sha256']
    point['fit'] = str(fit_path)
    point['fit_sha256'] = sha(fit_path)
    point['collection'] = dict(original_report=str(original), original_report_sha256=sha(original),
        original_fit_sha256=sha(raw_fit), relocation='Only fit report identity/path changed to the previously verified collected copy; numerical point and all native evaluations unchanged.')
    point_path = Path('reports/gpu_tensor_asimov_a100_completed_gpu_fit_native_point.json')
    assert not point_path.exists()
    save_json(point_path, point)
    revision = 'a5e26272f9904428d369a2ef32f4036edd90263f'
    result = compare(fit_path, native_path, point_path,
        manifest='build/tensor_hhvv_asimov/manifest.json', case='hhvv',
        source_revision=revision, allow_uncertified_curvature=True)
    worker_path = directory / 'gpu_tensor_edm_gpu_fit_comparison.json'
    worker = json.loads(worker_path.read_text())
    for key in ('status', 'minimum_nll_difference', 'max_parameter_delta_over_sigma',
                'same_point_nll_error', 'parameter_gate_passed', 'minimum_nll_gate_passed'):
        assert result[key] == worker[key]
    assert result['status'] == 'failed_minimum_agreement'
    result['collection_audit'] = dict(script_sha256=sha(__file__), job_id='16967869.0',
        raw_worker_comparison=str(worker_path), raw_worker_comparison_sha256=sha(worker_path),
        reproduced_worker_comparison=True)
    result['measurement'] = dict(timings=fit['timings'], compiler_memory=fit['compiler_memory'],
        resources=fit['resources'], gpu_memory=gpu_memory_summary(fit), fit_result=fit['fit_result'],
        scope='Historical full-curvature fit under its recorded 28 GiB guard. It is not a result of the newly approved 8 GiB gradient-only campaign. No speed ratio because the native minimum gate failed.')
    hessian_path = Path('reports/gpu_tensor_asimov_a100_completed_gpu_hessian.json')
    hessian = json.loads(hessian_path.read_text())
    result['nominal_hessian_measurement'] = dict(report=str(hessian_path), report_sha256=sha(hessian_path),
        source=verified_source_record(hessian, 'scripts/benchmark_tensor.py', revision),
        timings=hessian['timings'], resources=hessian['resources'], compiler_memory=hessian['compiler_memory'],
        gpu_memory=gpu_memory_summary(hessian), numerical_validation='Full-matrix comparison pending; memory and timing only.')
    output = Path('reports/gpu_tensor_asimov_a100_completed_gpu_fit_comparison.json')
    assert not output.exists()
    save_json(output, result)
    print(result['status'], result['minimum_nll_difference'], result['max_parameter_delta_over_sigma'])


if __name__ == '__main__':
    main()
