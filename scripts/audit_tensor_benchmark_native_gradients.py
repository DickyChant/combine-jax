"""Apply the unchanged HHVV native FD policy to portable benchmark outputs."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from compare_gpu_results import verified_refinement
from gpu_gradient_evidence import random_validation
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nll', type=Path)
    parser.add_argument('--gradient', type=Path)
    parser.add_argument('--torch-validation', type=Path,
                        help='Saved CPU PyTorch HHVV validation capture, containing values and gradients')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source-revision',
                        help='Explicit committed producer revision for historical captures; all source hashes must match')
    args = parser.parse_args()
    if args.torch_validation:
        if args.nll or args.gradient:
            parser.error('Use either a Torch validation capture or the paired benchmark reports')
    elif not (args.nll and args.gradient):
        parser.error('Both --nll and --gradient are required for paired benchmark reports')
    assert not args.output.exists()
    adapter = args.output.with_name(args.output.stem+'_input.json')
    arrays_path = Path('build')/(args.output.stem+'_input.npz')
    assert not adapter.exists() and not arrays_path.exists()
    if args.torch_validation:
        capture = json.loads(args.torch_validation.read_text())
        assert capture['status'] == 'passed' and not capture['jax_imported']
        rows = [row for row in capture['cases'] if row['case'] == 'hhvv']
        assert len(rows) == 1 and rows[0]['passed']
        value = grad = {**capture, **rows[0]}
        inputs = (args.torch_validation,)
        expected_status = 'passed'
    else:
        inputs = (args.nll, args.gradient)
        value, grad = [json.loads(p.read_text()) for p in inputs]
        assert value['operation'] == 'nll' and grad['operation'] == 'value_gradient'
        expected_status = 'measured_pending_validation'
    producer = 'scripts/validate_tensor_torch.py' if args.torch_validation else (
        'scripts/benchmark_tensor_torch.py' if 'torch_version' in value else 'scripts/benchmark_tensor.py')
    source_evidence = []
    for report in (value, grad):
        assert report['status'] == expected_status and report['float64'] and not report['root_imported']
        assert sha(report['arrays']) == report['arrays_sha256']
        assert report['model_sha256'] == {n: sha(Path(report['model'])/n) for n in ('model.json','arrays.npz')}
        source_evidence.append(verified_source_record(report, producer, args.source_revision))
    identity_keys = ('host','model_sha256','sources')
    identity_keys += ('torch_version',) if 'torch_version' in value else ('jax_version','jaxlib_version')
    if not args.torch_validation:
        identity_keys += ('backend','mode','devices')
    for key in identity_keys:
        assert value[key] == grad[key]
    accepted_path = Path('reports/tensor_hhvv_unique_gradients.json')
    accepted = json.loads(accepted_path.read_text())
    assert accepted['status'] == 'passed'
    # The adapter changes only names of stored arrays; the policy and all frozen
    # native references remain exactly those used for the accepted CPU result.
    for p, digest in accepted['checker_sources'].items():
        assert sha(p) == digest
    previous = json.loads(Path(accepted['capture']).read_text())
    assert sha(accepted['capture']) == accepted['capture_sha256']
    assert sha(previous['arrays']) == previous['arrays_sha256']
    native_path = Path('reports/hhvv_random_values.json')
    native = json.loads(native_path.read_text())
    expected_points = np.array([c['theta'] for c in native['cases']])
    expected_values = np.array([c['combine_nll'] for c in native['cases']])
    with np.load(value['arrays'], allow_pickle=False) as a, np.load(grad['arrays'], allow_pickle=False) as b, np.load(previous['arrays'], allow_pickle=False) as old:
        for table in (a, b):
            np.testing.assert_array_equal(table['parameter_names'], native['parameter_names'])
            np.testing.assert_array_equal(table['points'], expected_points)
        np.testing.assert_array_equal(old['validation_points'], expected_points)
        fixed = np.isin(b['parameter_names'], previous['fixed_parameters'])
        assert fixed.sum() == 4 and b['gradients'].shape == (26, 2656)
        np.testing.assert_array_equal(b['gradients'][:,fixed], np.zeros((26,4)))
        assert np.isfinite(b['gradients']).all()
        errors = dict(nll=float(np.max(np.abs(a['values']-expected_values))),
                      value_gradient=float(np.max(np.abs(b['values']-expected_values))))
        assert max(errors.values()) < 1e-6
        np.savez_compressed(arrays_path, parameter_names=b['parameter_names'], point=b['points'][0],
            gradient=b['gradients'][0], validation_points=b['points'], validation_labels=old['validation_labels'],
            validation_values=a['values'], validation_vg_values=b['values'], validation_gradients=b['gradients'])
    capture = dict(status='native_policy_input', producer_script=str(Path(__file__)), producer_sha256=sha(__file__),
        source=native['source'], parameters=2656, arrays=str(arrays_path), arrays_sha256=sha(arrays_path),
        numeric_inputs={str(p): sha(p) for p in inputs},
        accepted_policy=str(accepted_path), accepted_policy_sha256=sha(accepted_path),
        scope='Lossless array-name adapter for the unchanged native gradient comparison tools; this is not a new numerical capture or reference selection.')
    save_json(adapter, capture)
    reference = json.loads(Path(accepted['random']['validation']).read_text())
    commands = [
        [sys.executable,'scripts/reconcile_gpu_gradient_references.py',str(adapter),
            '--directions-reference','reports/gpu_combine_directions_refined.json'],
        [sys.executable,'scripts/compare_gpu_random_gradients.py',str(adapter),
            '--reference',reference['reference'],'--stable-reference',reference['stable_reference'],
            '--refined-full-nll-reference',reference['refinements']['full_nll_reference'],
            '--refined-intensity-reference',reference['refinements']['intensity_reference']]]
    for command in commands:
        subprocess.run(command, check=True)
    nominal = verified_refinement(adapter, capture)
    random = random_validation(adapter, capture, sha(adapter), refined=True)
    assert nominal['status'] == nominal['directions_status'] == random['status'] == 'passed'
    assert random['status_counts'] == dict(passed=66300, fixed_bounds=100)
    for group, keys in [('nominal', ('reference_sha256','directions_reference_sha256')),
                        ('random', ('reference_sha256','stable_reference_sha256','refinement_documents_sha256'))]:
        actual = nominal if group == 'nominal' else random
        assert all(actual[k] == accepted[group][k] for k in keys)
    random_details = json.loads(Path(random['validation']).read_text())
    result = dict(status='passed', script_sha256=sha(__file__), adapter=str(adapter), adapter_sha256=sha(adapter),
        benchmark_inputs=capture['numeric_inputs'], native_values_reference=str(native_path),
        source_evidence=source_evidence,
        native_values_reference_sha256=sha(native_path), accepted_policy=str(accepted_path),
        accepted_policy_sha256=sha(accepted_path), nominal=nominal, random=random,
        random_max_relative_error=max(p['max_relative_error'] for p in random_details['points']),
        nll_max_absolute_errors=errors,
        input_kind='torch_cpu_validation_capture' if args.torch_validation else 'paired_benchmark_reports',
        runtime={k: grad[k] for k in
            ('host','backend','mode','jax_version','jaxlib_version','torch_version','devices','float64','environment') if k in grad},
        policy='Unchanged native FD references, selection rules and comparison scripts. All selected derivatives must meet relative error <1e-4; no AD-based reference selection.',
        scope='Full HHVV NLL and gradient at nominal plus 25 random points; no Hessian or fit acceptance.')
    save_json(args.output, result)
    print('passed', nominal['max_relative_error'], result['random_max_relative_error'], errors, flush=True)


if __name__ == '__main__':
    main()
