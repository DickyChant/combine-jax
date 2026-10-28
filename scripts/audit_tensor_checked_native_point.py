"""Check a tensor evaluation at the exact saved native HHVV point."""
import argparse
import json
from pathlib import Path

import numpy as np

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    plan = json.loads(args.plan.read_text())
    reference = json.loads(Path(plan['reference']).read_text())
    evaluation = json.loads(args.evaluation.read_text())
    assert sha(plan['reference']) == plan['reference_sha256']
    assert sha(reference['parent']) == reference['parent_sha256'] == plan['reference_parent_sha256']
    assert reference['status'] == 'point_checked'
    assert reference['combine_version'] == '11.0.0' and reference['root_version'] == '6.40.00'
    assert evaluation['status'] == 'measured_pending_validation'
    assert evaluation['operation'] == 'nll' and evaluation['float64']
    assert sha(evaluation['arrays']) == evaluation['arrays_sha256']
    assert evaluation['points_source_sha256'] == sha(plan['points']) == plan['points_sha256']
    assert evaluation['model_sha256'] == plan['model_sha256']
    for name, digest in evaluation['model_sha256'].items():
        assert sha(Path(evaluation['model'])/name) == digest
    imported = json.loads(Path('reports/tensor_hhvv_fit_import.json').read_text())
    assert imported['status'] == 'passed'
    assert {k: imported['source'][k] for k in reference['source']} == reference['source']
    assert imported['runtime_flags'] == reference['runtime_flags']
    for path, digest in imported['artifacts'].items():
        assert sha(path) == digest
    source = verified_source_record(evaluation, 'scripts/benchmark_tensor.py', args.source_revision)
    with np.load(evaluation['arrays'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['parameter_names'], reference['parameter_names'])
        np.testing.assert_array_equal(arrays['points'], [reference['point']])
        assert arrays['values'].shape == (1,)
        value = float(arrays['values'][0])
    expected = reference['canonical_candidate_nll']
    assert expected == plan['expected_native_nll']
    error = abs(value-expected)
    assert np.isfinite(value) and error < 1e-6
    save_json(args.output, dict(status='passed_point_only', script_sha256=sha(__file__),
        inputs={str(p): sha(p) for p in (args.plan, args.evaluation, Path(plan['reference']),
                                      Path('reports/tensor_hhvv_fit_import.json'))},
        source_evidence=source, native_nll=expected, tensor_nll=value,
        absolute_error=error, tolerance=1e-6, parameter_count=len(reference['point']),
        parameter_identity_verified=True, runtime_flags=reference['runtime_flags'],
        source=reference['source'], native_parent_status=reference['parent_status'],
        scope='NLL at the exact previously checked native HHVV point. No fit convergence, covariance, gradient, or timing acceptance; the local CPU fit overlapped this correctness check.'))
    print('passed_point_only', value, error, flush=True)


if __name__ == '__main__':
    main()
