"""Observed method gate on synthetic cards; no HHVV input is accessed."""
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    import jax
    from combine_jax.root_io import open_workspace, load_root
    from combine_jax.tensor.model import from_workspace
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.limit_profile import LimitProfiler
    from combine_jax.methods.asymptotic import observed_limit
    report_path = Path('reports/asymptotic_observed_generic_v10_validation.json')
    directory = Path('build/asymptotic_observed_generic_v10_validation')
    assert not directory.exists() and not report_path.exists()
    directory.mkdir()
    report = dict(status='running', scope='Synthetic counting, autoMCStats and parametric cards only.',
                  source_sha256=sha(__file__), native_combine_version='10.3.3', comparisons=[], relative_tolerance=.01)
    save_json(report_path, report)
    root = load_root()
    for case in ('counting', 'automcstats', 'parametric'):
        source = Path('build/observed_generic_native_v10_rebound')/case
        card = Path('build/tensor_native_policy_inputs')/case/'card.txt'
        prepared = directory/case/'prepared'
        target = directory/case
        target.mkdir()
        reference = json.loads(Path('build/observed_generic_native_v10_rebound/reference.json').read_text())
        row_native = next(r for r in reference['comparisons'] if r['case'] == case)
        assert row_native['workspace_sha256'] == sha(source/'workspace.root')
        command = row_native['command']
        native_seconds = row_native['native_seconds']
        values = [(-1., row_native['native_limit'])]
        with open_workspace(source/'workspace.root') as ws:
            observed = from_workspace(ws, card)
        from combine_jax.methods.workspace import prepare_expected_workspace
        asimov, _ = prepare_expected_workspace(source/'workspace.root', card, prepared, poi='r', run='expected')
        assert asimov.metadata['asymptotic_asimov']['mode'] == 'conditional_background_fit'
        begin = time.perf_counter()
        data = LimitProfiler(observed, 'r', global_up=.5, global_start_fraction=.1)
        expected = LimitProfiler(asimov, 'r', global_up=.5, global_start_fraction=.1)
        t = time.perf_counter()
        dmin = data.global_minimum()
        muhat = float(data.point()[data.index])
        amin = expected.global_minimum()
        result = observed_limit(data, expected, dmin, amin, muhat, data.r_max)
        elapsed = time.perf_counter()-t
        error = abs(result['observed_limit']/values[0][1]-1.)
        row = dict(case=case, status='passed' if error < .01 else 'failed', native_limit=values[0][1],
            native_command=command, native_seconds=native_seconds, native_log_sha256=row_native['log_sha256'],
            workspace_sha256=sha(source/'workspace.root'), conditional_asimov_preparation_sha256=sha(prepared/'preparation.json'),
            tensor_seconds=time.perf_counter()-begin, execution_seconds=elapsed,
            tensor_limit=result['observed_limit'], relative_error=error, result=result,
            data_fits=data.records, asimov_fits=expected.records,
            data_timings=data.timings, asimov_timings=expected.timings)
        report['comparisons'].append(row)
        save_json(report_path, report)
        print(case, row['status'], row['native_limit'], row['tensor_limit'], error, flush=True)
        del data, expected, observed, asimov
        jax.clear_caches()
    report['status'] = 'passed' if all(r['status']=='passed' for r in report['comparisons']) else 'failed'
    save_json(report_path, report)
    assert report['status'] == 'passed'


if __name__ == '__main__':
    main()
