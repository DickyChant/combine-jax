"""Audit every independent-channel scaling capture and summarize cold compilation."""
import argparse
import json
from pathlib import Path

import numpy as np

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    manifest, driver = read(args.manifest), read(args.driver)
    assert manifest['status'] == 'prepared_native_values_and_oracle_gradients'
    assert driver['status'] == 'completed_pending_audit'
    assert driver['manifest_sha256'] == sha(args.manifest)
    assert driver['script_sha256'] == manifest['script_sha256']
    assert len(driver['commands']) == 18
    native = read(manifest['native_reference'])
    assert sha(manifest['native_reference']) == manifest['native_reference_sha256']
    assert len(native['cases']) == 26
    assert all(sha(p) == digest for p, digest in manifest['oracle_sources'].items())
    references = {}
    for family, ref in manifest['families'].items():
        assert sha(ref['reference']) == ref['reference_sha256']
        with np.load(ref['reference'], allow_pickle=False) as archive:
            a = {name: archive[name] for name in archive.files}
        indices = [native['parameter_names'].index(n) for n in a['parameter_names']]
        np.testing.assert_array_equal(a['points'], np.array([c['theta'] for c in native['cases']])[:, indices])
        expected = [next(r['combine'] for r in c['channels'] if r['name'] == ref['source_channel'])
                    for c in native['cases']]
        np.testing.assert_array_equal(a['native_values'], expected)
        assert max(abs(a['oracle_values']-a['native_values'])) < 1e-6
        assert len(a['parameter_names']) == ref['parameters_per_channel']
        assert np.isfinite(a['gradients']).all() and a['gradients'].shape == a['points'].shape
        references[family] = a
    rows, sources, runtime = [], None, None
    previous_end = driver['started_unix']
    for command in driver['commands']:
        assert command['status'] == 'passed' and command['returncode'] == 0
        assert previous_end <= command['started_unix'] < command['finished_unix']
        previous_end = command['finished_unix']
        argv = command['command']
        path = Path(argv[argv.index('--output')+1])
        report = read(path)
        assert report['status'] == 'measured_native_values_old_ad_gradients'
        assert report['manifest_sha256'] == sha(args.manifest)
        assert report['script_sha256'] == driver['script_sha256']
        assert report['float64'] and report['backend'] == 'cpu' and not report['cache_enabled']
        assert 0 < len(report['cpu_affinity']) <= 8
        assert not report['root_imported']
        verified = verified_source_record(report, 'scripts/benchmark_tensor_independent_scaling.py', args.source_revision)
        identity = {k: report[k] for k in ('host', 'cpu_model', 'cpu_affinity', 'environment', 'jax_version', 'jaxlib_version')}
        if sources is None: sources, runtime = verified, identity
        assert sources == verified and runtime == identity
        family, count = report['family'], report['channels']
        case = next(c for c in manifest['models'] if c['family'] == family and c['channels'] == count)
        assert report['case'] == case
        assert {n: sha(Path(case['model'])/n) for n in ('model.json', 'arrays.npz')} == case['model_sha256']
        assert sha(report['arrays']) == report['arrays_sha256']
        reference = references[family]
        selection = (np.arange(26)[:, None]+np.arange(count)[None, :]) % 26
        with np.load(report['arrays'], allow_pickle=False) as archive:
            np.testing.assert_array_equal(archive['parameter_names'],
                [f'copy_{i}__{n}' for i in range(count) for n in reference['parameter_names']])
            np.testing.assert_array_equal(archive['points'], reference['points'][selection].reshape(26, -1))
            expected = reference['native_values'][selection].sum(axis=1)
            for operation in ('nll', 'value_gradient'):
                values = archive[operation+'_values']
                assert values.dtype == np.float64 and values.shape == (26,) and np.isfinite(values).all()
            errors = {op: float(max(abs(archive[op+'_values']-expected))) for op in ('nll', 'value_gradient')}
            assert max(errors.values()) < 1e-6
            expected_gradient = reference['gradients'][selection].reshape(26, -1)
            actual = archive['gradients']
            assert actual.dtype == np.float64 and actual.shape == expected_gradient.shape and np.isfinite(actual).all()
            delta = abs(actual-expected_gradient)
            relative = float(np.max(delta/np.maximum(abs(expected_gradient), 1e-8)))
            assert relative < 1e-4
        for operation, row in report['operations'].items():
            assert row['native_nll_max_absolute_error'] == errors[operation]
            assert row['closed_over_arrays'] == 0
            assert row['compile_seconds'] > 0 and row['lower_seconds'] > 0
            assert row['static_allocation_bytes'] < 8*1024**3
        assert report['operations']['value_gradient']['old_ad_gradient_max_relative_error'] == relative
        rows.append(dict(report=str(path), report_sha256=sha(path), arrays_sha256=report['arrays_sha256'],
            log=str(path.with_suffix('.log')), log_sha256=sha(path.with_suffix('.log')),
            family=family, channels=count, parameters=case['parameters'],
            buffer_bytes=case['buffer_bytes'], batches=case['batches'],
            operations=report['operations'], peak_rss_mib=report['resources']['peak_rss_mib']))
    assert previous_end <= driver['finished_unix']
    summaries = []
    for family in references:
        for operation in ('nll', 'value_gradient'):
            sizes = []
            for count in (1, 4, 16):
                selected = [row for row in rows if row['family'] == family and row['channels'] == count]
                assert len(selected) == 3
                sizes.append(dict(channels=count, parameters=selected[0]['parameters'],
                    batches=selected[0]['batches'],
                    top_level_equations=sorted({r['operations'][operation]['top_level_equations'] for r in selected}),
                    recursive_equations=sorted({r['operations'][operation]['recursive_equations'] for r in selected}),
                    compile_seconds=[r['operations'][operation]['compile_seconds'] for r in selected],
                    median_compile_seconds=float(np.median([r['operations'][operation]['compile_seconds'] for r in selected])),
                    median_trace_lower_compile_seconds=float(np.median([r['operations'][operation]['trace_lower_compile_seconds'] for r in selected]))))
            summaries.append(dict(family=family, operation=operation, sizes=sizes,
                constant_batched_trace=sizes[1]['recursive_equations'] == sizes[2]['recursive_equations'],
                compile_ratio_16_over_1=sizes[2]['median_compile_seconds']/sizes[0]['median_compile_seconds'],
                compile_ratio_16_over_4=sizes[2]['median_compile_seconds']/sizes[1]['median_compile_seconds']))
    save_json(args.output, dict(status='native_values_old_ad_gradients_validated_scaling_measured',
        script_sha256=sha(__file__), manifest=str(args.manifest), manifest_sha256=sha(args.manifest),
        driver=str(args.driver), driver_sha256=sha(args.driver), source_evidence=sources, runtime=runtime,
        float64=True, cases=rows, summaries=summaries,
        all_batched_traces_constant=all(s['constant_batched_trace'] for s in summaries),
        native_nll_max_absolute_error=max(r['operations'][o]['native_nll_max_absolute_error'] for r in rows for o in ('nll', 'value_gradient')),
        old_ad_gradient_max_relative_error=max(r['operations']['value_gradient']['old_ad_gradient_max_relative_error'] for r in rows),
        scope='Two real HHVV structures; all per-channel floating parameters, including POIs, are independent. '
              'Constraints are excluded. Three fresh processes per size; sequential owned workloads on a '
              'shared CPU host. All 26 mixed native channel-value points and old-graph AD gradients checked '
              'per capture. Native gradient agreement is not newly claimed by the graph comparison. '
              'These measured counts/timings do not establish scaling for all census structures.'))
    print('validated', [(s['family'], s['operation'], [r['median_compile_seconds'] for r in s['sizes']]) for s in summaries], flush=True)


if __name__ == '__main__':
    main()
