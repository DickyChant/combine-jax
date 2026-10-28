"""Audit fresh-process GPU cache hits, numerical differences and native gates."""
import argparse
import json
from pathlib import Path
import re

import numpy as np

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--native-audits', type=Path, nargs=2, required=True)
    parser.add_argument('--values-audit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    driver = json.loads(args.driver.read_text())
    assert driver['script_sha256'] == sha('scripts/run_tensor_phase2.py')
    native = [json.loads(path.read_text()) for path in args.native_audits]
    values = json.loads(args.values_audit.read_text())
    assert all(row['status'] == 'passed' for row in native) and values['status'] == 'passed'
    native_inputs = {path: digest for row in native for path, digest in row['benchmark_inputs'].items()}
    assert len(native_inputs) == 4
    assert len(driver['cache_snapshots']) == 4
    records = {}
    for command in driver['commands']:
        argv = command['command']
        if 'scripts/benchmark_tensor.py' not in argv or '--cache' not in argv:
            continue
        assert command['returncode'] == 0 and command['status'] == 'passed'
        path = Path(argv[argv.index('--output')+1])
        state = path.stem.rsplit('_', 1)[1]
        measured = json.loads(path.read_text())
        assert measured['status'] == 'measured_pending_validation'
        assert measured['backend'] == 'gpu' and measured['mode'] == 'jit' and measured['float64']
        assert measured['points'] == 26 and not measured['root_imported']
        assert measured['script_sha256'] == sha('scripts/benchmark_tensor.py')
        assert all(sha(p) == digest for p, digest in measured['sources'].items())
        assert sha(measured['arrays']) == measured['arrays_sha256']
        accepted = next(row for row in values['cases'] if row['report'] == str(path))
        assert accepted['passed'] and accepted['report_sha256'] == sha(path)
        operation = measured['operation']
        assert native_inputs[str(path)] == sha(path)
        snapshots = [row for row in driver['cache_snapshots'] if row['operation'] == operation and row['state'] == state]
        assert len(snapshots) == 1
        records[operation, state] = dict(path=path, measured=measured, files=snapshots[0]['files'], log=Path(command['log']))
    assert set(records) == {(operation, state) for operation in ('nll','value_gradient') for state in ('cold','warm')}
    results = []
    for operation in ('nll','value_gradient'):
        cold, warm = [records[operation, state] for state in ('cold','warm')]
        left, right = cold['measured'], warm['measured']
        for key in ('host','devices','environment','model_sha256','sources','structure_key','cache','jax_version','jaxlib_version'):
            assert left[key] == right[key], key
        assert left['resources']['gpu']['device_uuid'] == right['resources']['gpu']['device_uuid']
        assert left['resources']['gpu']['process_id'] != right['resources']['gpu']['process_id']
        assert cold['files'] == warm['files'], 'Warm process changed cached executable contents'
        cold_log, warm_log = [row['log'].read_text() for row in (cold, warm)]
        hits = re.findall(r"Persistent compilation cache hit for 'jit_nll' with key '(jit_nll-[0-9a-f]+)'", warm_log)
        assert len(hits) == 1 and hits[0] not in cold_log
        assert 'Finished XLA compilation of jit(nll)' in cold_log
        entries = [path for path in warm['files'] if Path(path).name == hits[0]+'-cache']
        assert len(entries) == 1
        comparisons = {}
        with np.load(left['arrays'], allow_pickle=False) as a, np.load(right['arrays'], allow_pickle=False) as b:
            assert set(a.files) == set(b.files)
            for name in a.files:
                if name in ('parameter_names', 'points'):
                    np.testing.assert_array_equal(a[name], b[name], err_msg=name)
                    continue
                assert np.isfinite(a[name]).all() and np.isfinite(b[name]).all()
                difference = np.abs(a[name]-b[name])
                comparisons[name] = dict(bitwise_equal=bool(np.array_equal(a[name], b[name])),
                    differing_entries=int(np.count_nonzero(a[name] != b[name])),
                    maximum_absolute_difference=float(difference.max(initial=0.)),
                    maximum_relative_difference=float((difference/np.maximum(np.abs(a[name]),1e-8)).max(initial=0.)))
        results.append(dict(operation=operation, cold_report=str(cold['path']), cold_report_sha256=sha(cold['path']),
            warm_report=str(warm['path']), warm_report_sha256=sha(warm['path']),
            cold_log_sha256=sha(cold['log']), warm_log_sha256=sha(warm['log']),
            executable_key=hits[0], executable_sha256=warm['files'][entries[0]],
            unchanged_cache_files=len(cold['files']), output_comparisons=comparisons, points=26,
            cold_lower_seconds=left['timings']['lower_seconds'], warm_lower_seconds=right['timings']['lower_seconds'],
            cold_compile_write_seconds=left['timings']['compile_or_cache_load_seconds'],
            warm_cache_load_seconds=right['timings']['compile_or_cache_load_seconds'],
            device=left['devices'][0], host=left['host'], jax_version=left['jax_version'],
            driver=left['resources']['gpu']['device_metadata'], lcg_view=left['environment']['COMBINE_JAX_LCG_VIEW']))
    report = dict(status='passed_fresh_process_gpu_reuse', script_sha256=sha(__file__),
        driver=str(args.driver), driver_sha256=sha(args.driver),
        native_audits={str(path):sha(path) for path in args.native_audits}, values_audit=str(args.values_audit),
        values_audit_sha256=sha(args.values_audit), cases=results,
        initial_bitwise_probe='reports/gpu_tensor_primary_phase2_cache_bitwise_probe.log',
        initial_bitwise_probe_sha256=sha('reports/gpu_tensor_primary_phase2_cache_bitwise_probe.log'),
        scope='Separate processes on one assigned A100, unchanged executable checksums and explicit outer-NLL cache-hit logs. Both cold and warm outputs independently pass all-26-point native NLL/FD gates; gradients are not bitwise repeatable and their differences are recorded. Tracing/lowering remains. Cross-batch-job reuse is not established.')
    save_json(args.output, report)
    print(report['status'], [(r['operation'], r['cold_compile_write_seconds'], r['warm_cache_load_seconds']) for r in results], flush=True)


if __name__ == '__main__':
    main()
