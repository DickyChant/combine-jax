"""Audit full-HHVV cache relocation and optional separate-job reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import tarfile

import numpy as np

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def read(path):
    return json.loads(Path(path).read_text())


def cache_hits(log):
    return re.findall(r"Persistent compilation cache hit for 'jit_nll' with key '(jit_nll-[0-9a-f]+)'",
                      Path(log).read_text())


def capture(driver, label, operation, native_path, revision):
    rows = [row for row in driver['commands'] if 'scripts/benchmark_tensor.py' in row['command']
            and row['command'][row['command'].index('--output')+1].endswith('_'+label+'_'+operation+'.json')]
    assert len(rows) == 1
    row = rows[0]
    assert row['status'] == 'passed' and row['returncode'] == 0
    argv = row['command']
    assert '--cache' in argv and argv[argv.index('--operation')+1] == operation
    path = Path(argv[argv.index('--output')+1])
    result = read(path)
    native = read(native_path)
    assert native['status'] == 'passed'
    assert native['random']['status_counts'] == dict(passed=66300, fixed_bounds=100)
    assert native['benchmark_inputs'][str(path)] == sha(path)
    assert result['status'] == 'measured_pending_validation' and not result['root_imported']
    assert result['backend'] == 'gpu' and result['mode'] == 'jit' and result['float64']
    assert result['points'] == 26 and result['cache']['auxiliary_xla_caches'] == 'none'
    assert sha(result['arrays']) == result['arrays_sha256']
    assert result['model_sha256'] == {n: sha(Path(result['model'])/n) for n in ('model.json','arrays.npz')}
    source = verified_source_record(result, 'scripts/benchmark_tensor.py', revision)
    evidence = dict(report=str(path), report_sha256=sha(path), source_evidence=source,
        log=row['log'], log_sha256=sha(row['log']),
        native_audit=str(native_path), native_audit_sha256=sha(native_path),
        cache_hit_keys=cache_hits(row['log']), timings=result['timings'],
        resources=result['resources'], jax_allocator=result.get('jax_allocator'),
        cache=result['cache'], host=result['host'], workdir=result['workdir'],
        devices=result['devices'], environment=result['environment'],
        jax_version=result['jax_version'], jaxlib_version=result['jaxlib_version'])
    return result, evidence


def compare(old, new):
    for key in ('operation','backend','mode','float64','model_sha256','structure_key',
                'jax_version','jaxlib_version','points_source_sha256','parameters','points','devices'):
        assert old[key] == new[key], key
    assert old['environment']['COMBINE_JAX_LCG_VIEW'] == new['environment']['COMBINE_JAX_LCG_VIEW']
    assert old['cache']['directory'] != new['cache']['directory'], 'Cache was not relocated'
    result = {}
    with np.load(old['arrays'], allow_pickle=False) as a, np.load(new['arrays'], allow_pickle=False) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            if name in ('parameter_names','points'):
                np.testing.assert_array_equal(a[name], b[name])
                continue
            assert a[name].shape == b[name].shape and np.isfinite(a[name]).all() and np.isfinite(b[name]).all()
            delta = abs(a[name]-b[name])
            result[name] = dict(bitwise_equal=bool(np.array_equal(a[name], b[name])),
                max_absolute_difference=float(delta.max(initial=0.)),
                max_relative_difference=float((delta/np.maximum(abs(a[name]),1e-8)).max(initial=0.)))
    # Each capture has already passed the full unchanged native policy. Record
    # inter-run roundoff without imposing bitwise equality or selecting references.
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--producer-driver', type=Path, required=True)
    parser.add_argument('--producer-submission', type=Path, required=True)
    parser.add_argument('--cold-native-audit', type=Path, required=True)
    parser.add_argument('--relocated-native-audit', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--consumer-driver', type=Path)
    parser.add_argument('--consumer-submission', type=Path)
    parser.add_argument('--consumer-native-audit', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    consumer_inputs = (args.consumer_driver, args.consumer_submission, args.consumer_native_audit)
    if any(consumer_inputs) and not all(consumer_inputs):
        parser.error('All three consumer inputs are required for a cross-job audit')
    assert not args.output.exists()
    producer = read(args.producer_submission)
    driver = read(args.producer_driver)
    assert driver['status'] == 'completed_pending_audit' and driver['stage'] == 'producer'
    snapshots = driver['snapshots']
    assert snapshots['cold_before'] == {}
    files = snapshots['cold_value_gradient_after']
    assert len(files) == 2 and snapshots['relocated_before'] == files
    archived = {}
    prefix = 'build/portable_consumer_cache/'
    with tarfile.open(args.archive) as stream:
        for member in stream.getmembers():
            if member.isfile() and member.name.endswith('-cache'):
                assert member.name.startswith(prefix)
                relative = member.name[len(prefix):]
                assert not Path(relative).is_absolute() and '..' not in Path(relative).parts
                assert relative not in archived
                archived[relative] = hashlib.file_digest(stream.extractfile(member), 'sha256').hexdigest()
    assert archived == snapshots['relocated_value_gradient_after']
    consumer_driver = consumer = None
    if all(consumer_inputs):
        consumer = read(args.consumer_submission)
        consumer_driver = read(args.consumer_driver)
        assert consumer['job_id'] != producer['job_id']
        assert consumer['cache_producer_job'] == producer['job_id']
        assert consumer['cache_input_sha256'] == sha(args.archive)
        assert consumer_driver['status'] == 'completed_pending_audit' and consumer_driver['stage'] == 'consumer'
        assert consumer_driver['snapshots']['cross_job_before'] == archived
    cases = []
    previous = snapshots['cold_before']
    for operation in ('nll','value_gradient'):
        old, cold = capture(driver, 'cold', operation, args.cold_native_audit, producer['source_commit'])
        after = snapshots['cold_'+operation+'_after']
        added = set(after)-set(previous)
        assert len(added) == 1 and all(after[k] == v for k,v in previous.items())
        executable = added.pop()
        key = Path(executable).name.removesuffix('-cache')
        assert key.startswith('jit_nll-') and not cold['cache_hit_keys']
        previous = after
        candidates = [('relocated', driver, args.relocated_native_audit, producer)]
        if consumer_driver:
            candidates.append(('cross_job', consumer_driver, args.consumer_native_audit, consumer))
        for label, owner, audit, submission in candidates:
            new, evidence = capture(owner, label, operation, audit, submission['source_commit'])
            if label == 'cross_job':
                assert old['workdir'] != new['workdir']
            unchanged = owner['snapshots'][label+'_'+operation+'_after'] == files
            reused = evidence['cache_hit_keys'] == [key] and unchanged
            cases.append(dict(operation=operation, stage=label, reused=reused,
                executable_key=key, executable_sha256=files[executable],
                cache_files_unchanged=unchanged, cold=cold, candidate=evidence,
                comparisons=compare(old,new)))
    passed = all(c['reused'] for c in cases)
    status = ('passed_cross_job_gpu_cache_reuse' if consumer_driver else 'passed_gpu_cache_relocation') if passed else 'cache_miss_numerically_valid'
    inputs = [args.producer_driver,args.producer_submission,args.cold_native_audit,args.relocated_native_audit]
    inputs += [p for p in consumer_inputs if p]
    save_json(args.output, dict(status=status, script_sha256=sha(__file__),
        inputs={str(p):sha(p) for p in inputs}, producer_job=producer['job_id'],
        consumer_job=consumer['job_id'] if consumer else None,
        archive=str(args.archive), archive_sha256=sha(args.archive), archived_executables=archived,
        cases=cases, scope='Full HHVV nominal plus 25 random native NLL/gradient gates, '
            'unaltered JAX executable compatibility keys, explicit outer cache-hit logs and '
            'transferred executable checksums. Lowering remains necessary. A cache miss is '
            'reported separately from numerical validity; no fit/Hessian acceptance.'))
    print(status, [(c['stage'],c['operation'],c['candidate']['timings']['compile_or_cache_load_seconds']) for c in cases], flush=True)


if __name__ == '__main__':
    main()
