"""Audit the three explicitly collected October 23 Asimov Condor jobs.

Reads saved artifacts only. No remote operations or fit execution.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

from audit_tensor_edm_timing import gpu_memory_summary
from run_tensor_edm_campaign import hhvv_native_outcome
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_provenance import verified_source_record


def copy_report(source, target):
    document = json.loads(source.read_text())
    if 'arrays' in document:
        numeric = source.parent / Path(document['arrays']).name
        assert sha(numeric) == document['arrays_sha256']
        destination = target.with_suffix('.npz')
        if not destination.exists():
            os.link(numeric, destination)
        assert sha(destination) == document['arrays_sha256']
        document['arrays'] = str(destination)
    document['collection'] = dict(original_report=str(source), original_report_sha256=sha(source),
                                  path_relocation_only=True)
    assert not target.exists()
    save_json(target, document)
    return document


def main():
    manifest = Path('build/tensor_hhvv_asimov/manifest.json')
    case = json.loads(manifest.read_text())['hhvv']
    assert case['dataset'] == 'asimov_prefit'
    metadata = json.loads((Path(case['model']) / 'model.json').read_text())['metadata']
    hashes = {n: sha(Path(case['model']) / n) for n in ('model.json', 'arrays.npz')}
    jobs_path = Path('build/condor_collection_20261023/jobs.json')
    jobs = {str(j['ClusterId']) + '.0': j for j in json.loads(jobs_path.read_text())}
    assert set(jobs) == {'16967869.0', '16968268.0', '16968269.0'}
    for job in jobs.values():
        assert job['JobStatus'] == 4 and job['ExitCode'] == 0 and not job['ExitBySignal']
    output = Path('reports/gpu_tensor_asimov_completed_memory_collection.json')
    assert not output.exists()
    result = dict(status='checking', dataset='asimov_prefit', script_sha256=sha(__file__),
                  jobs=jobs, scheduler_capture_sha256=sha(jobs_path), measurements=[],
                  scope='Saved Asimov results only; process GPU attribution stays unavailable when hidden by the Condor PID namespace. No fit-speed claim.')
    reference = json.loads(Path('reports/gpu_tensor_asimov_full_native_audit.json').read_text())
    extra = []
    for path, digest in reference['artifacts'].items():
        flag = ('--refinement' if 'native_refinement' in path else
                '--cached-deltas' if ('cached_delta' in path or 'refine_points' in path) else
                '--centered-scores' if 'centered_score' in path else
                '--native-shard' if ('partition_14_19' in path or 'fd_tail_' in path) else None)
        if flag:
            assert sha(path) == digest
            extra += [flag, path]
    assert len(extra) == 30
    for family, job_id in [('a100', '16968268.0'), ('h100', '16968269.0')]:
        directory = Path('build/asimov_' + family + '_memory_three_policy_results')
        prefix = 'gpu_tensor_edm_memory_probe_'
        driver = json.loads((directory / (prefix + 'driver.json')).read_text())
        assert driver['dataset'] == 'asimov_prefit'
        assert driver['status'] == 'measurements_finished_pending_numerical_audit'
        revision = driver['source_revision']
        assert revision == '07e04f953faa2782f3c8a6a318423ea3a77d15fc'
        for label in ('driver', 'environment'):
            copy_report(directory / (prefix + label + '.json'),
                        Path(f'reports/gpu_tensor_asimov_{family}_memory_{label}.json'))
        assert json.loads((directory / 'worker_exit.json').read_text())['exit_code'] == 0
        for backend in ('gpu', 'cpu'):
            for policy in ('ad', 'structural', 'structural_sequential'):
                label = backend + '_' + policy
                target = Path(f'reports/gpu_tensor_asimov_{family}_memory_{label}.json')
                d = copy_report(directory / (prefix + label + '.json'), target)
                assert d['model_sha256'] == hashes and d['float64'] and not d['root_imported']
                assert d['status'] == 'measured_pending_validation'
                evidence = verified_source_record(d, 'scripts/benchmark_tensor.py', revision)
                audit = target.with_name(target.stem + '_native_audit.json')
                command = [sys.executable, 'scripts/audit_asimov_native_reference.py',
                    '--manifest', str(manifest), '--native', 'reports/gpu_hhvv_asimov_native_base_partition.json',
                    '--capture', str(target), *extra, '--output', str(audit)]
                with audit.with_suffix('.log').open('x') as log:
                    checked = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                gate = json.loads(audit.read_text())
                assert checked.returncode == 0 and gate['status'] == 'passed'
                memory = d['compiler_memory']
                static = (memory['argument_size_in_bytes'] + memory['output_size_in_bytes']
                          + memory['temp_size_in_bytes'] - memory['alias_size_in_bytes'])
                row = dict(job_id=job_id, family=family.upper(), backend=backend, policy=policy,
                    report=str(target), report_sha256=sha(target), native_audit=str(audit),
                    native_audit_sha256=sha(audit), source_evidence=evidence,
                    host=d['host'], environment=d['environment'], jax_version=d['jax_version'],
                    jaxlib_version=d['jaxlib_version'], devices=d['devices'], static_bytes=static,
                    timings=d['timings'], resources=d['resources'], native_gate_status=gate['status'],
                    gpu_memory=gpu_memory_summary(d) if backend == 'gpu' else None)
                result['measurements'].append(row)
                save_json(output, result)
                print(family, label, 'native gate passed', flush=True)
    directory = Path('build/asimov_a100_tensor_first_results')
    revision = 'a5e26272f9904428d369a2ef32f4036edd90263f'
    native_path = Path('reports/gpu_tensor_asimov_a100_completed_native_fit.json')
    native = copy_report(directory / 'gpu_tensor_edm_native_fit.json', native_path)
    evidence = verified_source_record(native, 'scripts/benchmark_combine_fit.py', revision)
    outcome = hhvv_native_outcome(native, metadata, manifest_sha256=sha(manifest),
                                  workspace_sha256=sha(case['workspace']), returncode=0)
    assert outcome['minimum_comparison_eligible']
    worker = json.loads((directory / 'worker_exit.json').read_text())
    attachments = {}
    for name, record in worker['attachments'].items():
        path = directory / record['file']
        assert sha(path) == record['sha256']
        attachments[str(path)] = record['sha256']
    for label in ('environment', 'driver', 'gpu_fit', 'cpu_fit', 'gpu_hessian', 'cpu_hessian'):
        source = directory / ('gpu_tensor_edm_' + label + '.json')
        copy_report(source, Path('reports/gpu_tensor_asimov_a100_completed_' + label + '.json'))
    result['native'] = dict(job_id='16967869.0', report=str(native_path), report_sha256=sha(native_path),
        source_evidence=evidence, state_gate=outcome, host=native['host'],
        migrad_seconds=native['migrad_seconds'], executable_wall_seconds=native['executable_wall_seconds'],
        edm=native['fit']['edm'], evaluation_counts=native['evaluation_counts'],
        conservative_combined_peak_rss_mib=native['conservative_combined_peak_rss_mib'],
        verified_attachments=attachments)
    result['status'] = 'passed_twelve_native_gradient_gates_and_native_fit_state'
    save_json(output, result)
    print(result['status'], flush=True)


if __name__ == '__main__':
    main()
