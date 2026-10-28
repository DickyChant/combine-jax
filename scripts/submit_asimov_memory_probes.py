"""Submit per-call memory probes on A100/H100; no fit driver is invoked."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def worker_text(template, family, policies=('ad', 'structural', 'structural_sequential')):
    if family not in ('A100', 'H100'):
        raise ValueError('Expected a primary GPU family')
    if not policies or len(set(policies)) != len(policies) or not set(policies) <= {'ad', 'structural', 'structural_sequential'}:
        raise ValueError('Invalid gradient policies')
    original = 'bash scripts/run_tensor_lcg.sh scripts/run_tensor_edm_campaign.py'
    assert template.splitlines().count(original) == 1
    return template.replace(original + '\n',
        'bash scripts/run_tensor_lcg.sh scripts/run_asimov_memory_probe.py --family ' + family
        + ' --policies ' + ' '.join(policies) + '\n')


def submit_text(template, family, exclude_hosts):
    if family not in ('A100', 'H100'):
        raise ValueError('Expected a primary GPU family')
    for host in exclude_hosts:
        if not re.fullmatch(r'[A-Za-z0-9.-]+', host):
            raise ValueError('Invalid excluded hostname')
    requirement = f'regexp("{family}", GPUs_DeviceName)'
    for host in sorted({h for name in exclude_hosts for h in (name, name.split('.')[0])}):
        requirement += f' && (TARGET.Machine =!= "{host}")'
    assert template.count('regexp("A100|H100", GPUs_DeviceName)') == 1
    return template.replace('regexp("A100|H100", GPUs_DeviceName)', requirement)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assembly', 'directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--exclude-host', action='append', required=True)
    parser.add_argument('--policies', nargs='+', choices=('ad', 'structural', 'structural_sequential'),
                        default=['ad', 'structural', 'structural_sequential'])
    args = parser.parse_args()
    assert Path.cwd() == Path('/tmp/sqian/combine-jax') and socket.gethostname().split('.')[0] == 'lxplus905'
    assert not args.directory.exists() and not args.output.exists()
    assembly = json.loads(args.assembly.read_text())
    assert assembly['status'] == 'assembled_asimov_pending_numerical_validation'
    payload, patch = (Path(assembly[k]) for k in ('payload', 'patch'))
    assert sha(payload) == assembly['payload_sha256'] and sha(patch) == assembly['patch_sha256']
    work = payload.parent / 'work'
    runtime = json.loads((work / 'build/edm_runtime_manifest.json').read_text())
    assert all(sha(work / p) == h for p, h in runtime['sources'].items())
    manifest_path = work / 'build/tensor_benchmark_inputs/manifest.json'
    assert sha(manifest_path) == runtime['manifest_sha256']
    assert json.loads(manifest_path.read_text())['hhvv']['dataset'] == 'asimov_prefit'
    args.directory.mkdir()
    report = dict(status='submitting_asimov_memory_only', script_sha256=sha(__file__),
        assembly_sha256=sha(args.assembly), source_revision=assembly['source_revision'],
        payload_sha256=sha(payload), patch_sha256=sha(patch), dataset='asimov_prefit',
        dataset_identity=assembly['dataset_identity'], excluded_hosts=args.exclude_host, policies=args.policies, jobs=[],
        scope='A100/H100 CPU and GPU value+gradient comparisons; no fits or covariance computation. Existing fit jobs are neither changed nor interrupted. Separate native gates required.')
    save_json(args.output, report)
    for family in ('A100', 'H100'):
        directory = args.directory / family.lower()
        directory.mkdir()
        os.link(payload, directory / 'payload.tar.gz')
        os.link(patch, directory / 'runtime_patch.tar')
        worker = directory / 'worker.sh'
        worker.write_text(worker_text((work / 'scripts/condor_tensor_edm.sh').read_text(), family, args.policies))
        worker.chmod(0o755)
        submit = directory / 'job.sub'
        submit.write_text(submit_text((work / 'scripts/condor_tensor_edm.sub').read_text(), family, args.exclude_host))
        result = subprocess.run(['condor_submit', '-spool', 'job.sub'], cwd=directory, capture_output=True, text=True)
        row = dict(family=family, directory=str(directory), submitted_unix=time.time(),
            command=['condor_submit', '-spool', 'job.sub'], returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, worker_sha256=sha(worker), submit_sha256=sha(submit))
        report['jobs'].append(row)
        save_json(args.output, report)
        assert result.returncode == 0, result.stderr
        match = re.search(r'cluster (\d+)', result.stdout)
        assert match, result.stdout
        row['job_id'] = match[1] + '.0'
        save_json(args.output, report)
    report['status'] = 'submitted_asimov_memory_only'
    save_json(args.output, report)
    print(report['status'], [j['job_id'] for j in report['jobs']], flush=True)


if __name__ == '__main__':
    main()
