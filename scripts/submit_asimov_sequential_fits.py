"""Submit the two approved Asimov fits on the hosts of the native references."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def submit_description(template, family, host):
    if family not in ('A100', 'H100') or not re.fullmatch(r'[A-Za-z0-9.-]+', host):
        raise ValueError('Invalid GPU family or native hostname')
    assert not host.split('.')[0].startswith('lxplus')
    requirement = (f'regexp("{family}", GPUs_DeviceName) && '
                   f'((TARGET.Machine == "{host}") || (TARGET.Machine == "{host.split(".")[0]}"))')
    assert template.count('regexp("A100|H100", GPUs_DeviceName)') == 1
    return template.replace('regexp("A100|H100", GPUs_DeviceName)', requirement).replace(
        '+JobFlavour = "workday"', '+JobFlavour = "tomorrow"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('assembly', 'directory', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    host = socket.gethostname().split('.')[0]
    assert host.startswith('lxplus') and host != 'lxplus905'
    assert Path.cwd() == Path('/tmp/sqian/combine-jax')
    assert not args.directory.exists() and not args.output.exists()
    assembly = json.loads(args.assembly.read_text())
    assert assembly['status'] == 'staged_asimov_sequential_gpu_fit' and assembly['dataset'] == 'asimov_prefit'
    payload, patch = (Path(assembly[k]) for k in ('payload', 'patch'))
    assert sha(payload) == assembly['payload_sha256'] and sha(patch) == assembly['patch_sha256']
    work = payload.parent / 'work'
    runtime_path = work / 'build/sequential_fit_runtime.json'
    assert sha(runtime_path) == assembly['runtime_manifest_sha256']
    runtime = json.loads(runtime_path.read_text())
    assert all(sha(work / p) == h for p, h in {**runtime['sources'], **runtime['input_files']}.items())
    args.directory.mkdir()
    report = dict(status='submitting', host=socket.gethostname(), script_sha256=sha(__file__),
        assembly=str(args.assembly), assembly_sha256=sha(args.assembly),
        source_revision=assembly['source_revision'], dataset='asimov_prefit',
        static_guard_gib=8, native_hosts=assembly['native_hosts'], jobs=[],
        authorization='User explicitly approved one H100 and one A100 sequential-gradient, gradient-only EDM Asimov fit, with conditional r_hhvv covariance rows.',
        scope='Pinned to each native-reference host for same-node timing. No T4 or lxplus905 access. No automatic resubmission.')
    save_json(args.output, report)
    for family in ('H100', 'A100'):
        directory = args.directory / family.lower()
        directory.mkdir()
        os.link(payload, directory / 'payload.tar.gz')
        os.link(patch, directory / 'runtime_patch.tar')
        worker = directory / 'worker.sh'
        original = 'bash scripts/run_tensor_lcg.sh scripts/run_tensor_edm_campaign.py'
        template = (work / 'scripts/condor_tensor_edm.sh').read_text()
        assert template.splitlines().count(original) == 1
        worker.write_text(template.replace(original + '\n',
            'bash scripts/run_tensor_lcg.sh scripts/run_asimov_sequential_fit.py --family ' + family + '\n'))
        worker.chmod(0o755)
        submit = directory / 'job.sub'
        submit.write_text(submit_description((work / 'scripts/condor_tensor_edm.sub').read_text(),
                                             family, assembly['native_hosts'][family]))
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
        print(family, row['job_id'], flush=True)
    report['status'] = 'submitted_asimov_sequential_gpu_fits'
    save_json(args.output, report)


if __name__ == '__main__':
    main()
