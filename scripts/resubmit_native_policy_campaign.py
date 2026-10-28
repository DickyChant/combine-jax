"""Repair the empty output directory and submit the two frozen GPU payloads."""
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    assert Path.cwd() == Path('/tmp/sqian/combine-jax')
    assert socket.gethostname().split('.')[0] == 'lxplus905'
    base = Path('build/native_policy_stage/assembled')
    output = base / 'retry1.json'
    resume = output.exists()
    issue = json.loads((base / 'withdrawal_verified.json').read_text())
    assert issue['status'] == 'withdrawn_before_running'
    assembly = json.loads((base / 'assembly.json').read_text())
    payload = base / 'payload.tar.gz'
    assert sha(payload) == assembly['payload_sha256']
    patch = base / 'runtime_patch_retry1.tar'
    if not resume:
        assert not patch.exists()
        with tarfile.open(patch, 'w') as archive:
            member = tarfile.TarInfo('reports')
            member.type = tarfile.DIRTYPE
            member.mode = 0o755
            archive.addfile(member)
    # Exercise exactly the worker's extraction protocol, independently of the
    # assembler's existing work directory, before paying for any GPU allocation.
    check = base / 'retry1_extraction_check'
    if not resume:
        check.mkdir()
        subprocess.run(['tar', 'xzf', str(payload.resolve()), '-C', str(check)], check=True)
        subprocess.run(['tar', 'xf', str(patch.resolve()), '-C', str(check)], check=True)
    else:
        previous_report = json.loads(output.read_text())
        assert previous_report['status'] == 'extraction_verified' and not previous_report['jobs']
        assert sha(patch) == previous_report['patch_sha256']
        (base / 'retry1_before_python39_link_fix.json').write_text(output.read_text())
    assert (check / 'reports').is_dir()
    runtime = json.loads((check / 'build/edm_runtime_manifest.json').read_text())
    for name, digest in {**runtime['sources'], **runtime['input_files']}.items():
        assert sha(check / name) == digest, name
    assert (check / '.venv-combine-v11/bin/python').exists()
    report = dict(status='extraction_verified', observed_unix=time.time(),
                  script_sha256=sha(__file__), source_revision=runtime['source_revision'],
                  payload_sha256=sha(payload), patch_sha256=sha(patch),
                  patch_members=['reports/'], extraction_check=str(check),
                  runtime_manifest_sha256=sha(check / 'build/edm_runtime_manifest.json'),
                  preserved_issue=str(base / 'package_directory_issue.json'), jobs=[])
    if resume:
        report['resume_reason'] = 'Python 3.9 lacks Path.hardlink_to; the first attempt stopped before submission. Use os.link, recheck hashes, and preserve the extraction record.'
    output.write_text(json.dumps(report, indent=2) + '\n')
    for family in ('a100', 'h100'):
        previous = Path('build/condor_tensor_native_policy_' + family)
        directory = Path(str(previous) + '_retry1')
        directory.mkdir(exist_ok=resume)
        assert not list(directory.iterdir())
        os.link(payload, directory / 'payload.tar.gz')
        os.link(patch, directory / 'runtime_patch.tar')
        (directory / 'worker.sh').write_bytes((previous / 'worker.sh').read_bytes())
        (directory / 'worker.sh').chmod(0o755)
        (directory / 'job.sub').write_bytes((previous / 'job.sub').read_bytes())
        result = subprocess.run(['condor_submit', '-spool', 'job.sub'], cwd=directory,
                                capture_output=True, text=True)
        report['jobs'].append(dict(family=family, directory=str(directory),
            command=['condor_submit', '-spool', 'job.sub'], returncode=result.returncode,
            stdout=result.stdout, stderr=result.stderr, submitted_unix=time.time(),
            worker_sha256=sha(directory / 'worker.sh'), submit_sha256=sha(directory / 'job.sub')))
        output.write_text(json.dumps(report, indent=2) + '\n')
        assert result.returncode == 0, result.stderr
    report['status'] = 'submitted_pending_measurements'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
