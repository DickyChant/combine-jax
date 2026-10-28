"""Submit one verified HHVV mass-point payload and record its scheduler identity."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import socket
import subprocess

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname().split('.')[0] != 'lxplus905'
    output = args.directory/'submission.json'
    assert not output.exists(), 'Inspect the saved scheduler receipt; do not submit twice'
    staged = json.loads((args.directory/'staging.json').read_text())
    assert staged['status'] == 'staged' and staged['static_guard_gib'] == 8. and staged['float64']
    for name, checksum in staged['files'].items():
        assert sha(args.directory/name) == checksum
    dry = subprocess.run(['condor_submit', '-dry-run', 'dry_run.ad', 'job.sub'],
                         cwd=args.directory, capture_output=True, text=True)
    assert dry.returncode == 0, dry.stdout+dry.stderr
    report = dict(status='submitting', mass=staged['mass'],
        submitted_utc=datetime.now(timezone.utc).isoformat(), host=socket.gethostname(),
        script_sha256=sha(__file__), staging_sha256=sha(args.directory/'staging.json'),
        payload_sha256=staged['files']['payload.tar.gz'], static_guard_gib=8., float64=True,
        command=['condor_submit', '-spool', 'job.sub'],
        discrete_policy=staged['discrete_policy'], fixed_discrete_choices=staged['fixed_discrete_choices'])
    save_json(output, report)
    result = subprocess.run(report['command'], cwd=args.directory, capture_output=True, text=True)
    report.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    match = re.search(r'submitted to cluster (\d+)', result.stdout)
    if result.returncode or not match:
        report['status'] = 'submission_requires_inspection'; save_json(output, report)
        raise RuntimeError('Inspect submission receipt before retrying')
    job_id = match.group(1)+'.0'
    report.update(status='submitted', job_id=job_id)
    save_json(output, report)
    check = subprocess.run(['condor_q', job_id, '-json', '-attributes',
        'ClusterId,ProcId,JobStatus,RequestGPUs,RequestMemory,Requirements,Rank,JobFlavour,Owner'],
        capture_output=True, text=True, check=True)
    report['classads'] = json.loads(check.stdout or '[]')
    save_json(output, report)
    print('Submitted MX'+str(staged['mass']), job_id, staged['discrete_policy'], flush=True)


if __name__ == '__main__':
    main()
