"""Retry a native timing after a recorded sequential campaign has completed."""
import argparse
import json
from pathlib import Path
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after-driver', type=Path, required=True)
    parser.add_argument('--case', required=True)
    parser.add_argument('--repeat', type=int, default=26)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.repeat > 0
    record_path = args.output.with_name(args.output.stem+'_retry_driver.json')
    log = args.output.with_suffix('.log')
    assert not any(p.exists() for p in (args.output, args.output.with_suffix('.npz'), record_path, log))
    driver = json.loads(args.after_driver.read_text())
    assert driver['status'] in ('completed_pending_audit', 'partial_failures')
    assert all(row['status'] in ('passed', 'failed') for row in driver['commands'])
    assert all(row['finished_unix'] <= driver['finished_unix'] for row in driver['commands'])
    command = ['bash', 'scripts/run_tensor_native.sh', 'scripts/benchmark_native_tensor.py',
               '--case', args.case, '--repeat', str(args.repeat), '--output', str(args.output)]
    started = time.time()
    assert driver['finished_unix'] <= started
    record = dict(status='running', script_sha256=sha(__file__), command=command,
        started_unix=started, preceding_driver=str(args.after_driver),
        preceding_driver_sha256=sha(args.after_driver), preceding_finished_unix=driver['finished_unix'],
        log=str(log), sources={p: sha(p) for p in command[1:3]},
        scope='Native retry starts only after every preceding campaign command has finished. '
              'Original failed captures remain untouched. Numerical/native-stack checks are '
              'performed by the unchanged native benchmark in its separate v11/ROOT 6.40 process.')
    save_json(record_path, record)
    with log.open('x') as stream:
        completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record.update(status='completed_pending_audit' if completed.returncode == 0 else 'failed',
                  returncode=completed.returncode, finished_unix=time.time(), log_sha256=sha(log))
    save_json(record_path, record)
    print(record['status'], flush=True)
    raise SystemExit(completed.returncode)


if __name__ == '__main__':
    main()
