"""Stop an identified native shard after a saved prefix and seal that prefix."""
import argparse
import json
import os
from pathlib import Path
import signal
import time

from tensor_benchmark_common import save_json, sha


def worker_identity(controller, proc_root=Path('/proc')):
    process = proc_root/str(controller['pid'])
    try:
        fields = (process/'stat').read_text().rsplit(')', 1)[1].split()
        command = (process/'cmdline').read_bytes().split(b'\0')
        directory = (process/'cwd').resolve()
    except (FileNotFoundError, ProcessLookupError):
        return False
    if fields[0] == 'Z':
        return False
    assert int(fields[19]) == controller['start_ticks'], 'Worker PID was reused'
    assert b'scripts/capture_asimov_native_fd_shard.py' in command, 'Wrong worker script'
    assert directory == Path(controller['workdir']), 'Wrong worker directory'
    return True


def validate_prefix(report, indices):
    assert report['dataset'] == 'asimov_prefit'
    assert report['status'] == 'running'
    assert report['script_sha256'] == sha('scripts/capture_asimov_native_fd_shard.py')
    assert indices and indices == sorted(set(indices))
    assert report['selected_indices'][:len(indices)] == indices
    received = [row['index'] for row in report['points']]
    assert received == indices[:len(received)], 'Worker already saved outside the owned prefix'
    return received == indices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('controller', 'source-report', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--indices', nargs='+', type=int, required=True)
    args = parser.parse_args()
    base = Path('/tmp/sqian/combine-jax')
    for path in (args.controller, args.source_report, args.output):
        assert path.is_absolute() and path.is_relative_to(base) and '..' not in path.parts
    assert not args.output.exists()
    controller = json.loads(args.controller.read_text())
    assert worker_identity(controller), 'Worker is not live'
    report = dict(status='waiting_for_saved_prefix', script_sha256=sha(__file__),
        dataset='asimov_prefit', owned_indices=args.indices, source_controller=str(args.controller),
        worker={k:controller[k] for k in ('pid','start_ticks','workdir')}, started_unix=time.time())
    save_json(args.output, report)
    while True:
        source = json.loads(args.source_report.read_text())
        if validate_prefix(source, args.indices):
            break
        assert worker_identity(controller), 'Worker exited before saving the prefix'
        assert time.time()-report['started_unix'] < 7200, 'Two-hour checkpoint wait exceeded'
        time.sleep(2)
    assert worker_identity(controller)
    os.kill(controller['pid'], signal.SIGTERM)
    deadline = time.monotonic()+30
    while worker_identity(controller):
        assert time.monotonic() < deadline, 'Worker has not stopped; no sealed checkpoint created'
        time.sleep(.2)
    raw = args.source_report.read_bytes()
    source = json.loads(raw)
    assert validate_prefix(source, args.indices)
    # The original runtime report is unmodified. Preserve its exact bytes.
    frozen = args.output.with_name(args.output.stem+'_source.json')
    assert not frozen.exists()
    frozen.write_bytes(raw)
    report.update(status='sealed_native_partition', source_report=str(frozen.relative_to(base)),
        source_report_sha256=sha(frozen), stopped_unix=time.time(),
        worker_verified_stopped=True, signal='SIGTERM',
        scope='Only the saved native prefix is owned by this checkpoint. The original report, planned indices and point artifacts are retained unchanged.')
    save_json(args.output, report)
    print('sealed', args.indices, flush=True)


if __name__ == '__main__':
    main()
