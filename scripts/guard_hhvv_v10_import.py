"""Watch the native importer from another process so ROOT cannot block the RSS guard."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    base = Path('build/hhvv_v10_campaign_20261024')
    output = base/'remote_import_guard.json'
    assert not output.exists()
    command = ['taskset', '-c', ','.join(map(str, sorted(os.sched_getaffinity(0))[:8])),
        '.venv-import/bin/python', 'scripts/import_hhvv_prefit_asimov.py',
        '--workspace', str(base/'generation/workspace_asimov.root'),
        '--generation-report', str(base/'generation_for_import.json'),
        '--card', str(base/'card.txt'), '--points', str(base/'points.npz'),
        '--directory', str(base/'tensor'), '--output', str(base/'import_remote.json')]
    report = dict(status='running', command=command, rss_limit_mib=11264,
                  peak_process_group_rss_mib=0., scope='Same 11 GiB RSS guard, sampled outside ROOT/Python GIL.')
    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(output)
    save()
    started = time.monotonic()
    child = None
    def interrupted(signum, frame):
        raise InterruptedError('Import monitor received signal '+str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        with (base/'import_remote.log').open('x') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                ps = subprocess.run(['ps', '--no-headers', '-g', str(child.pid), '-o', 'rss='],
                                    capture_output=True, text=True)
                assert ps.returncode in (0, 1)
                rss = sum(float(v) for v in ps.stdout.split())/1024
                report['peak_process_group_rss_mib'] = max(rss, report['peak_process_group_rss_mib'])
                report['elapsed_seconds'] = time.monotonic()-started
                save()
                if rss > 11264:
                    raise MemoryError('Importer exceeded the unchanged 11 GiB RSS guard')
                if report['elapsed_seconds'] > 7200:
                    raise TimeoutError('Importer exceeded two hours')
                time.sleep(.25)
            report.update(status='completed' if child.returncode == 0 else 'failed', returncode=child.returncode)
    except BaseException as exc:
        report.update(status='failed_or_guarded', error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
            report['returncode'] = child.returncode
        report['elapsed_seconds'] = time.monotonic()-started
        save()
    assert child.returncode == 0
    print(report, flush=True)


if __name__ == '__main__':
    main()
