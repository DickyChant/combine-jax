"""Run a mass-point neutral import with an independent 11 GiB RSS guard."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--card', required=True)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-discrete', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    command = ['.venv-import/bin/python', 'scripts/import_hhvv_v10_limit_asimov.py',
        '--generation', args.generation, '--card', args.card, '--directory', args.directory]
    if args.fixed_discrete:
        command.append('--fixed-discrete')
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve())+':'+str(Path('scripts').resolve()),
        PYTHONDONTWRITEBYTECODE='1', JAX_PLATFORMS='cpu', JAX_ENABLE_X64='true',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
        XLA_FLAGS='--xla_cpu_multi_thread_eigen=false')
    report = dict(status='running', command=command, rss_limit_mib=11264,
        peak_process_group_rss_mib=0., generation=args.generation,
        scope='Independent process-group RSS sampling, outside ROOT/Python GIL; no fit campaign.')
    def save():
        tmp = args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2)+'\n'); tmp.replace(args.output)
    def interrupted(signum, frame):
        raise InterruptedError('Import guard received signal '+str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    started = time.monotonic(); child = None
    save()
    try:
        with args.output.with_suffix('.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while child.poll() is None:
                ps = subprocess.run(['ps', '--no-headers', '-g', str(child.pid), '-o', 'rss='], capture_output=True, text=True)
                assert ps.returncode in (0, 1)
                rss = sum(float(x) for x in ps.stdout.split())/1024
                report['peak_process_group_rss_mib'] = max(rss, report['peak_process_group_rss_mib'])
                report['elapsed_seconds'] = time.monotonic()-started
                save()
                if rss > 11264:
                    raise MemoryError('Import exceeds the unchanged 11 GiB RSS guard')
                if report['elapsed_seconds'] > 7200:
                    raise TimeoutError('Import exceeded two hours')
                time.sleep(.25)
        report.update(status='completed' if child.returncode == 0 else 'failed', returncode=child.returncode)
    except BaseException as exc:
        report.update(status='failed_or_guarded', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        report.update(elapsed_seconds=time.monotonic()-started,
            returncode=None if child is None else child.returncode)
        save()
    assert child.returncode == 0
    print(report['status'], report['elapsed_seconds'], report['peak_process_group_rss_mib'], flush=True)


if __name__ == '__main__':
    main()
