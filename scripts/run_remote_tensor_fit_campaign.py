"""Hold a foreground lxplus905 SSH session for the sequential fit campaign.

Run this small controller detached on the local host. No automatic retry follows
an SSH failure: inspect the remote controller's exact identity and saved state.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import time

from tensor_benchmark_common import save_json, sha


def identity(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=int(fields[19]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--remote-directory', type=Path, default=Path('/tmp/sqian/combine-jax'))
    parser.add_argument('--campaign', choices=('fits', 'edm'), default='fits')
    parser.add_argument('--backends', nargs='+', choices=('cpu','gpu'),
                        help='Backend selection for the EDM campaign only')
    args = parser.parse_args()
    if args.backends is not None:
        if args.campaign != 'edm':parser.error('--backends requires --campaign edm')
        from run_tensor_edm_campaign import selected_backends
        try:selected_backends('lcg',args.backends)
        except ValueError as exc:parser.error(str(exc))
    assert re.fullmatch(r'gpu_tensor_[a-z0-9_]+_', args.prefix)
    log = args.output.with_suffix('.log')
    assert args.output.parent == Path('reports') and args.output.suffix == '.json'
    assert not args.output.exists() and not log.exists()
    directory = args.remote_directory
    assert directory.is_absolute() and '..' not in directory.parts
    assert directory.is_relative_to(Path('/tmp/sqian/combine-jax'))
    script = 'scripts/run_tensor_t4_fits.py' if args.campaign == 'fits' else 'scripts/run_tensor_edm_campaign.py'
    arguments = ['--gpu-class', 't4'] if args.campaign == 'edm' else []
    if args.backends is not None:arguments += ['--backends',*args.backends]
    remote = ('cd '+shlex.quote(str(directory))+' && exec '+shlex.join([
        'bash', 'scripts/run_tensor_lcg.sh', script, '--prefix', args.prefix, *arguments]))
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
               '-o', 'HostName=lxplus905.cern.ch', '-o', 'HostKeyAlias=lxplus.cern.ch',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=6',
               'sqian@lxplus.cern.ch', remote]
    report = dict(status='running', controller=identity(os.getpid()), script_sha256=sha(__file__),
                  started_unix=time.time(), command=command, log=str(log),
                  remote_driver=str(directory / 'reports' / (args.prefix+'driver.json')),
                  campaign=args.campaign,
                  requested_backends=args.backends,
                  scope='Foreground remote command held by a detached local SSH controller. '
                        'No numerical acceptance or automatic restart follows SSH exit status.')
    child = None

    def interrupted(signum, frame):
        raise InterruptedError(f'Local SSH controller received signal {signum}')

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    save_json(args.output, report)
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            report['ssh_child'] = identity(child.pid); save_json(args.output, report)
            code = child.wait()
        report.update(returncode=code, status='remote_finished_pending_audit' if code == 0 else 'ssh_or_remote_failed')
    except BaseException as exc:
        report.update(status='interrupted', error=repr(exc)); raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
            report['interrupted_ssh_returncode'] = child.returncode
        report.update(finished_unix=time.time(), log_sha256=sha(log))
        save_json(args.output, report)


if __name__ == '__main__':
    main()
