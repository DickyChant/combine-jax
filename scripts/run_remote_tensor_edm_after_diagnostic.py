"""Hold SSH and start the isolated T4 EDM campaign after the old diagnostic exits."""
import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tarfile
import time

from tensor_benchmark_common import save_json, sha
from run_remote_tensor_fit_campaign import identity


def worker(check_once):
    root=Path('/tmp/sqian/combine-jax')
    assert Path.cwd()==root/'build/edm_t4_run/work'
    output=Path('reports/gpu_tensor_edm_t4_dependency.json')
    old_path=root/'reports/gpu_tensor_t4_fit_comparison_retry2_driver.json'
    old=json.loads(old_path.read_text())
    expected=dict(pid=old['controller_pid'],start_ticks=old['controller_start_ticks'])
    def live(item):
        try:return identity(item['pid'])==item
        except FileNotFoundError:return False
    report=dict(status='waiting',previous_controller=expected,script_sha256=sha(__file__),started_unix=time.time())
    try:
        while live(expected):
            report['last_check_unix']=time.time();save_json(output,report)
            if check_once:
                print('verified waiting for previous controller',expected,flush=True)
                return
            time.sleep(20)
        old=json.loads(old_path.read_text());assert old['status']!='running'
        assert all(not live(dict(pid=r['pid'],start_ticks=r['start_ticks'])) for r in old['commands'])
        assert all(sha(root/p)==digest for p,digest in old['sources'].items())
        manifest=root/old['runtime_manifest'];runtime=json.loads(manifest.read_text())
        assert sha(manifest)==old['runtime_manifest_sha256']
        assert all(sha(root/p)==digest for p,digest in runtime['inputs'].items())
        if check_once:
            report.update(status='ready',finished_unix=time.time());save_json(output,report)
            print('verified previous diagnostic finished',flush=True)
            return
        files=list((root/'reports').glob('gpu_tensor_t4_fit_comparison_retry2_*'))
        files=[p for p in files if p.is_file()]
        archive=Path('../previous_diagnostic_capture.tar.gz')
        assert not archive.exists()
        with tarfile.open(archive,'w:gz') as capture:
            for p in files:capture.add(p,arcname=str(p.relative_to(root)))
            capture.add(manifest,arcname='build/previous_runtime_manifest.json')
        report.update(status='previous_diagnostic_finished_sources_verified',previous_status=old['status'],
            previous_driver_sha256=sha(old_path),previous_archive=str(archive),previous_archive_sha256=sha(archive),
            previous_artifacts={str(p.relative_to(root)):sha(p) for p in files},finished_unix=time.time())
        save_json(output,report)
        os.execvp('bash',['bash','scripts/run_tensor_lcg.sh','scripts/run_tensor_edm_campaign.py',
                         '--gpu-class','t4','--prefix','gpu_tensor_edm_t4_'])
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),finished_unix=time.time());save_json(output,report);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--check-once',action='store_true')
    args=parser.parse_args()
    if args.worker:return worker(args.check_once)
    assert not args.check_once
    output=Path('reports/gpu_tensor_edm_t4_ssh_driver.json');log=output.with_suffix('.log')
    assert not output.exists() and not log.exists()
    remote='cd /tmp/sqian/combine-jax/build/edm_t4_run/work && exec '+shlex.join([
        'bash','scripts/run_tensor_lcg.sh','scripts/run_remote_tensor_edm_after_diagnostic.py','--worker'])
    command=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','HostName=lxplus905.cern.ch',
        '-o','HostKeyAlias=lxplus.cern.ch','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=6',
        'sqian@lxplus.cern.ch',remote]
    report=dict(status='running',controller=identity(os.getpid()),script_sha256=sha(__file__),
                command=command,started_unix=time.time(),log=str(log),scope='Foreground remote session held locally; no automatic retry after SSH failure. The remote controller waits for the earlier diagnostic and preserves its artifacts before starting the accepted EDM campaign.')
    child=None
    def interrupted(signum,frame):raise InterruptedError(f'Signal {signum}')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    save_json(output,report)
    try:
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            report['ssh_child']=identity(child.pid);save_json(output,report);code=child.wait()
        report.update(returncode=code,status='remote_finished_pending_audit' if code==0 else 'ssh_or_remote_failed')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        report.update(finished_unix=time.time(),log_sha256=sha(log));save_json(output,report)


if __name__=='__main__':main()
