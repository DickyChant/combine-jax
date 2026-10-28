"""Wait for an idle T4, then launch one fresh frozen campaign (no retry loop)."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20',
       '-o', 'HostName=lxplus905.cern.ch', '-o', 'HostKeyAlias=lxplus.cern.ch',
       'sqian@lxplus.cern.ch']
BASE = '/tmp/sqian/combine-jax'
PREFIX = 'gpu_tensor_native_policy_t4_retry1_'
DIRECTORY = BASE + '/build/native_policy_t4_retry1'

PROBE = '''import json, subprocess
row=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert len(row)==2 and 'T4' in row[0]
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
print(json.dumps(dict(device=row[0].strip(),free_mib=float(row[1]),compute_process_count=len(pids))))
'''

STAGE = '''import hashlib,json,subprocess
from pathlib import Path
assert Path.cwd()==Path('/tmp/sqian/combine-jax')
base=Path('build/native_policy_stage/assembled')
report=json.loads((base/'retry1.json').read_text())
assert report['status']=='submitted_pending_measurements'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
 return h.hexdigest()
payload=base/'payload.tar.gz';patch=base/'runtime_patch_retry1.tar'
assert sha(payload)==report['payload_sha256'] and sha(patch)==report['patch_sha256']
work=Path('build/native_policy_t4_retry1');work.mkdir()
subprocess.run(['tar','xzf',str(payload),'-C',str(work)],check=True)
subprocess.run(['tar','xf',str(patch),'-C',str(work)],check=True)
runtime=work/'build/edm_runtime_manifest.json'
assert sha(runtime)==report['runtime_manifest_sha256']
data=json.loads(runtime.read_text())
for name,digest in {**data['sources'],**data['input_files']}.items():assert sha(work/name)==digest,name
assert (work/'reports').is_dir()
print(json.dumps(dict(status='fresh_runtime_verified',directory=str(work.resolve()),source_revision=data['source_revision'],runtime_manifest_sha256=sha(runtime))))
'''


def remote_python(script, timeout):
    result = subprocess.run([*SSH, 'cd ' + shlex.quote(BASE) + ' && python3 -'],
        input=script, capture_output=True, text=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr[-4000:])
    return json.loads(result.stdout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-wait-seconds', type=float, default=21600)
    args = parser.parse_args()
    assert not args.output.exists() and args.max_wait_seconds > 0
    report = dict(status='waiting_for_idle_device', started_unix=time.time(), pid=os.getpid(),
        script_sha256=sha(__file__), source_revision='fea6bf8283a0dc2d2732983762829af981571983',
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()[19]),
        probes=0, requirement='No GPU compute processes and at least 12 GiB free. No other process is stopped.',
        scope='One new T4 campaign after memory becomes available. No restart after SSH, staging, or numerical failure.')
    save_json(args.output, report)
    try:
        while time.time() - report['started_unix'] < args.max_wait_seconds:
            observed = remote_python(PROBE, 45)
            report.update(probes=report['probes'] + 1, last_probe=observed, last_probe_unix=time.time())
            save_json(args.output, report)
            if observed['compute_process_count'] == 0 and observed['free_mib'] >= 12288:
                break
            time.sleep(60)
        else:
            report['status'] = 'wait_limit_reached'
            return
        report['status'] = 'staging'; save_json(args.output, report)
        report['runtime'] = remote_python(STAGE, 300)
        assert report['runtime']['directory'] == DIRECTORY
        helper = Path(__file__).with_name('run_remote_tensor_fit_campaign.py')
        output = Path('reports') / (PREFIX + 'ssh_driver.json')
        log = Path('reports') / (PREFIX + 'local_controller.log')
        assert not output.exists() and not log.exists()
        command = [sys.executable, str(helper), '--prefix', PREFIX, '--output', str(output),
                   '--remote-directory', DIRECTORY, '--campaign', 'edm']
        with log.open('x') as stream:
            child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=stream,
                stderr=subprocess.STDOUT, start_new_session=True)
        report.update(status='launched_once', command=command, controller_pid=child.pid,
            controller_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')', 1)[1].split()[19]),
            controller_script_sha256=sha(helper), controller_output=str(output), controller_log=str(log))
    except BaseException as exc:
        report.update(status='failed_no_retry', error=repr(exc))
        raise
    finally:
        report['updated_unix'] = time.time()
        save_json(args.output, report)


if __name__ == '__main__':
    main()
