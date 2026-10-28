"""Launch one verified Asimov T4 campaign after GPU and native-FD work are idle."""
import argparse
import inspect
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from tensor_benchmark_common import save_json,sha

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20',
     '-o','HostName=lxplus905.cern.ch','-o','HostKeyAlias=lxplus.cern.ch','sqian@lxplus.cern.ch']


def native_controller_active(controller, proc_root):
    """Check the script and, when recorded, the process start identity.

    Earlier native controllers have no start ticks; conservatively wait on a
    matching reference process at that PID instead of assuming it has finished.
    """
    process=proc_root/str(controller['pid'])
    try:
        command=(process/'cmdline').read_bytes().split(b'\0')
        ticks=int((process/'stat').read_text().rsplit(')',1)[1].split()[19])
    except (FileNotFoundError,ProcessLookupError):
        return False
    scripts={b'capture_asimov_native_reference.py',b'probe_asimov_cached_deltas.py',b'capture_asimov_native_fd_shard.py'}
    matches=any(part.rsplit(b'/',1)[-1] in scripts for part in command)
    return matches and ticks==controller.get('start_ticks',ticks)


def validation_ready(path, model_hashes, parameter_count):
    """Require the complete, matching native gate before reserving T4 resources."""
    if path is None:
        return dict(required=False, ready=True)
    if not path.exists():
        return dict(required=True, ready=False, path=str(path))
    report=json.loads(path.read_text())
    assert report['dataset']=='asimov_prefit', 'Validation must use the pre-fit Asimov'
    assert report['model_sha256']==model_hashes, 'Validation belongs to a different model'
    if report['status']!='passed':
        return dict(required=True, ready=False, path=str(path), status=report['status'])
    points=report['points']
    assert len(points)==26 and {p['index'] for p in points}==set(range(26))
    assert report['nll_passed'] is True
    assert report['max_nll_absolute_error']<1e-6
    assert all(p['unresolved_or_failed']==[] for p in points)
    assert all(sum(p['status_counts'].values())==parameter_count for p in points)
    for counts in [report['status_counts'],*(p['status_counts'] for p in points)]:
        assert counts.get('passed',0)>0 and set(counts)<={'passed','fixed_bounds'}
    for key in report['status_counts']:
        assert report['status_counts'][key]==sum(p['status_counts'].get(key,0) for p in points)
    return dict(required=True, ready=True, path=str(path), sha256=sha(path), status='passed')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-directory',type=Path,required=True)
    parser.add_argument('--native-controller',type=Path,required=True)
    parser.add_argument('--additional-native-controller',type=Path,action='append',default=[])
    parser.add_argument('--native-validation',type=Path,
                        help='Local full 26-point native audit for this exact Asimov model')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-wait-seconds',type=float,default=86400)
    args=parser.parse_args()
    base=Path('/tmp/sqian/combine-jax')
    controllers=[args.native_controller,*args.additional_native_controller]
    for path in (args.remote_directory,*controllers):
        assert path.is_absolute() and '..' not in path.parts and path.is_relative_to(base)
    assert not args.output.exists()
    config=dict(directory=str(args.remote_directory),native_controllers=list(map(str,controllers)),
                native_validation=str(args.native_validation) if args.native_validation else None)
    probe=('config='+repr(config)+'\nimport hashlib, json, pathlib, subprocess\n'
           +inspect.getsource(native_controller_active)+'''
base=pathlib.Path(config['directory'])
manifest=json.loads((base/'build/tensor_benchmark_inputs/manifest.json').read_text())
case=manifest['hhvv'];assert case['dataset']=='asimov_prefit'
model_bytes=(base/case['model']/'model.json').read_bytes()
document=json.loads(model_bytes)
model=document['metadata']
model_hashes={'model.json':hashlib.sha256(model_bytes).hexdigest(),'arrays.npz':document['arrays_sha256']}
assert model['provenance']['dataset']['name']=='asimov_prefit'
assert model['provenance']['dataset_generation']['kind']=='combine_prefit_asimov'
row=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.free','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert len(row)==2 and 'T4' in row[0]
pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).split()
states=[dict(controller=p,active=native_controller_active(json.loads(pathlib.Path(p).read_text()),pathlib.Path('/proc')))
        for p in config['native_controllers']]
print(json.dumps(dict(device=row[0].strip(),free_mib=float(row[1]),compute_process_count=len(pids),native_reference_active=any(s['active'] for s in states),native_controllers=states,dataset='asimov_prefit',model_sha256=model_hashes,parameter_count=len(model['parameter_names']))))
''')
    report=dict(status='waiting_for_idle_resources',started_unix=time.time(),pid=os.getpid(),
        start_ticks=int(Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()[19]),
        script_sha256=sha(__file__),dataset='asimov_prefit',config=config,probes=0,
        requirement='No GPU compute processes, at least 12 GiB free, all named native reference processes finished, and the requested full native validation passed. No other process is stopped.',
        scope='One Asimov campaign; no automatic retry after launch or failure.')
    save_json(args.output,report)
    try:
        while time.time()-report['started_unix']<args.max_wait_seconds:
            result=subprocess.run([*SSH,'python3 -'],input=probe,capture_output=True,text=True,timeout=45)
            assert result.returncode==0,result.stderr[-2000:]
            observed=json.loads(result.stdout)
            observed['native_validation']=validation_ready(args.native_validation,observed['model_sha256'],observed['parameter_count'])
            report.update(probes=report['probes']+1,last_probe=observed,last_probe_unix=time.time())
            save_json(args.output,report)
            if (observed['compute_process_count']==0 and observed['free_mib']>=12288
                    and not observed['native_reference_active'] and observed['native_validation']['ready']):
                break
            time.sleep(60)
        else:
            report['status']='wait_limit_reached';return
        helper=Path(__file__).with_name('run_remote_tensor_fit_campaign.py')
        output=Path('reports/gpu_tensor_asimov_t4_ssh_driver.json')
        log=Path('reports/gpu_tensor_asimov_t4_local_controller.log')
        assert not output.exists() and not log.exists()
        command=[sys.executable,str(helper),'--prefix','gpu_tensor_asimov_t4_',
            '--output',str(output),'--remote-directory',str(args.remote_directory),'--campaign','edm']
        with log.open('x') as stream:
            child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        report.update(status='launched_once',command=command,controller_pid=child.pid,
            controller_start_ticks=int(Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()[19]),
            controller_script_sha256=sha(helper),controller_output=str(output),controller_log=str(log))
    except BaseException as exc:
        report.update(status='failed_no_retry',error=repr(exc));raise
    finally:
        report['updated_unix']=time.time();save_json(args.output,report)


if __name__=='__main__':main()
