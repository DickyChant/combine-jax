"""Start one isolated local campaign and retain its exact controller identity."""
import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--prefix',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    assert re.fullmatch(r'gpu_tensor_[a-z0-9_]+_',args.prefix)
    directory=args.directory.resolve();assert directory.is_relative_to(Path.cwd())
    runtime_path=directory/'build/edm_runtime_manifest.json';runtime=json.loads(runtime_path.read_text())
    assert runtime['policy']=='combine_v11_default'
    assert all(sha(directory/p)==h for p,h in {**runtime['sources'],**runtime['input_files']}.items())
    driver=directory/'reports'/(args.prefix+'driver.json');log=directory/'reports'/(args.prefix+'controller.log')
    assert not driver.exists() and not log.exists()
    environment=dict(os.environ,PYTHONPATH=str(directory/'src')+':'+str(directory/'scripts'))
    check=subprocess.check_output([str(directory/'.venv/bin/python'),'-c',
        'import combine_jax;print(combine_jax.__file__)'],cwd=directory,env=environment,text=True).strip()
    assert Path(check).is_relative_to(directory/'src'),check
    shell='source .venv/bin/activate\nsource scripts/env.sh\nexport PYTHONPATH="$PWD/src:$PWD/scripts"\nexec '+shlex.join([
        str(directory/'.venv/bin/python'),'scripts/run_tensor_edm_campaign.py','--runtime','local','--prefix',args.prefix])
    command=['bash','-c',shell]
    with log.open('x') as stream:
        child=subprocess.Popen(command,cwd=directory,stdin=subprocess.DEVNULL,stdout=stream,
            stderr=subprocess.STDOUT,start_new_session=True)
    stat=Path(f'/proc/{child.pid}/stat').read_text().rsplit(')',1)[1].split()
    save_json(args.output,dict(status='launched',controller_pid=child.pid,start_ticks=int(stat[19]),
        started_unix=time.time(),workdir=str(directory),command=command,script_sha256=sha(__file__),
        source_revision=runtime['source_revision'],runtime_manifest_sha256=sha(runtime_path),
        source_import_path=check,driver=str(driver),controller_log=str(log),
        scope='Sequential local CPU native/tensor campaign with explicit native fit policy, exact source/input hashes and one timed fit. No strict-gradient HHVV fit; no convergence or speed claim at launch.'))
    print('launched',child.pid,int(stat[19]),flush=True)


if __name__=='__main__':main()
