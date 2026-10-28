"""Submit the verified Asimov payload to separate A100 and H100 jobs."""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import time

from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assembly',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--families',nargs='+',choices=('A100','H100'),default=['A100','H100'])
    parser.add_argument('--hhvv-fit-order',choices=('native-first','tensor-first'),default='native-first')
    parser.add_argument('--gpu-hessian-batch-size',type=int)
    args=parser.parse_args()
    if len(set(args.families))!=len(args.families):parser.error('Select each GPU family at most once')
    if args.gpu_hessian_batch_size is not None and args.gpu_hessian_batch_size<1:
        parser.error('gpu-hessian-batch-size must be positive')
    assert Path.cwd()==Path('/tmp/sqian/combine-jax') and socket.gethostname().split('.')[0]=='lxplus905'
    assert not args.directory.exists() and not args.output.exists()
    assembly=json.loads(args.assembly.read_text())
    assert assembly['status']=='assembled_asimov_pending_numerical_validation'
    payload,patch=(Path(assembly[k]) for k in ('payload','patch'))
    assert sha(payload)==assembly['payload_sha256'] and sha(patch)==assembly['patch_sha256']
    work=payload.parent/'work'
    manifest=json.loads((work/'build/tensor_benchmark_inputs/manifest.json').read_text())
    assert manifest['hhvv']['dataset']=='asimov_prefit'
    args.directory.mkdir()
    report=dict(status='submitting_asimov',script_sha256=sha(__file__),
        assembly_sha256=sha(args.assembly),source_revision=assembly['source_revision'],
        payload_sha256=sha(payload),dataset='asimov_prefit',
        dataset_identity=assembly['dataset_identity'],jobs=[],
        families=args.families,hhvv_fit_order=args.hhvv_fit_order,
        gpu_hessian_batch_size=args.gpu_hessian_batch_size,patch_sha256=sha(patch),
        validation='Measurements remain provisional pending native NLL/gradient and fit agreement gates.')
    save_json(args.output,report)
    for family in args.families:
        directory=args.directory/family.lower();directory.mkdir()
        os.link(payload,directory/'payload.tar.gz');os.link(patch,directory/'runtime_patch.tar')
        worker=directory/'worker.sh'
        worker_source=(work/'scripts/condor_tensor_edm.sh').read_text()
        invocation='bash scripts/run_tensor_lcg.sh scripts/run_tensor_edm_campaign.py'
        assert worker_source.splitlines().count(invocation)==1
        options=' --hhvv-fit-order '+args.hhvv_fit_order
        if args.gpu_hessian_batch_size is not None:options+=' --gpu-hessian-batch-size '+str(args.gpu_hessian_batch_size)
        worker.write_text(worker_source.replace(invocation+'\n',invocation+options+'\n'))
        worker.chmod(0o755)
        submit=(work/'scripts/condor_tensor_edm.sub').read_text()
        submit=submit.replace('regexp("A100|H100", GPUs_DeviceName)',f'regexp("{family}", GPUs_DeviceName)')
        (directory/'job.sub').write_text(submit)
        result=subprocess.run(['condor_submit','-spool','job.sub'],cwd=directory,capture_output=True,text=True)
        row=dict(family=family,directory=str(directory),submitted_unix=time.time(),
            command=['condor_submit','-spool','job.sub'],returncode=result.returncode,
            stdout=result.stdout,stderr=result.stderr,worker_sha256=sha(worker),submit_sha256=sha(directory/'job.sub'))
        report['jobs'].append(row);save_json(args.output,report)
        assert result.returncode==0,result.stderr
        match=re.search(r'cluster (\d+)',result.stdout);assert match,result.stdout
        row['job_id']=match[1]+'.0';save_json(args.output,report)
    report['status']='submitted_asimov_pending_measurements';save_json(args.output,report)
    print(report['status'],[j['job_id'] for j in report['jobs']],flush=True)


if __name__=='__main__':main()
