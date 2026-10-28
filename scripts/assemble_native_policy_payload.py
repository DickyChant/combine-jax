"""Assemble a checked Condor payload on lxplus905 without another ROOT stack."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import socket
import tarfile
import time


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def checked_members(archive):
    for member in archive:
        path=PurePosixPath(member.name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Unsafe archive member: '+member.name)
        yield member


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base',type=Path,required=True)
    parser.add_argument('--base-sha256',required=True)
    parser.add_argument('--overlay',type=Path,required=True)
    parser.add_argument('--overlay-report',type=Path,required=True)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    assert Path.cwd()==Path('/tmp/sqian/combine-jax') and socket.gethostname().split('.')[0]=='lxplus905'
    assert not args.directory.exists()
    evidence=json.loads(args.overlay_report.read_text())
    assert sha(args.base)==args.base_sha256 and sha(args.overlay)==evidence['overlay_sha256']
    args.directory.mkdir();work=args.directory/'work';work.mkdir()
    shared=set(evidence['shared_base_inputs'])
    with tarfile.open(args.base) as archive:
        selected=[m for m in checked_members(archive) if str(PurePosixPath(m.name)) in shared
            or str(PurePosixPath(m.name)).split('/')[0]=='.venv-combine-v11']
        archive.extractall(work,members=selected)
    with tarfile.open(args.overlay) as archive:
        archive.extractall(work,members=list(checked_members(archive)))
    runtime_path=work/'build/edm_runtime_manifest.json'
    assert sha(runtime_path)==evidence['runtime_manifest_sha256']
    runtime=json.loads(runtime_path.read_text())
    for name,digest in {**runtime['sources'],**runtime['input_files']}.items():assert sha(work/name)==digest,name
    assert sha(work/'build/tensor_benchmark_inputs/manifest.json')==runtime['manifest_sha256']
    assert (work/'.venv-combine-v11/bin/python').exists()
    payload=args.directory/'payload.tar.gz'
    files=set(runtime['sources'])|set(runtime['input_files'])|{
        'build/edm_runtime_manifest.json','build/tensor_benchmark_inputs/manifest.json'}
    with tarfile.open(payload,'w:gz',compresslevel=1) as archive:
        archive.add(work/'.venv-combine-v11',arcname='.venv-combine-v11')
        for name in sorted(files):archive.add(work/name,arcname=name,recursive=False)
    # File-only source/input archives omit empty output directories. Include
    # this directory explicitly in the patch extracted before the controller.
    patch=args.directory/'runtime_patch.tar'
    with tarfile.open(patch,'w') as archive:
        member=tarfile.TarInfo('reports');member.type=tarfile.DIRTYPE;member.mode=0o755
        archive.addfile(member)
    report=dict(status='assembled_and_verified',finished_unix=time.time(),host=socket.gethostname(),
        script_sha256=sha(__file__),base_sha256=args.base_sha256,overlay_sha256=sha(args.overlay),
        source_revision=runtime['source_revision'],runtime_manifest_sha256=sha(runtime_path),
        payload=str(payload),payload_sha256=sha(payload),payload_bytes=payload.stat().st_size,
        patch=str(patch),patch_sha256=sha(patch),input_files=len(runtime['input_files']),
        scope='Only the existing native v11 environment and unchanged HHVV shared inputs come from the old payload. All sources, model/reference files and canonical manifest match the new frozen runtime. LCG CUDA Python and native ROOT remain separate processes.')
    (args.directory/'assembly.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
