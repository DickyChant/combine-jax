"""Freeze committed sources and immutable native-policy inputs for a local run."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile

from tensor_benchmark_common import save_json,sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--source-revision',default='HEAD')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=Path('build/tensor_native_policy_inputs/manifest.json'))
    args=parser.parse_args();assert not args.directory.exists() and not args.output.exists()
    root=Path.cwd();directory=args.directory.resolve();directory.mkdir()
    revision=subprocess.check_output(['git','rev-parse',args.source_revision],text=True).strip()
    manifest_path=args.manifest
    manifest=json.loads(manifest_path.read_text())
    sources={}
    archive=subprocess.Popen(['git','archive',revision,'src','scripts'],stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout,mode='r|') as stream:
        stream.extractall(directory,filter='data')
    assert archive.wait()==0
    for name in ('src','scripts'):
        for p in sorted((directory/name).rglob('*')):
            if p.is_file():sources[str(p.relative_to(directory))]=sha(p)
    inputs={}
    for case in manifest.values():
        candidates=dict(case['hashes'])
        candidates[case['workspace']]=case.get('workspace_sha256',sha(case['workspace']))
        for name,digest in candidates.items():
            source=(root/name).resolve();assert source.is_relative_to(root)
            assert sha(source)==digest,name
            target=directory/name;assert target.is_relative_to(directory)
            if not target.exists():
                target.parent.mkdir(parents=True,exist_ok=True)
                os.link(source,target)
            assert sha(target)==digest
            inputs[name]=digest
    # Sources are copied; read-only model/input artifacts are hard-linked to
    # avoid duplicating hundreds of MB locally. Jobs never mutate those inputs.
    (directory/'.venv').symlink_to(root/'.venv',target_is_directory=True)
    (directory/'reports').mkdir(exist_ok=True)
    target_manifest=directory/'build/tensor_benchmark_inputs/manifest.json'
    target_manifest.parent.mkdir(parents=True,exist_ok=True)
    save_json(target_manifest,manifest)
    runtime=directory/'build/edm_runtime_manifest.json'
    save_json(runtime,dict(source_revision=revision,sources=sources,input_files=inputs,
        manifest_sha256=sha(target_manifest),policy='combine_v11_default'))
    save_json(args.output,dict(status='staged',source_revision=revision,directory=str(directory),
        script_sha256=sha(__file__),source_manifest_sha256=sha(manifest_path),
        runtime_manifest_sha256=sha(runtime),sources=len(sources),input_files=len(inputs),
        scope='Committed source snapshot, separate reports/build outputs, project venv, immutable input hard links. No numerical job has started.'))
    print('staged',directory,revision,len(inputs),flush=True)


if __name__=='__main__':main()
