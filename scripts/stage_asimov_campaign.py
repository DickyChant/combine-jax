"""Build an Asimov-only HHVV campaign from verified inputs on lxplus905."""
import argparse
import json
import os
from pathlib import Path
import socket
import tarfile
import time
from types import SimpleNamespace

from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import benchmark_dataset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('source-archive','import-runtime','native-reference-runtime','generic-runtime','directory','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--revision',required=True)
    parser.add_argument('--reuse-payload',type=Path,
                        help='Reuse verified identical inputs/native environment, replacing committed sources in the patch')
    parser.add_argument('--reuse-payload-sha256')
    args=parser.parse_args()
    root=Path('/tmp/sqian/combine-jax')
    assert Path.cwd()==root and socket.gethostname().split('.')[0]=='lxplus905'
    assert not args.directory.exists() and not args.output.exists()
    imported=json.loads((args.import_runtime/'reports/tensor_hhvv_asimov_shared_import.json').read_text())
    assert imported['status']=='imported_identity_verified_numerical_gates_pending'
    reference=json.loads((args.native_reference_runtime/'reports/gpu_hhvv_asimov_valid_native_reference.json').read_text())
    assert reference['dataset']=='asimov_prefit' and reference['nll_points']==26
    assert reference['workspace_sha256']==imported['workspace_sha256']
    native_values=args.native_reference_runtime/reference['nll_arrays']
    assert sha(native_values)==reference['nll_arrays_sha256']
    args.directory.mkdir();work=args.directory/'work';work.mkdir()
    with tarfile.open(args.source_archive) as archive:
        archive.extractall(work,filter='data')
    sources={str(p.relative_to(work)):sha(p) for folder in ('src','scripts')
             for p in sorted((work/folder).rglob('*')) if p.is_file()}
    manifest=json.loads((args.generic_runtime/'build/tensor_benchmark_inputs/manifest.json').read_text())
    manifest.pop('hhvv')
    assert set(manifest)=={'counting','automcstats','parametric','multipdf_exp','multipdf_bern'}
    inputs={}
    def link(source,name,digest):
        target=work/name
        assert target.resolve().is_relative_to(work.resolve()) and sha(source)==digest
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():os.link(source,target)
        assert sha(target)==digest
        inputs[name]=digest
    for case in manifest.values():
        for name,digest in case['hashes'].items():link(args.generic_runtime/name,name,digest)
    hh=json.loads((args.import_runtime/imported['manifest']).read_text())['hhvv']
    for name,digest in hh['hashes'].items():link(args.import_runtime/name,name,digest)
    link(native_values,hh['reference'],reference['nll_arrays_sha256'])
    hh['hashes'][hh['reference']]=reference['nll_arrays_sha256']
    hh['reference_validation']=dict(native_nll='26 points captured; comparison required',
        native_gradients='pending',reference_kind='native_nll_only',
        scope='No observed-data or fitted-point references. Gradient captures require the independent native FD audit before any speed claim.')
    manifest['hhvv']=hh
    previous=Path.cwd()
    os.chdir(work)
    try:
        doc=json.loads((Path(hh['model'])/'model.json').read_text())
        benchmark_dataset(hh,SimpleNamespace(metadata=doc['metadata']),label='hhvv')
    finally:os.chdir(previous)
    (work/'reports').mkdir(exist_ok=True)
    manifest_path=work/'build/tensor_benchmark_inputs/manifest.json'
    manifest_path.parent.mkdir(parents=True,exist_ok=True);save_json(manifest_path,manifest)
    runtime_path=work/'build/edm_runtime_manifest.json'
    save_json(runtime_path,dict(source_revision=args.revision,sources=sources,input_files=inputs,
        manifest_sha256=sha(manifest_path),policy='combine_v11_default',dataset='asimov_prefit'))
    native_env=(args.generic_runtime/'.venv-combine-v11').resolve()
    assert (native_env/'bin/python').exists()
    (work/'.venv-combine-v11').symlink_to(native_env,target_is_directory=True)
    payload=args.directory/'payload.tar.gz'
    runtime_files={'build/edm_runtime_manifest.json','build/tensor_benchmark_inputs/manifest.json'}
    if args.reuse_payload:
        assert sha(args.reuse_payload)==args.reuse_payload_sha256
        with tarfile.open(args.reuse_payload) as archive:
            prior=json.load(archive.extractfile('build/edm_runtime_manifest.json'))
        assert prior['input_files']==inputs, 'Only identical immutable inputs may reuse a payload'
        os.link(args.reuse_payload,payload)
    else:
        with tarfile.open(payload,'w:gz',compresslevel=1) as archive:
            archive.add(native_env,arcname='.venv-combine-v11')
            for name in sorted(set(sources)|set(inputs)|runtime_files):
                archive.add(work/name,arcname=name,recursive=False)
            member=tarfile.TarInfo('reports');member.type=tarfile.DIRTYPE;member.mode=0o755
            archive.addfile(member)
    patch=args.directory/'runtime_patch.tar'
    with tarfile.open(patch,'w') as archive:
        if args.reuse_payload:
            for name in sorted(set(sources)|runtime_files):
                archive.add(work/name,arcname=name,recursive=False)
    report=dict(status='assembled_asimov_pending_numerical_validation',finished_unix=time.time(),
        source_revision=args.revision,source_archive_sha256=sha(args.source_archive),
        script_sha256=sha(__file__),import_report_sha256=sha(args.import_runtime/'reports/tensor_hhvv_asimov_shared_import.json'),
        native_nll_sha256=reference['nll_arrays_sha256'],dataset_identity=imported['dataset']['identity'],
        payload=str(payload),payload_sha256=sha(payload),payload_bytes=payload.stat().st_size,
        patch=str(patch),patch_sha256=sha(patch),runtime_manifest_sha256=sha(runtime_path),
        reused_identical_input_payload=bool(args.reuse_payload),
        source_files=len(sources),input_files=len(inputs),
        scope='Fresh committed sources, unchanged generic inputs, verified nominal pre-fit HHVV Asimov and native NLL references. Only the existing isolated ROOT 6.40/Combine v11 environment is reused. No observed fit outputs are read or included.')
    save_json(args.output,report)
    print(report['status'],report['payload_bytes'],report['payload_sha256'],flush=True)


if __name__=='__main__':main()
