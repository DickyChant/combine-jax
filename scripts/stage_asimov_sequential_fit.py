"""Freeze a ROOT-free Asimov GPU fit payload from committed project sources."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tarfile

from tensor_benchmark_common import save_json, sha


def required_inputs(manifest, case, native):
    # The worker hashes the original Asimov workspace as part of its blinding
    # proof. Transporting these bytes does not load ROOT on the GPU worker.
    return [str(manifest), case['workspace'], case['points'], case['generation_report'],
            *(str(Path(case['model']) / n) for n in ('model.json', 'arrays.npz')),
            *native.values()]


def check_staged_references(work, runtime):
    """Run the real guard from the payload cwd, without access to local inputs."""
    code = '''
import json, sys
from pathlib import Path
sys.path.insert(0, 'scripts')
sys.path.insert(0, 'src')
from run_asimov_sequential_fit import checked_reference
from tensor_benchmark_common import sha
runtime = json.loads(Path('build/sequential_fit_runtime.json').read_text())
case = json.loads(Path(runtime['manifest']).read_text())['hhvv']
metadata = json.loads((Path(case['model'])/'model.json').read_text())['metadata']
for path in runtime['native_references'].values():
    native = json.loads(Path(path).read_text())
    checked_reference(case, metadata, native, manifest_sha256=sha(runtime['manifest']))
assert 'ROOT' not in sys.modules
print('Both native/Asimov reference guards pass from the staged payload; ROOT not imported.')
'''
    import sys
    result = subprocess.run([sys.executable, '-c', code], cwd=work,
                            capture_output=True, text=True, timeout=180)
    if result.returncode:
        raise RuntimeError('Staged payload reference guard failed: ' + result.stderr)
    return dict(returncode=result.returncode, stdout=result.stdout, root_imported=False,
                input_hashes_verified=all(sha(work/p) == h for p,h in runtime['input_files'].items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    collection = Path('reports/gpu_tensor_asimov_completed_memory_collection.json')
    assert json.loads(collection.read_text())['status'] == 'passed_twelve_native_gradient_gates_and_native_fit_state'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    sources = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', revision,
                                      '--', 'src', 'scripts'], text=True).splitlines()
    source_hashes = {}
    for name in sources:
        expected = subprocess.check_output(['git', 'cat-file', 'blob', revision + ':' + name])
        assert Path(name).read_bytes() == expected, 'Uncommitted runtime source: ' + name
        source_hashes[name] = sha(name)
    manifest = Path('build/tensor_hhvv_asimov/manifest.json')
    case = json.loads(manifest.read_text())['hhvv']
    assert case['dataset'] == 'asimov_prefit'
    native = {family: f'reports/gpu_tensor_asimov_{family.lower()}_completed_native_fit.json'
              for family in ('A100', 'H100')}
    inputs = required_inputs(manifest, case, native)
    input_hashes = {p: sha(p) for p in inputs}
    metadata = json.loads((Path(case['model']) / 'model.json').read_text())['metadata']
    from run_asimov_sequential_fit import checked_reference
    hosts = {}
    for family, path in native.items():
        n = json.loads(Path(path).read_text())
        checked_reference(case, metadata, n, manifest_sha256=sha(manifest))
        hosts[family] = n['host']
    args.directory.mkdir(parents=True)
    work = args.directory / 'work'
    work.mkdir()
    for name in sources + inputs:
        destination = work / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if name in source_hashes:
            # A snapshot copy prevents later source edits changing the payload.
            destination.write_bytes(Path(name).read_bytes())
        else:
            os.link(name, destination)
    (work / 'reports').mkdir(exist_ok=True)
    runtime = dict(source_revision=revision, sources=source_hashes, input_files=input_hashes,
                   manifest=str(manifest), manifest_sha256=sha(manifest), native_references=native,
                   dataset='asimov_prefit', native_hosts=hosts)
    runtime_path = work / 'build/sequential_fit_runtime.json'
    save_json(runtime_path, runtime)
    staged_check = check_staged_references(work, runtime)
    assert staged_check['input_hashes_verified']
    payload = args.directory / 'payload.tar.gz'
    with tarfile.open(payload, 'w:gz', compresslevel=1) as archive:
        for name in sorted(set(sources + inputs + ['build/sequential_fit_runtime.json'])):
            archive.add(work / name, arcname=name, recursive=False)
    patch = args.directory / 'runtime_patch.tar'
    with tarfile.open(patch, 'w') as archive:
        member = tarfile.TarInfo('reports')
        member.type = tarfile.DIRTYPE
        member.mode = 0o755
        archive.addfile(member)
    save_json(args.output, dict(status='staged_asimov_sequential_gpu_fit', script_sha256=sha(__file__),
        source_revision=revision, payload=str(payload), payload_sha256=sha(payload),
        payload_bytes=payload.stat().st_size, patch=str(patch), patch_sha256=sha(patch),
        runtime_manifest_sha256=sha(runtime_path), dataset='asimov_prefit', native_hosts=hosts,
        input_files=input_hashes, completed_collection_sha256=sha(collection),
        staged_reference_check=staged_check,
        scope='Neutral translated arrays for execution; original Asimov workspace bytes included only for the unchanged provenance hash check. No ROOT import on the GPU worker. Exact native v11/ROOT 6.40 validation remains local. Original canonical manifest is preserved.'))


if __name__ == '__main__':
    main()
