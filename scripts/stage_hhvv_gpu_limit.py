"""Freeze a prepared mass-point payload for the authorized H100/A100 limit round."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--build-report', type=Path, required=True)
    parser.add_argument('--guard-report', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--fixed-discrete', action='store_true')
    parser.add_argument('--source-revision', required=True)
    args = parser.parse_args()
    assert not args.directory.exists()
    prep = json.loads((args.prepared/'preparation.json').read_text())
    built = json.loads(args.build_report.read_text())
    guarded = json.loads(args.guard_report.read_text())
    assert prep['status'] == 'imported_native_v10_background_asimov'
    assert built['status'] == 'built' and guarded['status'] == 'completed'
    generation = prep['generation']
    assert generation['reopen_verified'] and generation['signal_strength'] == 0.
    assert generation['source_workspace_sha256'] == built['workspace_sha256']
    assert prep['card_sha256'] == built['card_sha256']
    assert generation['mass'] == built['mass']
    metadata = json.loads((args.prepared/'model/model.json').read_text())
    # TensorBundle serializes metadata under its own top-level key.
    metadata = metadata.get('metadata', metadata)
    choices = prep['discrete_choices']
    policy = 'fixed' if args.fixed_discrete else 'profile'
    if choices:
        assert args.fixed_discrete
        assert metadata['asymptotic_asimov']['fixed_discrete_choices'] == choices
    args.directory.mkdir(parents=True)
    mapping = {p: p for p in Path('src/combine_jax').rglob('*.py')}
    mapping.update({p:p for p in Path('scripts').iterdir() if p.suffix in ('.py', '.cc', '.sh')})
    for source, target in [(args.prepared/'preparation.json', 'preparation.json'),
                           (args.build_report, 'build.json'), (args.guard_report, 'import_guard.json')]:
        mapping[source] = Path('build/hhvv_limit')/target
    for name, checksum in prep['model_sha256'].items():
        source = args.prepared/'model'/name
        assert sha(source) == checksum
        mapping[source] = Path('build/hhvv_limit/model')/name
    manifest = dict(source_revision=args.source_revision, mass=generation['mass'],
        discrete_policy=policy, fixed_discrete_choices=choices,
        static_guard_gib=8., float64=True, files={str(target):sha(source) for source,target in mapping.items()})
    manifest_path = args.directory/'runtime_manifest.json'
    save_json(manifest_path, manifest)
    empty = args.directory/'empty_reports'; empty.mkdir()
    with tarfile.open(args.directory/'payload.tar.gz', 'w:gz') as archive:
        for source, target in mapping.items():
            archive.add(source, arcname=str(target), recursive=False)
        archive.add(manifest_path, arcname='build/hhvv_limit_runtime.json')
        archive.add(empty, arcname='reports')
    shutil.copyfile('scripts/condor_hhvv_gpu_limit.sh', args.directory/'worker.sh')
    (args.directory/'worker.sh').chmod(0o755)
    shutil.copyfile('scripts/condor_hhvv_gpu_limit.sub', args.directory/'job.sub')
    receipt = dict(status='staged', mass=generation['mass'], source_revision=args.source_revision,
        script_sha256=sha(__file__), directory=str(args.directory), discrete_policy=policy,
        fixed_discrete_choices=choices, manifest_sha256=sha(manifest_path),
        files={n:sha(args.directory/n) for n in ('payload.tar.gz','worker.sh','job.sub')},
        dataset=prep['dataset'], model_sha256=prep['model_sha256'],
        workspace_build_seconds=built['wall_seconds'], asimov_generation_seconds=generation['generation_seconds'],
        import_seconds=prep['import_seconds'], static_guard_gib=8., float64=True,
        native_quantile_validation='pending_reference_and_matching_discrete_configuration')
    save_json(args.directory/'staging.json', receipt)
    print(receipt['status'], receipt['mass'], args.directory, flush=True)


if __name__ == '__main__':
    main()
