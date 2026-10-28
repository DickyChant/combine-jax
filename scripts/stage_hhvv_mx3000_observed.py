"""Freeze a single combined-MX3000 observed job, including isolated importer."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
from hhvv_mx3000_authorization import authorize, WORKSPACE_SHA256, CARD_SHA256, CATEGORIES
from tensor_benchmark_common import save_json, sha
from orchestrate_hhvv_mx3000_observed import stage_environment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--source-revision', required=True)
    p.add_argument('--reuse-import-environment', type=Path,
        help='Reuse an archive from an existing payload after verifying its saved check hash')
    args = p.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    assert not args.directory.exists()
    args.directory.mkdir(parents=True)
    base = Path('build/hhvv_gpu_limits_20261028')
    built = json.loads((base/'MX3000_stack_retry/build.json').read_text())
    prepared = base/'MX3000_tensor_stack_retry'
    prep = json.loads((prepared/'preparation.json').read_text())
    assert built['workspace_sha256'] == WORKSPACE_SHA256 and built['card_sha256'] == CARD_SHA256
    assert prep['generation']['source_workspace_sha256'] == WORKSPACE_SHA256
    assert prep['generation']['mass'] == 3000 and prep['generation']['signal_strength'] == 0.
    native_gate = json.loads(Path('reports/asymptotic_observed_generic_v10_validation.json').read_text())
    assert native_gate['status'] == 'passed'
    mapping = {p:p for p in Path('src/combine_jax').rglob('*.py')}
    mapping.update({p:p for p in Path('scripts').iterdir() if p.suffix in ('.py','.sh','.cc')})
    for name in ('workspace.root','card.txt','build.json'):
        mapping[base/'MX3000_stack_retry'/name] = Path('build/hhvv_observed/input')/name
    for name, checksum in prep['model_sha256'].items():
        source = prepared/'model'/name
        assert sha(source) == checksum
        mapping[source] = Path('build/hhvv_observed/prefit_model')/name
    mapping[Path('reports/asymptotic_observed_generic_v10_validation.json')] = Path('build/hhvv_observed/generic_validation.json')
    # Python's absolute interpreter symlink resolves to the same immutable CVMFS
    # Python on workers; the site-packages tree relocates through pyvenv.cfg.
    archive_path = args.directory/'import_environment.tar.gz'
    if args.reuse_import_environment:
        previous = args.reuse_import_environment
        proof = json.loads(previous.with_name('import_environment_check.json').read_text())
        assert proof['returncode'] == 0 and sha(previous) == proof['archive_sha256']
        shutil.copyfile(previous, archive_path)
    else:
        with tarfile.open(archive_path, 'w:gz') as archive:
            def keep(info):
                if '__pycache__' in Path(info.name).parts or info.name.endswith('.pyc'):
                    return None
                return info
            archive.add('.venv-import', arcname='.venv-import', filter=keep)
    mapping[archive_path] = Path('build/hhvv_observed/import_environment.tar.gz')
    check = args.directory/'relocation_check'
    check.mkdir()
    with tarfile.open(archive_path) as archive:
        # All symlinks come from our existing venv, including its CVMFS Python.
        archive.extractall(check, filter='fully_trusted')
    env = stage_environment(gpu=False)
    # Do not resolve the executable for the actual import: retain venv detection.
    command = [str((check/'.venv-import/bin/python').absolute()), '-c',
        'import ROOT,json,jax,jax.numpy as jnp; from importlib.metadata import version; '
        'assert jax.config.x64_enabled; assert all(d.platform=="cpu" for d in jax.devices()); '
        'x=jnp.asarray([1.,2.],dtype=jnp.float64); assert float(jnp.sum(x*x))==5.; '
        'print(json.dumps(dict(root=ROOT.gROOT.GetVersion(),combine=version("higgs-combine-tool"),'
        'jax=jax.__version__,platforms=[d.platform for d in jax.devices()],float64=jax.config.x64_enabled))); '
        'assert ROOT.gROOT.GetVersion()=="6.40.00"; assert version("higgs-combine-tool")=="11.0.0"']
    test = subprocess.run(command, env=env, capture_output=True, text=True, timeout=120)
    save_json(args.directory/'import_environment_check.json',dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr,archive_sha256=sha(archive_path),jax_platforms=env['JAX_PLATFORMS']))
    assert test.returncode == 0, test.stdout+test.stderr
    manifest = dict(run='observed', mass=3000, combination='HHVV_v4_final',
        workspace_sha256=WORKSPACE_SHA256, card_sha256=CARD_SHA256,
        fixed_categories=CATEGORIES, fixed_discrete_choices=prep['discrete_choices'],
        native_library_sha256=built['native_library_sha256'], source_revision=args.source_revision,
        static_guard_gib=8., float64=True, template_convention='v10_density',
        optional_expected_quantile=.025, optional_seconds=600,
        files={str(target):sha(source) for source,target in mapping.items()})
    authorize(manifest, base/'MX3000_stack_retry/workspace.root', base/'MX3000_stack_retry/card.txt')
    manifest_path = args.directory/'runtime_manifest.json'
    save_json(manifest_path,manifest)
    empty = args.directory/'empty_reports'; empty.mkdir()
    with tarfile.open(args.directory/'payload.tar.gz','w:gz') as archive:
        for source,target in mapping.items():
            archive.add(source,arcname=str(target),recursive=False)
        archive.add(manifest_path,arcname='build/hhvv_observed_runtime.json')
        archive.add(empty,arcname='reports')
    shutil.copyfile('scripts/condor_hhvv_mx3000_observed.sh', args.directory/'worker.sh')
    (args.directory/'worker.sh').chmod(0o755)
    text = Path('scripts/condor_hhvv_gpu_limit.sub').read_text().replace('request_disk = 6GB','request_disk = 14GB')
    (args.directory/'job.sub').write_text(text)
    receipt = dict(status='staged',run='observed',mass=3000,source_revision=args.source_revision,
        directory=str(args.directory),manifest_sha256=sha(manifest_path),static_guard_gib=8.,float64=True,
        discrete_policy='fixed',fixed_discrete_choices=prep['discrete_choices'],
        workspace_sha256=WORKSPACE_SHA256,card_sha256=CARD_SHA256,
        files={n:sha(args.directory/n) for n in ('payload.tar.gz','worker.sh','job.sub')},
        payload_bytes=(args.directory/'payload.tar.gz').stat().st_size,
        environment_check_sha256=sha(args.directory/'import_environment_check.json'),
        generic_validation_sha256=sha('reports/asymptotic_observed_generic_v10_validation.json'),
        authorization='User explicitly unblinded combined HHVV MX3000 only; other masses/channels remain blind.',
        native_limit_validation='pending')
    save_json(args.directory/'staging.json',receipt)
    print('staged observed MX3000',args.directory,receipt['payload_bytes'],flush=True)


if __name__ == '__main__':
    main()
