"""Stage the approved single A100 fit with a new same-node native-v10 reference."""
import json
import math
from pathlib import Path
import shutil
import subprocess
import tarfile

import numpy as np

from native_fit_model import load_native_fit_model
from combine_jax.tensor.parameter_info import model_parameter_info
from tensor_benchmark_common import save_json, sha


def main():
    base = Path('build/hhvv_v10_campaign_20261024')
    imported = json.loads((base/'import_remote.json').read_text())
    assert imported['status'] == 'imported_identity_verified_numerical_gates_pending'
    native = json.loads((base/'generation.json').read_text())
    assert native['status'] == 'generated_native_v10_nll_reference' and native['nll_points'] == 26
    preflight = json.loads((base/'preflight_retry1.json').read_text())
    assert preflight['status'] == 'passed_preflight_only'
    assert preflight['script_sha256'] == sha('scripts/run_hhvv_v10_native_fit.py')
    model, _ = load_native_fit_model(base/'tensor/model')
    assert model.metadata['provenance']['template_convention'] == 'v10_density'
    assert model.metadata['provenance']['dataset_generation']['generation_report_sha256'] == sha(base/'generation_for_import.json')
    info = model_parameter_info(model)
    config = json.loads((base/'configuration.json').read_text())
    assert config['dataset_identity'] == model.metadata['provenance']['dataset']['identity']
    assert config['fit_state'] == info.metadata['fit_policy']
    assert config['names'] == info.metadata['names']
    np.testing.assert_array_equal(config['initial'], info.arrays['initial'])
    for field in ('lower', 'upper'):
        np.testing.assert_array_equal([float(v) for v in config[field]], info.arrays[field])
    for field, array in [('profile_lower','templates__profile_minimum'), ('profile_upper','templates__profile_maximum')]:
        np.testing.assert_array_equal([float(v) for v in config[field]], model.arrays[array])
    assert sha(base/'generation/workspace_asimov.root') == config['workspace_sha256'] == native['workspace_sha256']
    assert sha(base/'reference.npz') == native['reference_sha256']
    with np.load(base/'reference.npz', allow_pickle=False) as ref, np.load(base/'tensor/points.npz', allow_pickle=False) as points:
        np.testing.assert_array_equal(ref['points'], points['points'])
        np.testing.assert_array_equal(ref['parameter_names'], points['parameter_names'])
        assert np.isfinite(ref['values']).all()
    stage = base/'a100_payload'
    stage.mkdir(exist_ok=False)
    work = stage/'work'
    work.mkdir()
    mapping = {p: p for p in Path('src/combine_jax').rglob('*.py')}
    mapping.update({p: p for p in Path('scripts').iterdir() if p.suffix in ('.py', '.cc', '.sh')})
    for source, name in [('tensor/model/model.json', 'model/model.json'),
                         ('tensor/model/arrays.npz', 'model/arrays.npz'),
                         ('tensor/points.npz', 'points.npz'), ('reference.npz', 'reference.npz'),
                         ('generation/workspace_asimov.root', 'workspace_asimov.root'),
                         ('configuration.json', 'configuration.json'), ('generation.json', 'generation.json'),
                         ('generation_for_import.json', 'generation_for_import.json'),
                         ('import_remote.json', 'import.json'), ('remote_import_guard.json', 'import_guard.json'),
                         ('preflight_retry1.json', 'preflight.json')]:
        mapping[base/source] = Path('build/hhvv_v10')/name
    files = {}
    for source, destination in mapping.items():
        target = work/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        files[str(destination)] = sha(target)
    (work/'reports').mkdir()
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    save_json(work/'build/hhvv_v10_runtime.json', dict(files=files, source_revision=revision,
        native_reference='10.3.3 / ROOT 6.30.07', dataset='asimov_prefit', template_convention='v10_density'))
    with tarfile.open(stage/'payload.tar.gz', 'w:gz') as archive:
        for path in sorted(work.rglob('*')):
            if path.is_file():
                archive.add(path, arcname=str(path.relative_to(work)))
        archive.add(work/'reports', arcname='reports')
    shutil.copyfile('scripts/condor_hhvv_v10_a100.sh', stage/'worker.sh')
    (stage/'worker.sh').chmod(0o755)
    shutil.copyfile('scripts/condor_hhvv_v10_a100.sub', stage/'job.sub')
    checks = {name: sha(stage/name) for name in ('payload.tar.gz', 'worker.sh', 'job.sub')}
    save_json(stage/'checksums.json', checks)
    save_json('reports/gpu_hhvv_v10_a100_staging.json', dict(status='staged', directory=str(stage),
        source_revision=revision, script_sha256=sha(__file__), checksums=checks,
        native_generation_sha256=sha(base/'generation.json'), preflight_sha256=sha(base/'preflight_retry1.json'),
        import_sha256=sha(base/'import_remote.json'), dataset_identity=config['dataset_identity'],
        workspace_sha256=config['workspace_sha256'], model_sha256={n:sha(base/'tensor/model'/n) for n in ('model.json','arrays.npz')},
        static_guard_gib=8, float64=True, same_start_and_bounds_checked=True,
        scope='GPU 26-point NLL gate precedes fitting. One GPU fit and one native-v10 single-core Migrad on the same A100 worker. No old pip-v11 fit reference is used.'))
    print(stage, checks, flush=True)


if __name__ == '__main__':
    main()
