"""Create portable neutral models and frozen native/old-backend reference points."""
import json
from pathlib import Path
import shutil
import subprocess

import jax
import numpy as np

from build_template_fixture import build_template_fixture
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.validation import validate_workspace
from combine_jax.tensor.model import from_workspace
from tensor_benchmark_common import provenance, save_json, sha


def main():
    directory = Path('build/tensor_benchmark_inputs')
    output = Path('reports/gpu_tensor_benchmark_inputs.json')
    assert not directory.exists() and not output.exists()
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    directory.mkdir()
    report = dict(provenance(), status='preparing', script_sha256=sha(__file__), cases={},
        scope='Transport and numerical reference preparation only; no benchmark timings.')
    save_json(output, report)
    root = load_caching_nll()
    counting_dir = directory/'counting'
    counting_dir.mkdir()
    shutil.copyfile('tests/cards/counting.txt', counting_dir/'card.txt')
    subprocess.run(['text2workspace.py', str(counting_dir/'card.txt'), '-o', str(counting_dir/'workspace.root')],
                   check=True, stdout=subprocess.DEVNULL)
    automatic = build_template_fixture(directory/'automcstats')
    analytic_path = Path('reports/tensor_analytic_roundoff_fits.json')
    analytic = json.loads(analytic_path.read_text())
    assert analytic['status'] == 'passed'
    items = [('counting', counting_dir/'workspace.root', None), ('automcstats', automatic, None)]
    items += [(c['label'], Path(c['workspace']), i if c['label'].startswith('multipdf') else None)
              for c, i in zip(analytic['cases'], (None, 0, 1), strict=True)]
    for label, workspace_path, discrete in items:
        print('prepare', label, flush=True)
        case_dir = directory/label
        case_dir.mkdir(exist_ok=True)
        if workspace_path.resolve() != (case_dir/'workspace.root').resolve():
            shutil.copyfile(workspace_path, case_dir/'workspace.root')
            shutil.copyfile(workspace_path.parent/'card.txt', case_dir/'card.txt')
        workspace_path = case_dir/'workspace.root'
        with open_workspace(workspace_path) as workspace:
            if discrete is not None:
                workspace.cat('pdf_index').setIndex(discrete)
                for n, e in [('multi_slope', .005), ('multi_c1', .1), ('multi_c2', .1)]:
                    workspace.var(n).setError(e)
            model = from_workspace(workspace, case_dir/'card.txt')
            model.save(case_dir/'model')
            native = validate_workspace(workspace, random_points=25, seed=1819)
        save_json(case_dir/'reference.json', native)
        assert native['max_nll_absolute_error'] < 1e-6
        assert native['max_gradient_relative_error'] < 1e-4
        assert native['max_hessian_scaled_error'] < 1e-6
        points = np.array([c['theta'] for c in native['cases']], dtype=np.float64)
        names = np.array(native['parameters'])
        np.savez_compressed(case_dir/'points.npz', points=points, parameter_names=names)
        np.savez_compressed(case_dir/'curvature_points.npz', points=points[:3], parameter_names=names)
        np.savez_compressed(case_dir/'reference.npz', points=points, parameter_names=names,
            values=np.array([c['combine_nll'] for c in native['cases']]),
            gradients=np.array([c['jax_gradient'] for c in native['cases']]),
            native_fd_gradients=np.array([c['combine_fd_gradient'] for c in native['cases']]),
            hessians=np.array([c['hessian'] for c in native['cases'][:3]]))
        report['cases'][label] = dict(workspace=str(workspace_path), configuration={}, model=str(case_dir/'model'),
            points=str(case_dir/'points.npz'), curvature_points=str(case_dir/'curvature_points.npz'),
            reference=str(case_dir/'reference.npz'), native_reference=str(case_dir/'reference.json'),
            parameters=len(names), points_count=len(points),
            native_gates={n: native[n] for n in ('max_nll_absolute_error','max_gradient_relative_error','max_hessian_scaled_error')},
            hashes={str(p): sha(p) for p in sorted(case_dir.rglob('*')) if p.is_file()})
        save_json(output, report)
        jax.clear_caches()
    hhdir = directory/'hhvv'
    hhdir.mkdir()
    native_path = Path('reports/hhvv_random_values.json')
    native = json.loads(native_path.read_text())
    capture_path = Path('reports/tensor_hhvv_unique_gradients_capture.json')
    capture = json.loads(capture_path.read_text())
    assert sha(capture['arrays']) == capture['arrays_sha256']
    gate_path = Path('reports/tensor_hhvv_unique_gradients.json')
    assert json.loads(gate_path.read_text())['status'] == 'passed'
    points = np.array([c['theta'] for c in native['cases']], dtype=np.float64)
    names = np.array(native['parameter_names'])
    with np.load(capture['arrays'], allow_pickle=False) as a:
        np.testing.assert_array_equal(a['validation_points'], points)
        np.testing.assert_array_equal(a['parameter_names'], names)
        gradients = a['validation_gradients']
    np.savez_compressed(hhdir/'points.npz', points=points, parameter_names=names)
    np.savez_compressed(hhdir/'curvature_points.npz', points=points[:1], parameter_names=names)
    np.savez_compressed(hhdir/'reference.npz', points=points, parameter_names=names,
                       values=np.array([c['combine_nll'] for c in native['cases']]), gradients=gradients)
    import_path = Path('reports/tensor_hhvv_fit_import.json')
    imported = json.loads(import_path.read_text())
    assert imported['status'] == 'passed'
    report['cases']['hhvv'] = dict(workspace='inputs/workspace_MX2000.root',
        workspace_sha256=imported['source']['sha256'], configuration=imported['source']['configuration'],
        model=imported['model'], points=str(hhdir/'points.npz'), curvature_points=str(hhdir/'curvature_points.npz'),
        reference=str(hhdir/'reference.npz'), parameters=len(names), points_count=len(points),
        reference_provenance={str(p): sha(p) for p in (native_path, capture_path, gate_path, import_path)},
        gradient_reference='Accepted tensor CPU gradients; native FD policy remains in the linked gate. Cross-device agreement is supplemental to that policy.',
        hashes={str(p): sha(p) for p in [*hhdir.iterdir(), *Path(imported['model']).iterdir()] if p.is_file()})
    manifest = directory/'manifest.json'
    save_json(manifest, report['cases'])
    report.update(status='prepared', manifest=str(manifest), manifest_sha256=sha(manifest),
                  root_version=str(root.gROOT.GetVersion()))
    save_json(output, report)


if __name__ == '__main__':
    main()
