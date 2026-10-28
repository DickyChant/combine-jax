"""Cross-check clipping normals with NumPy differences and native branch probes."""
import argparse
import json
from pathlib import Path

import numpy as np

from tensor_benchmark_common import save_json, sha


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--native-probes', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    report = read(args.inventory)
    assert report['status'] == 'geometry_diagnostic_complete' and report['normal_rank'] == 4
    assert report['script_sha256'] == sha('scripts/inventory_tensor_clipping_surfaces.py')
    fit = read(report['fit'])
    assert sha(report['fit']) == report['fit_sha256']
    assert sha(fit['arrays']) == fit['arrays_sha256'] == report['arrays_sha256']
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import unpack_model
    from combine_jax.tensor.binned_channels import unpack_binned
    from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
    from combine_jax.tensor.scalar import evaluate_numpy
    from combine_jax.tensor.fast_histograms import morph_weights
    model = TensorBundle.load(fit['model'])
    assert report['model_sha256'] == {n: sha(Path(fit['model'])/n) for n in ('model.json', 'arrays.npz')}
    parts = unpack_model(model); scalars = parts['scalars']; hist = unpack_binned(parts['binned'])['histograms']
    coordinates = fit_coordinates(model_parameter_info(model)); free, scale = coordinates['free_indices'], coordinates['scale']
    with np.load(fit['arrays'], allow_pickle=False) as archive:
        point = archive['fit_values'].copy(); gradient = archive['fit_projected_gradient'].copy()
    native = {}
    for path in args.native_probes:
        probes = read(path)
        assert probes['status'] == 'diagnostic_complete' and probes['fit_sha256'] == report['fit_sha256']
        assert probes['arrays_sha256'] == report['arrays_sha256']
        assert probes['saved_nll_absolute_error'] == probes['canonical_restoration_absolute_error'] == 0
        assert probes['combine_version'] == '11.0.0' and probes['root_version'] == '6.40.00'
        for row in probes['points']:
            assert row['name'] not in native
            assert len(row['probes']) == 7 and all(p['restoration_absolute_error'] == 0 for p in row['probes'])
            native[row['name']] = row
    rows, normals = [], []
    for surface in report['selected']:
        batch = hist.metadata['schedule'][surface['batch']]
        assert not batch['logarithmic']
        buffers = {k: hist.arrays[v] for k, v in batch['buffers'].items()}
        hi, flat = surface['row'], surface['flat_bin']
        mask = buffers['morph_mask'][hi]; indices = buffers['inputs'][hi][mask]
        diffs = buffers['diffs'][hi][mask].reshape(len(indices), -1)[:, flat]
        sums = buffers['sums'][hi][mask].reshape(len(indices), -1)[:, flat]
        region = np.array([buffers['region'][hi]])
        norm = surface['scaled_normal_norm']
        normal = np.zeros(len(free)); normal[surface['free_indices']] = surface['normalized_components']; normals.append(normal)
        comparisons = []
        for j in surface['free_indices']:
            for fraction in (1e-3, 1e-4):
                plus, minus = point.copy(), point.copy()
                plus[free[j]] += fraction*scale[j]; minus[free[j]] -= fraction*scale[j]
                tp = evaluate_numpy(scalars, plus)[indices]; tm = evaluate_numpy(scalars, minus)[indices]
                ap = morph_weights(tp[None, :], region)[0]; am = morph_weights(tm[None, :], region)[0]
                # Difference the morph increments before adding the common base;
                # subtracting two almost-cancelled raw bins needlessly loses bits.
                difference = ((tp-tm)@diffs+(tp*ap-tm*am)@sums)/2
                estimated = float(difference/(2*fraction*norm))
                relative = abs(estimated-normal[j])/max(abs(normal[j]), 1e-8)
                comparisons.append(dict(parameter=model.metadata['parameter_names'][free[j]], step_fraction=fraction,
                    normalized_ad_partial=float(normal[j]), normalized_numpy_fd_partial=estimated,
                    relative_error=float(relative)))
        assert max(r['relative_error'] for r in comparisons) < 1e-4
        leader = int(np.argmax(abs(normal))); name = model.metadata['parameter_names'][free[leader]]
        assert name in native, name
        probe = native[name]
        assert probe['coordinate'] == point[free[leader]]
        branch_checks = [dict(step=p['step'], backward=p['backward'], forward=p['forward'],
                             signs_bracket_zero=p['backward'] < 0 < p['forward']) for p in probe['probes']]
        assert sum(p['signs_bracket_zero'] for p in branch_checks) >= 2
        rows.append(dict(histogram=surface['name'], flat_bin=flat, distance_to_floor=surface['distance_to_floor'],
            scalar_names=surface['scalar_names'], normal_comparisons=comparisons,
            dominant_parameter=name, native_one_sided_probes=branch_checks))
    normals = np.array(normals)
    coefficients, _, rank, _ = np.linalg.lstsq(normals.T, gradient, rcond=None)
    residual = gradient-normals.T@coefficients
    assert rank == report['normal_rank']
    np.testing.assert_allclose(coefficients, report['projection_coefficients'], rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(max(abs(residual)), report['normal_projected_gradient_max'], rtol=1e-12, atol=1e-14)
    save_json(args.output, dict(status='normals_and_native_kinks_checked_no_fit_acceptance', script_sha256=sha(__file__),
        inventory=str(args.inventory), inventory_sha256=sha(args.inventory),
        native_probes={str(p): sha(p) for p in args.native_probes}, surfaces=rows,
        normal_numpy_fd_max_relative_error=max(r['relative_error'] for s in rows for r in s['normal_comparisons']),
        original_projected_gradient_max=report['original_gradient_max'], normal_projected_gradient_max=float(max(abs(residual))),
        scope='All four selected surface normals checked by independent NumPy finite differences at two fixed steps. '
              'Native one-sided full-NLL probes for each dominant parameter bracket zero at multiple predeclared steps; '
              'smallest steps remain roundoff-sensitive and are not selected to agree with AD. This explains the '
              'large smooth-gradient residuals but does not prove a multidimensional minimum, feasible convex '
              'subgradient weights, or curvature. Fit status and uncertainty acceptance are unchanged.'))
    print('checked four surfaces; normal-projected residual', float(max(abs(residual))), flush=True)


if __name__ == '__main__':
    main()
