"""Inventory additive fast-histogram floors and their parameter-space normals.

This is a diagnostic at a saved candidate, not a convergence criterion. Normal
projection alone is not a valid generalized-gradient or minimum certificate.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    parser.add_argument('--nearest', type=int, default=4)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and args.nearest > 0
    fit = json.loads(args.fit.read_text())
    assert fit['status'] == 'measured_pending_validation' and fit['operation'] == 'fit'
    source = verified_source_record(fit, 'scripts/benchmark_tensor.py', args.source_revision)
    assert sha(fit['arrays']) == fit['arrays_sha256']
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import unpack_model
    from combine_jax.tensor.binned_channels import unpack_binned
    from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
    from combine_jax.tensor.scalar import evaluate_numpy
    from combine_jax.tensor.fast_histograms import morph_weights
    model = TensorBundle.load(fit['model'])
    assert fit['model_sha256'] == {n: sha(Path(fit['model'])/n) for n in ('model.json', 'arrays.npz')}
    components = unpack_model(model)
    scalars = components['scalars']
    histograms = unpack_binned(components['binned'])['histograms']
    coordinates = fit_coordinates(model_parameter_info(model))
    with np.load(fit['arrays'], allow_pickle=False) as archive:
        names = archive['parameter_names'].copy()
        point = archive['fit_values'].copy()
        gradient = archive['fit_projected_gradient'].copy()
    np.testing.assert_array_equal(names, model.metadata['parameter_names'])
    values = evaluate_numpy(scalars, point)
    by_offset = {h['offset']: h for h in histograms.metadata['histograms']}
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        fit=str(args.fit), fit_sha256=sha(args.fit), arrays_sha256=fit['arrays_sha256'],
        model_sha256=fit['model_sha256'], source_evidence=source, floor=1e-9,
        selection_policy=f'Nearest bin in each additive dynamic histogram, ordered by absolute raw distance to 1e-9; first {args.nearest} selected before normal projection.',
        nearest_per_histogram=[], selected=[],
        scope='Batched neutral histogram inventory and scalar AD geometry at an unconverged point. '
              'Raw arithmetic may differ by roundoff from native sequential melds or fused device evaluation. '
              'Normals span possible gradient changes but do not supply convex subgradient weights; '
              'projection is not a minimum or uncertainty certificate. No tolerance or objective changes.')
    monitor = Resources(False, args.output)
    monitor.stage = 'inventory'
    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)
    save()
    try:
        batches = {}
        checked_bins = 0
        for bi, batch in enumerate(histograms.metadata['schedule']):
            if batch['logarithmic']: continue
            b = {n: histograms.arrays[k] for n, k in batch['buffers'].items()}
            batches[bi] = b
            theta = np.where(b['morph_mask'], values[b['inputs']], 0.)
            alpha = morph_weights(theta, b['region'])
            raw = b['base']+np.einsum('hm,hmsk->hsk', theta/2, b['diffs'])+np.einsum('hm,hmsk->hsk', theta*alpha/2, b['sums'])
            checked_bins += raw.size
            for hi in range(len(raw)):
                h = by_offset[int(b['outputs'][hi, 0])]
                flat = int(np.argmin(abs(raw[hi].ravel()-1e-9)))
                value = float(raw[hi].ravel()[flat])
                report['nearest_per_histogram'].append(dict(name=h['name'], class_name=h['class_name'],
                    batch=bi, row=hi, shape=h['shape'], flat_bin=flat, raw_bin=value,
                    distance_to_floor=value-1e-9, absolute_distance=abs(value-1e-9)))
        report['nearest_per_histogram'].sort(key=lambda r: r['absolute_distance'])
        report['additive_dynamic_bins_checked'] = checked_bins
        report['additive_dynamic_histograms_checked'] = len(report['nearest_per_histogram'])
        selected = report['nearest_per_histogram'][:args.nearest]
        assert len(selected) == args.nearest
        indices = sorted({int(i) for r in selected for i in batches[r['batch']]['inputs'][r['row']][batches[r['batch']]['morph_mask'][r['row']]]})
        save()  # Selection and full ranking precede scalar AD and projection.
        import jax
        import jax.numpy as jnp
        from combine_jax.tensor.jax_scalar import make_function
        assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
        scalar_function = make_function(scalars)
        def selected_scalars(theta, arrays, gather):
            return scalar_function(theta, arrays)[gather]
        runtime = (jax.device_put(point), jax.device_put(dict(scalars.arrays)), jax.device_put(np.array(indices, dtype=np.int32)))
        monitor.stage = 'scalar_jacobian'; save()
        compiled = jax.jit(jax.jacrev(selected_scalars)).lower(*runtime).compile()
        memory = compiled.memory_analysis()
        allocation = sum(getattr(memory, n) for n in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes'))
        assert allocation < 1024**3
        jacobian = np.asarray(jax.device_get(compiled(*runtime)))
        assert jacobian.shape == (len(indices), len(names)) and np.isfinite(jacobian).all()
        lookup = {index: j for j, index in enumerate(indices)}
        normals = []
        free, scales = coordinates['free_indices'], coordinates['scale']
        for row in selected:
            b = batches[row['batch']]; hi = row['row']; active = b['morph_mask'][hi]
            coefs = b['inputs'][hi][active]; theta = values[coefs]; region = b['region'][hi]
            assert region > 0
            x = theta/region
            alpha = np.where(abs(theta) >= region, np.sign(theta), x*(x*x*(3*x*x-10)+15)/8)
            derivative = np.where(abs(theta) >= region, 0., 15*(x*x-1)**2/(8*region))
            k = row['flat_bin']
            diffs = b['diffs'][hi][active].reshape(len(theta), -1)[:, k]
            sums = b['sums'][hi][active].reshape(len(theta), -1)[:, k]
            partials = (diffs+(alpha+theta*derivative)*sums)/2
            normal = (partials@jacobian[[lookup[int(i)] for i in coefs]])[free]*scales
            norm = np.linalg.norm(normal)
            assert norm > 0 and np.isfinite(norm)
            normal /= norm; normals.append(normal)
            support = np.flatnonzero(normal)
            report['selected'].append(dict(row, scalar_names=[scalars.metadata['output_names'][i] for i in coefs],
                scaled_normal_norm=float(norm), distance_over_normal_norm=row['distance_to_floor']/norm,
                free_indices=support, parameters=names[free[support]], normalized_components=normal[support]))
        normals = np.array(normals)
        coefficients, _, rank, singular_values = np.linalg.lstsq(normals.T, gradient, rcond=None)
        projected = gradient-normals.T@coefficients
        order = np.argsort(abs(projected))[-20:][::-1]
        report.update(status='geometry_diagnostic_complete', jax_version=jax.__version__,
            scalar_jacobian_allocation_bytes=allocation, normal_rank=int(rank), normal_singular_values=singular_values,
            projection_coefficients=coefficients, original_gradient_max=float(max(abs(gradient))),
            normal_projected_gradient_max=float(max(abs(projected))),
            remaining_largest=[dict(name=str(names[free[j]]), scaled_gradient=float(gradient[j]),
                                   normal_projected_gradient=float(projected[j])) for j in order])
        save()
        print(report['status'], len(report['nearest_per_histogram']), checked_bins,
              report['original_gradient_max'], report['normal_projected_gradient_max'], flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); save(); raise
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
