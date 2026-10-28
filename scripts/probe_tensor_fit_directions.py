"""Native multidimensional probes after removing identified clipping normals.

Directions and all step sizes are saved before native evaluation. This diagnoses
an unconverged point; it changes neither the objective nor convergence policy.
"""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from tensor_benchmark_provenance import verified_source_record


def raw_bins(node, point):
    """Exported native additive morph before CropUnderflows, using NumPy only."""
    attrs = node.attrs
    assert node.class_name == 'FastVerticalInterpHistPdf2D2'
    assert not attrs['logarithmic'] and not attrs['conditional'] and attrs['region'] > 0
    theta = np.array([point[name] for name in attrs['coefs']])
    x = theta/attrs['region']
    alpha = np.where(abs(theta) >= attrs['region'], np.sign(theta), x*(x*x*(3*x*x-10)+15)/8)
    return (np.asarray(attrs['nominal']) + np.einsum('m,m...->...', theta/2, attrs['diffs'])
            + np.einsum('m,m...->...', theta*alpha/2, attrs['sums']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kinks', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    diagnosis = json.loads(args.kinks.read_text())
    assert diagnosis['status'] == 'clipping_surfaces_identified'
    assert sha(diagnosis['native_probes']) == diagnosis['native_probes_sha256']
    previous = json.loads(Path(diagnosis['native_probes']).read_text())
    assert sha(previous['fit']) == previous['fit_sha256']
    fit = json.loads(Path(previous['fit']).read_text())
    source = verified_source_record(fit, 'scripts/benchmark_tensor.py',
                                    previous['source_evidence']['source_commit'])
    assert sha(fit['arrays']) == fit['arrays_sha256'] == previous['arrays_sha256']
    assert sha(diagnosis['export']) == diagnosis['export_sha256']
    assert sha(diagnosis['manifest']) == diagnosis['manifest_sha256']
    manifest = json.loads(Path(diagnosis['manifest']).read_text())
    # Even unpickling old profile dataclasses imports JAX. Isolate that legacy
    # step and return only JSON/numeric arrays for the two selected histograms.
    selected_histograms = sorted({cell['histogram'] for row in diagnosis['parameters']
                                  for cell in row['channels']})
    preparation = subprocess.run([sys.executable, '-c', '''
import json,pickle,sys
with open(sys.argv[1], 'rb') as stream: payload=pickle.load(stream)
fields=('coefs','region','logarithmic','conditional','nominal','diffs','sums')
nodes={name: (payload['nodes'][name][0], {k:payload['nodes'][name][2][k] for k in fields})
       for name in sys.argv[2:]}
print(json.dumps(dict(source=payload['source'],nodes=nodes),default=lambda value:value.tolist()))
''', diagnosis['export'], *selected_histograms], text=True, capture_output=True, check=True)
    payload = json.loads(preparation.stdout)
    assert payload['source'] == manifest['models']['hhvv']['source']
    nodes = {name: SimpleNamespace(name=name, class_name=cls, attrs=attrs)
             for name, (cls, attrs) in payload['nodes'].items()}
    assert 'jax' not in sys.modules and 'torch' not in sys.modules
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
    model = TensorBundle.load(fit['model'])
    assert fit['model_sha256'] == {n: sha(Path(fit['model'])/n) for n in ('model.json', 'arrays.npz')}
    info = model_parameter_info(model)
    coordinates = fit_coordinates(info)
    free, scales = coordinates['free_indices'], coordinates['scale']
    names = np.array(model.metadata['parameter_names'])
    with np.load(fit['arrays'], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['parameter_names'], names)
        point = archive['fit_values'].copy()
        gradient = archive['fit_projected_gradient'].copy()
        physical_gradient = archive['fit_free_gradient'].copy()
        expected = float(archive['fit_value'])
    mapping = dict(zip(names, point, strict=True))
    free_map = {names[i]: j for j, i in enumerate(free)}
    cells = {cell['histogram']: cell for row in diagnosis['parameters'] for cell in row['channels']}
    normals, surfaces = [], []
    for name, cell in cells.items():
        node = nodes[name]
        attrs = node.attrs
        # The scalar morph is separable in its direct nuisance coefficients.
        theta = np.array([mapping[n] for n in attrs['coefs']])
        region = attrs['region']
        x = theta/region
        alpha = np.where(abs(theta) >= region, np.sign(theta), x*(x*x*(3*x*x-10)+15)/8)
        derivative = np.where(abs(theta) >= region, 0., 15*(x*x-1)**2/(8*region))
        flat = cell['bin_flat_index']
        diffs = np.asarray(attrs['diffs']).reshape(len(theta), -1)[:, flat]
        sums = np.asarray(attrs['sums']).reshape(len(theta), -1)[:, flat]
        partials = (diffs+(alpha+theta*derivative)*sums)/2
        normal = np.zeros(len(free))
        for coefficient, partial in zip(attrs['coefs'], partials, strict=True):
            j = free_map[coefficient]
            normal[j] = partial*scales[j]
        norm = np.linalg.norm(normal)
        assert norm > 0 and np.isfinite(norm)
        normals.append(normal/norm)
        surfaces.append(dict(histogram=name, bin_flat_index=flat,
            coefficients=attrs['coefs'], raw_bin_partials=partials,
            raw_bin=float(raw_bins(node, mapping).ravel()[flat]),
            scaled_normal_norm=norm))
    normals = np.stack(normals)
    assert len(normals) == 2
    projected = gradient-normals.T@np.linalg.solve(normals@normals.T, normals@gradient)
    touched = np.any(normals != 0, axis=0)
    interior = (point[free] > coordinates['physical_lower']) & (point[free] < coordinates['physical_upper'])
    assert np.all(interior[touched])
    directions = []
    # Separate movement tangent to the two surfaces from movement in coordinates
    # that do not affect either of the identified histograms at all.
    for label, support in [('surface_tangent', touched), ('other_interior_parameters', ~touched & interior)]:
        vector = np.where(support, -projected, 0.)
        norm = np.max(abs(vector))
        assert norm > 0
        vector /= norm
        assert np.max(abs(normals@vector)) < 1e-9
        delta = np.zeros(len(names)); delta[free] = vector*scales
        indices = np.flatnonzero(delta)
        steps = [10.**(-k) for k in range(2, 8)]
        for step in steps:
            assert np.all(point-2*step*abs(delta) >= info.arrays['lower'])
            assert np.all(point+2*step*abs(delta) <= info.arrays['upper'])
        directions.append(dict(name=label, residual_max=float(norm),
            indices=indices, parameters=names[indices], physical_direction=delta[indices],
            scaled_ad_directional_derivative=float(physical_gradient@(vector*scales)),
            normal_dot_direction=normals@vector, steps=steps, probes=[]))
    assert not (set(directions[0]['parameters']) & set(directions[1]['parameters']))
    case = json.loads(Path('build/tensor_benchmark_inputs/manifest.json').read_text())['hhvv']
    assert sha(case['workspace']) == case['workspace_sha256'] == previous['workspace_sha256']
    report = dict(provenance(), status='directions_prepared', script_sha256=sha(__file__),
        kinks=str(args.kinks), kinks_sha256=sha(args.kinks), fit=previous['fit'],
        fit_sha256=previous['fit_sha256'], arrays_sha256=fit['arrays_sha256'], source_evidence=source,
        workspace_sha256=case['workspace_sha256'], surfaces=surfaces, directions=directions,
        original_projected_gradient_max=float(max(abs(gradient))),
        after_normal_projection_max=float(max(abs(projected))),
        preparation='Trusted legacy pickle read in a separate subprocess; only selected histogram JSON/arrays enter this native process.',
        preparation_stderr=preparation.stderr,
        scope='Native fixed-step directional finite differences at an unconverged point. '
              'Directions are chosen from saved AD and exact exported morph normals before '
              'native probes; steps are not selected by AD agreement. Tangency is first-order '
              'only. No new gradient acceptance, fit certification, curvature or performance '
              'claim. Concurrent local fit makes this correctness work only.')
    monitor = Resources(False, args.output)

    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)

    save()
    try:
        from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
        root = load_caching_nll()
        assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
        report.update(root_version='6.40.00', combine_version='11.0.0', runtime_flags=REFERENCE_FLAGS)
        with open_workspace(case['workspace']) as workspace:
            for name, config in case['configuration'].items():
                workspace.var(name).setVal(config['value']); workspace.var(name).setConstant(config['constant'])
            for choice in model.metadata['provenance']['discrete_choices'].values():
                workspace.cat(choice['category']).setIndex(choice['index'])
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
            variables = [workspace.var(n) for n in names]
            for variable, value in zip(variables, point, strict=True): variable.setVal(float(value))

            def evaluate():
                root.RooAbsReal.clearEvalErrorLog()
                value = float(nll.getVal())
                assert root.RooAbsReal.numEvalErrors() == 0 and np.isfinite(value)
                return value

            canonical = evaluate()
            report.update(canonical_nll=canonical, saved_nll_absolute_error=abs(canonical-expected))
            assert abs(canonical-expected) < 1e-6
            nll.setZeroPoint(); baseline = evaluate(); assert abs(baseline) < 1e-6
            for row in directions:
                for step in row['steps']:
                    deltas, bin_offsets = {}, {}
                    for multiple in (-2, -1, 1, 2):
                        changed = dict(mapping)
                        for i, delta in zip(row['indices'], row['physical_direction'], strict=True):
                            value = float(point[i]+multiple*step*delta)
                            variables[i].setVal(value); changed[names[i]] = value
                        deltas[multiple] = evaluate()-baseline
                        bin_offsets[multiple] = [float(raw_bins(nodes[s['histogram']], changed).ravel()[s['bin_flat_index']]-s['raw_bin']) for s in surfaces]
                    for i in row['indices']: variables[i].setVal(float(point[i]))
                    restored = evaluate()-baseline
                    assert abs(restored) < 1e-6
                    if row['name'] == 'other_interior_parameters':
                        assert all(offset == 0. for offsets in bin_offsets.values() for offset in offsets)
                    row['probes'].append(dict(step=step, zeroed_nll_deltas=deltas,
                        raw_bin_offsets=bin_offsets, central=(deltas[1]-deltas[-1])/(2*step),
                        forward=(4*deltas[1]-deltas[2])/(2*step),
                        backward=(-4*deltas[-1]+deltas[-2])/(2*step),
                        restoration_absolute_error=abs(restored)))
                    save()
                print(row['name'], 'AD', row['scaled_ad_directional_derivative'],
                      'native', [(p['step'], p['central'], p['zeroed_nll_deltas'][1]) for p in row['probes']], flush=True)
            nll.clearZeroPoint(); nll.clearConstantZeroPoint()
            report['canonical_restoration_absolute_error'] = abs(evaluate()-canonical)
            assert report['canonical_restoration_absolute_error'] < 1e-6
            nll.setAnalyticBarlowBeeston(False)
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='diagnostic_complete', native_evaluation_errors=0, jax_imported=False, torch_imported=False)
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc()); raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
