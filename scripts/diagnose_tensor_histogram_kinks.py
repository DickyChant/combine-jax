"""Locate native fast-template clipping surfaces at a saved tensor fit point."""
import argparse
import json
from pathlib import Path

import numpy as np

from hhvv_export_cache import load_export
from tensor_benchmark_common import save_json, sha
from combine_jax.discrete import active_dependencies


def raw_bins(node, point):
    """Evaluate the exported native morph formula before CropUnderflows."""
    attrs = node.attrs
    assert node.class_name == 'FastVerticalInterpHistPdf2D2'
    assert not attrs['logarithmic'] and not attrs['conditional']
    theta = np.array([point[name] for name in attrs['coefs']])
    region = attrs['region']
    assert region > 0
    x = theta/region
    alpha = np.where(abs(theta) >= region, np.sign(theta),
                     x*(x*x*(3*x*x-10)+15)/8)
    return (np.asarray(attrs['nominal'])
        + np.einsum('m,m...->...', theta/2, attrs['diffs'])
        + np.einsum('m,m...->...', theta*alpha/2, attrs['sums']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-probes', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    probes = json.loads(args.native_probes.read_text())
    assert probes['status'] == 'diagnostic_complete'
    fit = json.loads(Path(probes['fit']).read_text())
    assert sha(probes['fit']) == probes['fit_sha256']
    assert sha(fit['arrays']) == probes['arrays_sha256'] == fit['arrays_sha256']
    export = Path('build/gpu_exports/hhvv.pkl')
    manifest_path = Path('build/gpu_exports/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    entry = manifest['models']['hhvv']
    export_reference = Path('reports/tensor_hhvv_compile_scaling_audited_ww_2d_1_0.json')
    exported = json.loads(export_reference.read_text())
    assert sha(export) == exported['input_export_sha256']
    assert export == manifest_path.parent/entry['cache']
    source = entry['source']
    assert source['sha256'] == probes['workspace_sha256']
    model = load_export(export, source)
    with np.load(fit['arrays'], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays['parameter_names'], model.parameter_names)
        point = dict(zip(model.parameter_names, arrays['fit_values'], strict=True))
    dependencies = {c.name: active_dependencies(model.nodes, [c.pdf]) for c in model.channels}
    floor = 1e-9
    rows = []
    for item in probes['points']:
        name = item['name']
        assert point[name] == item['coordinate']
        row = dict(name=name, physical_ad_gradient=item['physical_ad_gradient'], channels=[])
        rows.append(row)
        for channel in model.channels:
            if name not in dependencies[channel.name]:
                continue
            affected = [model.nodes[n] for n in sorted(dependencies[channel.name])
                if model.nodes[n].class_name == 'FastVerticalInterpHistPdf2D2'
                and name in active_dependencies(model.nodes, [n])]
            assert len(affected) == 1
            node = affected[0]
            assert name in node.attrs['coefs'], 'Diagnostic requires a direct morph parameter'
            raw = raw_bins(node, point)
            nearest = int(np.argmin(abs(raw.ravel()-floor)))
            index = np.unravel_index(nearest, raw.shape)
            cell = dict(channel=channel.name, histogram=node.name, class_name=node.class_name,
                shape=raw.shape, observed_rows=len(channel.data.weights),
                bin_flat_index=nearest, bin_indices=index,
                bin_edges=[list(map(float, edge[i:i+2])) for edge, i in zip(node.attrs['edges'], index, strict=True)],
                pre_crop_bin=float(raw.ravel()[nearest]),
                distance_to_floor=float(raw.ravel()[nearest]-floor), probes=[])
            row['channels'].append(cell)
            for probe in item['probes']:
                h = probe['step']
                left = dict(point); left[name] -= h
                right = dict(point); right[name] += h
                lo = float(raw_bins(node, left).ravel()[nearest])
                hi = float(raw_bins(node, right).ravel()[nearest])
                cell['probes'].append(dict(step=h, pre_crop_minus=lo, pre_crop_plus=hi,
                    crosses_floor=min(lo, hi) < floor < max(lo, hi),
                    native_backward=probe['backward'], native_forward=probe['forward'],
                    native_central=probe['central'],
                    one_sided_signs_bracket_zero=probe['backward'] < 0 < probe['forward']))
        assert row['channels']
    native_files = [
        'vendor/combine_pypi/vendor/CombinedLimit/src/VerticalInterpHistPdf.cc',
        'vendor/combine_pypi/vendor/CombinedLimit/interface/FastTemplate.h',
        'vendor/combine_pypi/vendor/CombinedLimit/interface/FastTemplate.hpp']
    assert all(p['crosses_floor'] for row in rows for cell in row['channels'] for p in cell['probes'])
    save_json(args.output, dict(status='clipping_surfaces_identified',
        script_sha256=sha(__file__), native_probes=str(args.native_probes),
        native_probes_sha256=sha(args.native_probes), export=str(export),
        export_sha256=sha(export), export_reference=str(export_reference),
        export_reference_sha256=sha(export_reference),
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
        native_sources={p: sha(p) for p in native_files}, floor=floor, parameters=rows,
        formula='Native FastVerticalInterpHistPdf2Base::syncTotal melds nominal, diff and sum, '
                'then calls CropUnderflows; FastTemplate default minimum is 1e-9.',
        interpretation='The two nearest bins cross the native clipping floor under all tested '
            'parameter displacements. Native one-sided derivatives bracket zero at the saved point. '
            'This supports a nonsmooth explanation for these residuals, not smooth stationarity '
            'or a certified multidimensional minimum. The saved AD derivative is branch-specific; '
            'a classical Hessian at an exact clipping kink is not defined.',
        scope='Read-only diagnosis using the independently exported native templates and native '
              'finite differences. No smoothing, convergence-tolerance change, covariance repair '
              'or accepted HHVV uncertainties. NumPy bin arithmetic need not be bitwise identical '
              'to native sequential melds or fused XLA arithmetic.'))


if __name__ == '__main__':
    main()
