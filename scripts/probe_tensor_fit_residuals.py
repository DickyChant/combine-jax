"""Native one-sided finite differences at a saved, unconverged tensor point.

This is a diagnostic, not a new acceptance policy. Steps and parameter selection
are recorded before evaluation; no step is selected to agree with AD.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from tensor_benchmark_provenance import verified_source_record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--source-revision', required=True)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--count', type=int, default=5)
    selection.add_argument('--parameter', action='append',
                           help='Probe explicitly selected free parameters; repeat for multiple names')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and args.count > 0
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    fit = json.loads(args.fit.read_text())
    assert fit['status'] == 'measured_pending_validation' and fit['operation'] == 'fit'
    source = verified_source_record(fit, 'scripts/benchmark_tensor.py', args.source_revision)
    assert sha(fit['arrays']) == fit['arrays_sha256']
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates
    model = TensorBundle.load(fit['model'])
    assert fit['model_sha256'] == {n: sha(Path(fit['model'])/n) for n in ('model.json', 'arrays.npz')}
    info = model_parameter_info(model)
    coordinates = fit_coordinates(info)
    with np.load(fit['arrays'], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['parameter_names'], model.metadata['parameter_names'])
        point = archive['fit_values'].copy()
        gradient = archive['fit_projected_gradient'].copy()
        physical_gradient = archive['fit_free_gradient'].copy()
        expected = float(archive['fit_value'])
    if args.parameter:
        free_names = [model.metadata['parameter_names'][i] for i in coordinates['free_indices']]
        if len(set(args.parameter)) != len(args.parameter) or any(n not in free_names for n in args.parameter):
            parser.error('Selected parameters must be unique free model parameters')
        selected = np.array([free_names.index(n) for n in args.parameter], dtype=int)
    else:
        selected = np.argsort(np.abs(gradient))[-args.count:][::-1]
    steps = [10.**(-k) for k in range(2, 9)]
    manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
    case = json.loads(manifest_path.read_text())['hhvv']
    assert sha(case['workspace']) == case['workspace_sha256']
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    monitor = Resources(False, args.output)
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        fit=str(args.fit), fit_sha256=sha(args.fit), arrays_sha256=fit['arrays_sha256'],
        source_evidence=source, workspace_sha256=case['workspace_sha256'],
        combine_version='11.0.0', root_version='6.40.00', runtime_flags=REFERENCE_FLAGS,
        step_fractions=steps, selected_free_indices=selected, points=[],
        selection_policy=('Explicit parameter names' if args.parameter else 'Largest absolute projected AD gradients'),
        scope='Native full-NLL central and one-sided FD diagnostics at a saved tensor point. '
              'Steps are fixed before evaluation, scaled by fit coordinates. No gradient '
              'acceptance, fit convergence, curvature or performance claim. Local CPU fit '
              'runs concurrently; this is correctness work only.')

    def save():
        report['resources'] = monitor.snapshot()
        save_json(args.output, report)

    save()
    try:
        with open_workspace(case['workspace']) as workspace:
            for name, configuration in case['configuration'].items():
                workspace.var(name).setVal(configuration['value'])
                workspace.var(name).setConstant(configuration['constant'])
            for choice in model.metadata['provenance']['discrete_choices'].values():
                workspace.cat(choice['category']).setIndex(choice['index'])
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
            variables = [workspace.var(name) for name in model.metadata['parameter_names']]
            for variable, value in zip(variables, point, strict=True):
                variable.setVal(float(value))

            def value():
                root.RooAbsReal.clearEvalErrorLog()
                result = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                assert errors == 0 and np.isfinite(result), (result, errors)
                return result

            canonical = value()
            report['canonical_nll'] = canonical
            report['saved_nll_absolute_error'] = abs(canonical-expected)
            assert abs(canonical-expected) < 1e-6
            nll.setZeroPoint()
            baseline = value()
            assert abs(baseline) < 1e-6
            for j in selected:
                i = int(coordinates['free_indices'][j])
                origin = point[i]
                row = dict(free_index=int(j), physical_index=i,
                    name=model.metadata['parameter_names'][i], coordinate=origin,
                    scale=coordinates['scale'][j], physical_ad_gradient=physical_gradient[j],
                    scaled_projected_ad_gradient=gradient[j], probes=[])
                report['points'].append(row)
                for fraction in steps:
                    h = fraction*row['scale']
                    assert info.arrays['lower'][i] <= origin-2*h < origin < origin+2*h <= info.arrays['upper'][i]
                    deltas = {}
                    for multiple in (-2, -1, 1, 2):
                        variables[i].setVal(float(origin+multiple*h))
                        deltas[multiple] = value()-baseline
                    variables[i].setVal(float(origin))
                    restored = value()-baseline
                    assert abs(restored) < 1e-6
                    row['probes'].append(dict(step=h, deltas=deltas,
                        central=(deltas[1]-deltas[-1])/(2*h),
                        forward=(4*deltas[1]-deltas[2])/(2*h),
                        backward=(-4*deltas[-1]+deltas[-2])/(2*h),
                        restoration_absolute_error=abs(restored)))
                    save()
                print(row['name'], row['probes'], flush=True)
            nll.clearZeroPoint(); nll.clearConstantZeroPoint()
            report['canonical_restoration_absolute_error'] = abs(value()-canonical)
            assert report['canonical_restoration_absolute_error'] < 1e-6
            nll.setAnalyticBarlowBeeston(False)
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='diagnostic_complete', jax_imported=False, torch_imported=False)
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
