"""Check a saved tensor-fit candidate with native Combine, without fitting it."""
import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from check_exported_fit_candidate import candidate_errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fit', type=Path, required=True)
    parser.add_argument('--profile-index',type=int,
                        help='Check this point from a saved device profile scan')
    parser.add_argument('--case', default='hhvv')
    parser.add_argument('--manifest',type=Path,default=Path('build/tensor_benchmark_inputs/manifest.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    manifest_path = args.manifest
    case = json.loads(manifest_path.read_text())[args.case]
    from combine_jax.tensor.parameter_info import model_parameter_info
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    from tensor_benchmark_dataset import benchmark_dataset
    from native_fit_model import load_native_fit_model
    model,model_load = load_native_fit_model(case['model'])
    dataset_name = benchmark_dataset(case, model, label=args.case)
    fit = json.loads(args.fit.read_text())
    assert fit['status'] == 'measured_pending_validation' and fit['float64']
    if args.profile_index is None:
        assert fit['operation']=='fit'
    else:
        assert 0<=args.profile_index<len(fit['grid'])
        from combine_jax.tensor.profile_point import configured_profile_model,apply_native_profile_point
        model=configured_profile_model(model,{fit['parameter']:fit['grid'][args.profile_index]})
        assert fit['dataset']==dataset_name
    info = model_parameter_info(model)
    assert sha(fit['arrays']) == fit['arrays_sha256']
    assert fit['model_sha256'] == {n:sha(Path(case['model'])/n) for n in ('model.json','arrays.npz')}
    workspace_sha256 = (case['workspace_sha256'] if 'workspace_sha256' in case
                        else case['hashes'][case['workspace']])
    assert sha(case['workspace']) == workspace_sha256
    with np.load(fit['arrays'], allow_pickle=False) as archive:
        np.testing.assert_array_equal(archive['parameter_names'], model.metadata['parameter_names'])
        if args.profile_index is None:
            point = archive['fit_values'].copy()
            expected = float(archive['fit_value'])
            status = int(archive['fit_status'])
            covariance_status = int(archive['fit_covariance_status'])
        else:
            np.testing.assert_array_equal(archive['grid'],fit['grid'])
            assert archive['values'].shape==(len(fit['grid']),len(info.metadata['names']))
            index=args.profile_index
            point=archive['values'][index].copy()
            expected=float(archive['value'][index])
            status=int(archive['status'][index]);covariance_status=None
    assert point.shape == info.arrays['initial'].shape and np.isfinite(point).all()
    assert np.all(point >= info.arrays['lower']) and np.all(point <= info.arrays['upper'])
    fixed = info.arrays['lower'] == info.arrays['upper']
    np.testing.assert_array_equal(point[fixed], info.arrays['initial'][fixed])
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    monitor = Resources(False, args.output)
    report = dict(provenance(), status='checking', script_sha256=sha(__file__), fit=str(args.fit),
        fit_sha256=sha(args.fit), arrays_sha256=fit['arrays_sha256'], optimizer_status=status,
        covariance_status=covariance_status, manifest_sha256=sha(manifest_path),
        workspace_sha256=workspace_sha256, combine_version='11.0.0', root_version='6.40.00',
        dataset=dataset_name, runtime_flags=REFERENCE_FLAGS, parameter_names=model.metadata['parameter_names'], point=point,
        saved_tensor_nll=expected, probes=[], scope='Native same-point and zero-point/restoration checks only. No convergence, covariance or successful-fit timing acceptance.')
    report.update(native_fit_model_load=model_load,profile_index=args.profile_index,
                  profile_point=model.metadata.get('profile_point'))
    def save():
        report['resources'] = monitor.snapshot()
        save_json(args.output, report)
    save()
    try:
        with open_workspace(case['workspace']) as workspace, ExitStack() as stack:
            for name, configuration in case['configuration'].items():
                workspace.var(name).setVal(configuration['value'])
                workspace.var(name).setConstant(configuration['constant'])
            for choice in model.metadata['provenance']['discrete_choices'].values():
                workspace.cat(choice['category']).setIndex(choice['index'])
            if args.profile_index is not None:
                apply_native_profile_point(root,workspace,model)
            if 'fit_policy' in info.metadata:
                from combine_jax.tensor.fit_policy import check_native_fit_state, combine_v11_fit_context
                report['fit_state']=check_native_fit_state(workspace,model)
                stack.enter_context(combine_v11_fit_context(workspace))
            config = workspace.obj('ModelConfig')
            nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data(dataset_name), config.GetNuisanceParameters())
            if config.GetPdf().ClassName() == 'RooSimultaneousOpt':
                nll.setChannelMasks(config.GetPdf().channelMasks())
            nll.setAnalyticBarlowBeeston(True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
            variables = [workspace.var(name) for name in model.metadata['parameter_names']]
            def evaluate(theta, label):
                for variable, value in zip(variables, theta, strict=True):
                    variable.setVal(float(value))
                root.RooAbsReal.clearEvalErrorLog()
                value = float(nll.getVal())
                errors = int(root.RooAbsReal.numEvalErrors())
                report['probes'].append(dict(label=label, value=value, roofit_errors=errors))
                save()
                assert errors == 0 and np.isfinite(value)
                return value
            nominal = evaluate(info.arrays['initial'], 'nominal_canonical')
            nll.setZeroPoint()
            zero = evaluate(point, 'candidate_zero_point')
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            canonical = evaluate(point, 'candidate_canonical')
            restored = evaluate(info.arrays['initial'], 'restored_nominal_canonical')
            nll.setAnalyticBarlowBeeston(False)
            del nll
        errors, passed = candidate_errors(nominal, zero, canonical, restored, expected)
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='passed_point_only' if passed else 'failed', errors=errors,
                      native_nll=canonical, jax_imported=False, torch_imported=False)
        save()
        print(report['status'], errors, flush=True)
        assert passed
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc()); save(); raise
    finally:
        monitor.close(); save()


if __name__ == '__main__':
    main()
