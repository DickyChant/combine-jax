"""Check every saved tensor interval fit point with native Combine v11."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tensor', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    tensor = read(args.tensor)
    assert tensor['status'] == 'passed' and tensor['mode'] == 'tensor'
    native = read(tensor['native_reference'])
    assert sha(tensor['native_reference']) == tensor['native_reference_sha256']
    assert native['status'] == 'passed' and native['mode'] == 'native'
    assert not native['jax_imported'] and not native['torch_imported']
    for capture in (tensor, native):
        assert capture['script_sha256'] == sha('scripts/validate_tensor_profile_endpoints.py')
        assert all(sha(p) == digest for p, digest in capture['sources'].items())
        assert capture['manifest_sha256'] == sha(capture['manifest'])
        assert capture['target_delta_nll'] == [.5, 2.]
    from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == native['root_version'] == '6.40.00'
    assert version('higgs-combine-tool') == native['combine_version'] == '11.0.0'
    assert REFERENCE_FLAGS == native['runtime_flags']
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        tensor=str(args.tensor), tensor_sha256=sha(args.tensor),
        native_reference=tensor['native_reference'], native_reference_sha256=tensor['native_reference_sha256'],
        root_version=str(root.gROOT.GetVersion()), combine_version=version('higgs-combine-tool'),
        branches=[], endpoints=[],
        scope='Native full-NLL re-evaluation at every tensor baseline/root-search nuisance-fit point, '
              'plus independent crossing/boundary and branch-switch comparisons. No native CLI fit, '
              'HHVV profile, general global-minimum or performance acceptance.')
    monitor = Resources(False, args.output)
    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)
    save()
    try:
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CollectErrors)
        for label, branch in tensor['branches'].items():
            expected = native['branches'][label]
            for field in ('model', 'model_sha256', 'workspace', 'workspace_sha256', 'parameters', 'bounds'):
                assert branch[field] == expected[field]
            assert sha(branch['workspace']) == branch['workspace_sha256']
            assert branch['model_sha256'] == {name: sha(Path(branch['model'])/name) for name in ('model.json', 'arrays.npz')}
            model = TensorBundle.load(branch['model']); info = model_parameter_info(model)
            with open_workspace(branch['workspace']) as workspace:
                for choice in model.metadata['provenance']['discrete_choices'].values():
                    workspace.cat(choice['category']).setIndex(choice['index'])
                config = workspace.obj('ModelConfig')
                nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
                nll.setAnalyticBarlowBeeston(True); nll.clearConstantZeroPoint(); nll.clearZeroPoint()
                variables = [workspace.var(name) for name in branch['parameters']]
                row = dict(label=label, points=[]); report['branches'].append(row)
                for i, point in enumerate([branch['minimum'], *branch['evaluations']]):
                    values = np.array(point['values'])
                    assert point['status'] == 0 and point['projected_gradient_max'] <= 1e-7
                    assert np.isfinite(values).all() and np.all(values >= info.arrays['lower']) and np.all(values <= info.arrays['upper'])
                    if i: assert values[branch['poi_index']] == point['poi']
                    for variable, value in zip(variables, values, strict=True): variable.setVal(float(value))
                    root.RooAbsReal.clearEvalErrorLog()
                    actual = float(nll.getVal()); errors = int(root.RooAbsReal.numEvalErrors())
                    error = abs(actual-point['value'])
                    assert errors == 0 and np.isfinite(actual) and error < 1e-6
                    row['points'].append(dict(index=i, native_nll=actual, tensor_nll=point['value'],
                        nll_absolute_error=error, native_evaluation_errors=errors))
                nll.setAnalyticBarlowBeeston(False)
                save(); print(label, len(row['points']), max(p['nll_absolute_error'] for p in row['points']), flush=True)
        for label, calculated in tensor['intervals'].items():
            expected = native['intervals'][label]
            assert abs(calculated['minimum']-expected['minimum']) < 1e-6
            for a, b in zip(calculated['levels'], expected['levels'], strict=True):
                assert a['target'] == b['target']
                for x, y in zip(a['endpoints'], b['endpoints'], strict=True):
                    assert (x['side'], x['kind'], x['selected_branch']) == (y['side'], y['kind'], y['selected_branch'])
                    assert abs(x['poi']-y['poi']) < 1e-5
                    assert abs(x['delta_nll']-y['delta_nll']) < 1e-6
                    for endpoint, interval in ((x, calculated), (y, expected)):
                        direction = -1 if endpoint['side'] == 'lower' else 1
                        assert direction*(endpoint['poi']-interval['center']) >= 0
                        if endpoint['kind'] == 'crossing':
                            assert abs(endpoint['delta_nll']-a['target']) < 1e-6
                        else:
                            assert endpoint['kind'] == 'physical_bound' and endpoint['delta_nll'] < a['target']
                            bound = tensor['branches'][endpoint['selected_branch']]['bounds'][direction > 0]
                            assert endpoint['poi'] == bound
                    report['endpoints'].append(dict(case=label, target=a['target'], side=x['side'], kind=x['kind'],
                        native_poi=y['poi'], tensor_poi=x['poi'], absolute_error=abs(x['poi']-y['poi'])))
        switch_error = abs(tensor['multipdf_switch']['poi']-native['multipdf_switch']['poi'])
        assert switch_error < 1e-5
        for capture in (tensor, native):
            assert abs(capture['multipdf_switch']['branch_nll_difference']) < 1e-6
            assert capture['multipdf_switch']['bracket'][0] <= capture['multipdf_switch']['poi'] <= capture['multipdf_switch']['bracket'][1]
        assert 'jax' not in sys.modules and 'torch' not in sys.modules
        report.update(status='passed', jax_imported=False, torch_imported=False,
            points_checked=sum(len(b['points']) for b in report['branches']),
            maximum_native_nll_absolute_error=max(p['nll_absolute_error'] for b in report['branches'] for p in b['points']),
            maximum_endpoint_absolute_error=max(p['absolute_error'] for p in report['endpoints']),
            multipdf_switch_absolute_error=switch_error)
        save(); print('passed', report['maximum_endpoint_absolute_error'], switch_error, flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); save(); raise
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
