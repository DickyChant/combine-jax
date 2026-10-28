"""Compare a common-POI discrete profile envelope with native Combine objectives."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path

import jax
import numpy as np
from iminuit import Minuit

from build_multipdf_fixture import build_multipdf_fixture
from tensor_benchmark_common import Resources, provenance, save_json, sha
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.fitting import prepare_fit
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    root = load_caching_nll()
    grid = np.array([0., .1, .25, .5, .75, 1., 1.5, 2., 3., 3.5, 4., 5.])
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        fixture_script_sha256=sha('scripts/build_multipdf_fixture.py'),
        versions={n:version(n) for n in ('jax','jaxlib','iminuit','higgs-combine-tool')},
        root_version=str(root.gROOT.GetVersion()), float64=True, poi='r', grid=grid, branches=[],
        reference='Native Combine v11 CachingSimNLL, including RooMultiPdf correction, minimized by iminuit separately for both discrete choices.',
        execution='One compiled device fit per branch and fixed-POI layout, reused across all grid points. Each fit starts at the imported nuisance values, with a runtime fixed POI; no host callbacks during optimization.',
        scope='Common-grid discrete profile envelope and branch minimization gate on a fresh generic card. Not confidence-interval endpoints, an HHVV scan, or a performance benchmark.')
    monitor = Resources(False, args.output)
    def save():
        report['resources'] = monitor.snapshot()
        save_json(args.output, report)
    save()
    try:
        workspace_path = build_multipdf_fixture(args.directory)
        report.update(workspace=str(workspace_path), workspace_sha256=sha(workspace_path),
                      card_sha256=sha(workspace_path.parent/'card.txt'))
        for index, label in enumerate(('exponential','bernstein')):
            print('start', label, flush=True)
            with open_workspace(workspace_path) as workspace:
                workspace.cat('pdf_index').setIndex(index)
                for name, error in [('multi_slope',.005),('multi_c1',.1),('multi_c2',.1)]:
                    workspace.var(name).setError(error)
                model = from_workspace(workspace, workspace_path.parent/'card.txt')
                info = model_parameter_info(model)
                names = model.metadata['parameter_names']
                initial = info.arrays['initial']
                artifact = args.directory/(label+'_model'); model.save(artifact)
                native = root.cacheutils.CachingSimNLL(workspace.obj('ModelConfig').GetPdf(),
                    workspace.data('data_obs'), workspace.obj('ModelConfig').GetNuisanceParameters())
                native.setAnalyticBarlowBeeston(True); native.clearConstantZeroPoint(); native.clearZeroPoint()
                def objective(*point):
                    for name, value in zip(names, point, strict=True):
                        workspace.var(name).setVal(float(value))
                    return float(native.getVal())
                ref = Minuit(objective, *initial, name=names)
                ref.errordef=.5; ref.tol=1e-8; ref.strategy=2
                for i,name in enumerate(names):
                    ref.limits[name]=(info.arrays['lower'][i],info.arrays['upper'][i])
                    ref.errors[name]=max(float(info.arrays['error'][i]),1e-3)
                ref.fixed['r']=True
                function, _, arrays = prepare_fit(model, fixed={'r':float(grid[0])},
                    covariance=False, gradient_tolerance=1e-7)
                branch = dict(label=label, index=index, parameters=names, model=str(artifact),
                    model_sha256={n:sha(artifact/n) for n in ('model.json','arrays.npz')}, points=[])
                report['branches'].append(branch);save()
                for poi in grid:
                    coordinates = fit_coordinates(info, fixed={'r':float(poi)})
                    coordinates = jax.device_put({k:coordinates[k] for k in COORDINATE_FIELDS})
                    result = jax.device_get(function(coordinates, arrays))
                    ref.values=initial; ref.values['r']=float(poi)
                    ref.migrad()
                    native_value = float(ref.fval)
                    native_at_device = objective(*result['values'])
                    error = abs(float(result['value'])-native_value)
                    point_error = abs(float(result['value'])-native_at_device)
                    row = dict(poi=float(poi), native_valid=bool(ref.valid), native_edm=float(ref.fmin.edm),
                        native_nll=native_value, native_values=np.array(ref.values),
                        device_status=int(result['status']), device_nll=float(result['value']),
                        device_values=result['values'], iterations=int(result['iteration']),
                        projected_gradient_max=float(np.max(np.abs(result['projected_gradient']),initial=0.)),
                        nll_absolute_error=error, native_at_device_nll=native_at_device,
                        native_at_device_absolute_error=point_error)
                    row['passed']=bool(ref.valid and result['status']==0
                        and row['projected_gradient_max']<=1e-7 and error<1e-6 and point_error<1e-6)
                    branch['points'].append(row);save()
                    print(label, poi, row['passed'], error, row['device_status'], flush=True)
                native.setAnalyticBarlowBeeston(False)
        device = np.array([[p['device_nll'] for p in b['points']] for b in report['branches']])
        native = np.array([[p['native_nll'] for p in b['points']] for b in report['branches']])
        nd, nn = device.min(axis=0), native.min(axis=0)
        report['envelope'] = dict(native_nll=nn, device_nll=nd,
            native_branch=np.argmin(native,axis=0), device_branch=np.argmin(device,axis=0),
            max_nll_absolute_error=float(np.max(np.abs(nd-nn))),
            native_winner_separation=np.abs(native[0]-native[1]),
            delta_nll_reference='Grid minimum, not the unconstrained global minimum.',
            native_delta_nll=nn-nn.min(), device_delta_nll=nd-nd.min())
        report['status']='passed' if all(p['passed'] for b in report['branches'] for p in b['points']) \
            and np.array_equal(np.argmin(native,axis=0),np.argmin(device,axis=0)) \
            and report['envelope']['max_nll_absolute_error']<1e-6 else 'failed_gates'
        assert all(sha(p)==digest for p,digest in report['sources'].items())
        save();print(report['status'], report['envelope'], flush=True)
    except BaseException as exc:
        report.update(status='error', error=repr(exc));save();raise
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
