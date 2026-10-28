"""Independent native and tensor profile crossings on the generic cards.

The root search is a validation harness on the host. Tensor nuisance fits use
the public compiled device scan, reused with runtime coordinates. This is not
a new device interval API or a timing benchmark.
"""
import argparse
from contextlib import ExitStack
from importlib.metadata import version
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import brentq

from tensor_benchmark_common import Resources, provenance, save_json, sha
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS


LABELS = ('counting', 'automcstats', 'parametric', 'multipdf_exp', 'multipdf_bern')


def endpoints(evaluate, minimum, center, lower, upper, target):
    """First crossing on each side, or the physical boundary if below target."""
    result = []
    assert lower <= center <= upper
    assert abs(evaluate(center)['value']-minimum) < 1e-6
    for sign, bound in ((-1, lower), (1, upper)):
        inside, step = center, min(.25, (upper-lower)/8)
        for _ in range(32):
            outside = float(np.clip(center+sign*step, lower, upper))
            value = evaluate(outside)['value']-minimum-target
            if value >= 0:
                a, b = sorted((inside, outside))
                crossing, solver = brentq(lambda x: evaluate(x)['value']-minimum-target,
                    a, b, xtol=1e-9, rtol=1e-12, full_output=True)
                assert solver.converged
                row = evaluate(crossing)
                residual = row['value']-minimum-target
                assert abs(residual) < 1e-6
                result.append(dict(side='lower' if sign < 0 else 'upper', kind='crossing',
                    poi=crossing, bracket=[a, b], delta_nll=row['value']-minimum,
                    root_residual=residual, selected_branch=row['selected_branch']))
                break
            if outside == bound:
                result.append(dict(side='lower' if sign < 0 else 'upper', kind='physical_bound',
                    poi=outside, delta_nll=value+target, selected_branch=evaluate(outside)['selected_branch']))
                break
            inside, step = outside, step*2
        else:
            raise RuntimeError('Could not bracket profile crossing')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('native', 'tensor'), required=True)
    parser.add_argument('--native-reference', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    if (args.mode == 'tensor') != bool(args.native_reference):
        parser.error('--native-reference is required only for tensor mode')
    manifest_path = Path('build/tensor_benchmark_inputs/manifest.json')
    manifest = json.loads(manifest_path.read_text())
    reference = json.loads(args.native_reference.read_text()) if args.native_reference else None
    if reference:
        assert reference['status'] == 'passed' and reference['mode'] == 'native'
        assert reference['script_sha256'] == sha(__file__)
        assert reference['manifest_sha256'] == sha(manifest_path)
    report = dict(provenance(), status='running', mode=args.mode, script_sha256=sha(__file__),
        manifest=str(manifest_path), manifest_sha256=sha(manifest_path),
        target_delta_nll=[.5, 2.], root_xtol=1e-9, root_rtol=1e-12,
        branches={}, intervals={},
        scope='Connected profile interval around the fitted minimum on counting, autoMCStats, '
              'parametric and a two-branch MultiPdf envelope. Independent native/tensor root searches; '
              'native uses Combine v11 CachingSimNLL and iminuit. This validates the public tensor '
              'scan at confidence endpoints, not a device root-search API, HHVV scan, global '
              'minimality for arbitrary models, CLI fit, or performance.')
    if reference:
        report.update(native_reference=str(args.native_reference), native_reference_sha256=sha(args.native_reference))
    monitor = Resources(False, args.output)
    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)
    save()
    try:
        if args.mode == 'native':
            from iminuit import Minuit
            from combine_jax.root_io import load_caching_nll, open_workspace, REFERENCE_FLAGS
            root = load_caching_nll()
            assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
            report.update(root_version=str(root.gROOT.GetVersion()), combine_version=version('higgs-combine-tool'),
                          iminuit_version=version('iminuit'), runtime_flags=REFERENCE_FLAGS)
        else:
            import jax
            from combine_jax.tensor.fitting import prepare_fit
            from combine_jax.tensor.profiling import prepare_profile_scan
            from combine_jax.tensor.model import make_model_function
            assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
            report.update(jax_version=jax.__version__, float64=True, gradient_tolerance=1e-7)
        evaluators, models, infos = {}, {}, {}
        with ExitStack() as stack:
            for label in LABELS:
                case = manifest[label]
                for path, digest in case['hashes'].items():
                    assert sha(path) == digest
                model = TensorBundle.load(case['model']); info = model_parameter_info(model)
                names = info.metadata['names']; index = names.index('r')
                models[label], infos[label] = model, info
                row = dict(model=case['model'], model_sha256={n: sha(Path(case['model'])/n) for n in ('model.json', 'arrays.npz')},
                    workspace=case['workspace'], workspace_sha256=sha(case['workspace']), parameters=names,
                    poi_index=index, bounds=[info.arrays['lower'][index], info.arrays['upper'][index]], evaluations=[])
                report['branches'][label] = row; save()
                if args.mode == 'native':
                    workspace = stack.enter_context(open_workspace(case['workspace']))
                    for name, configuration in case['configuration'].items():
                        workspace.var(name).setVal(configuration['value']); workspace.var(name).setConstant(configuration['constant'])
                    for choice in model.metadata['provenance']['discrete_choices'].values():
                        workspace.cat(choice['category']).setIndex(choice['index'])
                    config = workspace.obj('ModelConfig')
                    nll = root.cacheutils.CachingSimNLL(config.GetPdf(), workspace.data('data_obs'), config.GetNuisanceParameters())
                    nll.setAnalyticBarlowBeeston(True); nll.clearConstantZeroPoint(); nll.clearZeroPoint()
                    variables = [workspace.var(name) for name in names]
                    def objective(*point, nll=nll, variables=variables):
                        for variable, value in zip(variables, point, strict=True): variable.setVal(float(value))
                        return float(nll.getVal())
                    ref = Minuit(objective, *info.arrays['initial'], name=names)
                    ref.errordef = .5; ref.tol = 1e-8; ref.strategy = 2
                    for i, name in enumerate(names):
                        ref.limits[name] = (info.arrays['lower'][i], info.arrays['upper'][i])
                        ref.errors[name] = max(float(info.arrays['error'][i]), 1e-3)
                    ref.migrad(); assert ref.valid
                    minimum = dict(value=float(ref.fval), values=np.array(ref.values), valid=bool(ref.valid), edm=float(ref.fmin.edm))
                    initial = np.array(ref.values)
                    ref.fixed['r'] = True
                    def evaluate(poi, ref=ref, initial=initial, row=row, objective=objective):
                        ref.values = initial; ref.values['r'] = float(poi); ref.migrad()
                        assert ref.valid
                        result = dict(poi=float(poi), value=float(ref.fval), values=np.array(ref.values), valid=bool(ref.valid), edm=float(ref.fmin.edm))
                        assert abs(objective(*result['values'])-result['value']) < 1e-8
                        row['evaluations'].append(result)
                        return result
                else:
                    function, c, arrays = prepare_fit(model, covariance=False, gradient_tolerance=1e-7)
                    fitted = jax.device_get(function(c, arrays))
                    assert fitted['status'] == 0
                    minimum = dict(value=float(fitted['value']), values=fitted['values'], status=int(fitted['status']),
                        projected_gradient_max=float(np.max(abs(fitted['projected_gradient']), initial=0.)))
                    initial = fitted['values']
                    scan, c, arrays = prepare_profile_scan(model, 'r', [float(initial[index])], initial=initial, gradient_tolerance=1e-7)
                    compiled = scan.lower(c, arrays).compile()
                    independent = jax.jit(jax.value_and_grad(make_model_function(model)))
                    def evaluate(poi, compiled=compiled, arrays=arrays, info=info, initial=initial, row=row, independent=independent):
                        coords = fit_coordinates(info, initial=initial, fixed={'r':float(poi)})
                        runtime = jax.device_put({k: np.expand_dims(coords[k],0) for k in COORDINATE_FIELDS})
                        got = jax.device_get(compiled(runtime, arrays))
                        assert got['status'][0] == 0
                        point = got['values'][0]
                        value, gradient = map(np.asarray, independent(point, arrays))
                        free, scale = coords['free_indices'], coords['scale']; g = gradient[free]*scale
                        z = (point[free]-coords['reference'][free])/scale
                        eps = 32*np.finfo(float).eps*(1+abs(z))
                        blocked = ((z <= coords['lower']+eps)&(g > 0)) | ((z >= coords['upper']-eps)&(g < 0))
                        pg = np.where(blocked, 0., g)
                        assert np.max(abs(pg), initial=0.) <= 1e-7
                        np.testing.assert_allclose(pg, got['projected_gradient'][0], rtol=1e-5, atol=2e-10)
                        assert abs(float(value)-got['value'][0]) < 1e-9
                        result = dict(poi=float(poi), value=float(value), values=point, status=int(got['status'][0]),
                            projected_gradient_max=float(np.max(abs(pg), initial=0.)), iterations=int(got['iteration'][0]))
                        row['evaluations'].append(result)
                        return result
                row['minimum'] = minimum
                cache = {}
                def cached(poi, evaluate=evaluate, cache=cache):
                    key = float(poi)
                    if key not in cache: cache[key] = evaluate(key)
                    return cache[key]
                evaluators[label] = cached
                print(args.mode, label, 'minimum', minimum['value'], flush=True); save()
            for name, labels in [('counting', ['counting']), ('automcstats', ['automcstats']),
                                 ('parametric', ['parametric']), ('multipdf', ['multipdf_exp', 'multipdf_bern'])]:
                branches = [report['branches'][label] for label in labels]
                selected = min(branches, key=lambda b:b['minimum']['value'])
                minimum = selected['minimum']['value']; center = float(selected['minimum']['values'][selected['poi_index']])
                lower, upper = selected['bounds']; assert all(b['bounds'] == selected['bounds'] for b in branches)
                def envelope(poi):
                    values = [evaluators[label](poi)['value'] for label in labels]
                    winner = int(np.argmin(values))
                    return dict(value=values[winner], selected_branch=labels[winner])
                intervals = []
                report['intervals'][name] = dict(branches=labels, minimum=minimum, center=center, levels=intervals)
                for target in report['target_delta_nll']:
                    intervals.append(dict(target=target, endpoints=endpoints(envelope, minimum, center, lower, upper, target)))
                    save()
                    print(args.mode, name, 'delta', target, intervals[-1]['endpoints'], flush=True)
            # Independently locate the known branch switch rather than infer its
            # position from the old grid. The fixed bracket predates this run.
            difference = lambda x: evaluators['multipdf_exp'](x)['value']-evaluators['multipdf_bern'](x)['value']
            switch = brentq(difference, 3.5, 4., xtol=1e-9, rtol=1e-12)
            report['multipdf_switch'] = dict(poi=switch, bracket=[3.5, 4.], branch_nll_difference=difference(switch))
            assert abs(report['multipdf_switch']['branch_nll_difference']) < 1e-6
        if args.mode == 'native':
            assert 'jax' not in sys.modules and 'torch' not in sys.modules
            report.update(jax_imported=False, torch_imported=False)
        else:
            checks = []
            for name, calculated in report['intervals'].items():
                expected = reference['intervals'][name]
                assert abs(calculated['minimum']-expected['minimum']) < 1e-6
                for a, b in zip(calculated['levels'], expected['levels'], strict=True):
                    assert a['target'] == b['target']
                    for x, y in zip(a['endpoints'], b['endpoints'], strict=True):
                        assert (x['side'], x['kind'], x['selected_branch']) == (y['side'], y['kind'], y['selected_branch'])
                        delta = abs(x['poi']-y['poi'])
                        assert delta < 1e-5 and abs(x['delta_nll']-y['delta_nll']) < 1e-6
                        checks.append(dict(case=name, target=a['target'], side=x['side'], kind=x['kind'], endpoint_absolute_error=delta))
            report['endpoint_checks'] = checks
            report['switch_absolute_error'] = abs(report['multipdf_switch']['poi']-reference['multipdf_switch']['poi'])
            assert report['switch_absolute_error'] < 1e-5
        assert all(sha(path) == digest for path, digest in report['sources'].items())
        report['status'] = 'passed'; save(); print('passed', args.mode, flush=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); save(); raise
    finally:
        monitor.close()


if __name__ == '__main__':
    main()
