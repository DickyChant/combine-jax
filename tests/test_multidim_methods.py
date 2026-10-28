"""Method semantics on analytic and imported likelihoods before native law gates."""
import numpy as np
import pytest

from combine_jax.tensor.multidim_profile import MultiDimProfiler
from combine_jax.tensor.model import from_workspace
from combine_jax.root_io import open_workspace
from combine_jax.methods.multidim import grid_points, interval, impacts, scan, DELTA68


def test_multidim_profiles_reuse_executable_and_restore_global(counting_path):
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace, fit_policy='combine_v11_default')
    floating = [n for n, lo, hi in zip(model.metadata['parameter_names'],
        model.arrays['parameter_info__lower'], model.arrays['parameter_info__upper']) if lo != hi]
    nuisance = next(n for n in floating if n != 'r')
    driver = MultiDimProfiler(model, conditional_parameters=[('r',), (nuisance,), ('r', nuisance)])
    global_result = driver.global_minimum()
    global_point = np.asarray(driver.global_point).copy()
    executable = driver.executables['conditional_1']
    first_point, first = driver.profile({'r': .4})
    driver.profile({nuisance: float(global_point[driver.names.index(nuisance)])+.3})
    repeated_point, repeated = driver.profile({'r': .4})
    assert driver.executables['conditional_1'] is executable
    np.testing.assert_array_equal(first_point, repeated_point)
    assert first['value'] == repeated['value']
    np.testing.assert_array_equal(driver.global_point, global_point)
    point, result = driver.profile({'r': .8, nuisance: float(global_point[driver.names.index(nuisance)])})
    assert float(point[driver.names.index('r')]) == .8
    assert result['value'] >= global_result['value']-1e-4
    assert all(r['status'] == 0 for r in driver.records)
    assert max(m['total_bytes'] for m in driver.compiler_memory.values()) < 8*2**30
    with pytest.raises(ValueError, match='outside bounds'):
        driver.profile({'r': 1e100})


def test_grid_order_selection_and_standard_output(tmp_path):
    import uproot
    from combine_jax.methods.multidim_output import write_multidim_tree
    points = grid_points(['r', 's'], {'r': [-1, 1], 's': [2, 8]}, [3, 4], 2, 5)
    assert points == [{'r': -1., 's': 6.}, {'r': -1., 's': 8.},
                      {'r': 0., 's': 2.}, {'r': 0., 's': 4.}]
    rows = [dict(parameters={'r': 0., 's': 5.}, deltaNLL=0., quantile=-1., nll0=12., nll=-3.)]
    rows += [dict(parameters=p, deltaNLL=.7, quantile=.3, nll0=12., nll=-3.) for p in points]
    target = tmp_path/'scan.root'
    write_multidim_tree(target, rows, seed=2)
    with target.open('rb') as stream, uproot.open(stream) as handle:
        assert handle['limit'].classname == 'TTree'
        values = handle['limit'].arrays(library='np')
        assert values['nll0'].dtype == np.float64
        assert values['deltaNLL'].dtype == values['r'].dtype == np.float32
        assert values['quantileExpected'][0] == -1
        np.testing.assert_array_equal(values['r'][1:], [p['r'] for p in points])


class QuadraticProfiler:
    """Independent analytic Gaussian oracle for method and output conventions."""
    names = ['r', 'theta']
    global_point = np.array([1., .2])
    global_value = 3.
    initial_nll = 5.
    records = []
    class info:
        arrays = dict(error=np.ones(2), lower=np.array([-10., -10.]), upper=np.array([10., 10.]))

    def profile(self, fixed):
        i = self.names.index(next(iter(fixed)))
        value = next(iter(fixed.values()))
        point = self.global_point.copy()
        delta = value-point[i]
        point[i] = value
        point[1-i] += .4*delta
        record = dict(value=3.+.5*delta**2, status=0)
        self.records.append(record)
        return point, record


def test_profile_intervals_and_impact_rows_follow_native_up():
    oracle = QuadraticProfiler()
    result = impacts(oracle, 'r', ['r', 'theta'], nll0=7.)
    width = 1.
    for name, center in [('r', 1.), ('theta', .2)]:
        bounds = result[name]['interval']
        assert bounds['lower']['value'] == pytest.approx(center-width, abs=1e-5)
        assert bounds['upper']['value'] == pytest.approx(center+width, abs=1e-5)
        assert all(r['nll'] == -2. for r in result[name]['rows'])
    theta_rows = result['theta']['rows']
    assert theta_rows[1]['parameters']['r'] == pytest.approx(1.-.4*width, abs=1e-5)
    assert theta_rows[2]['parameters']['r'] == pytest.approx(1.+.4*width, abs=1e-5)
    assert [r['quantile'] for r in theta_rows] == [-1., .32, .32]
    assert [r['quantile'] for r in result['r']['rows']] == [-1., -.32, .32]
    assert all(r['deltaNLL'] == 0. for r in result['r']['rows'])
    robust = interval(oracle, 'r', delta_nll=DELTA68)
    assert robust['upper']['value'] == pytest.approx(1.+np.sqrt(2*DELTA68), abs=1e-5)


def test_multidim_cli_requires_explicit_asimov():
    import argparse
    from combine_jax.runtime.multidim_options import add_multidim_command, request
    parser = argparse.ArgumentParser()
    add_multidim_command(parser.add_subparsers(dest='command'))
    base = ['multidim', '-d', 'model.root', '--poi', 'r', '--algo', 'grid',
            '--gridPoints', '3', '--setParameterRanges', 'r=-2,2']
    with pytest.raises(SystemExit):
        parser.parse_args(base)
    with pytest.raises(ValueError, match='Explicit nominal'):
        request(parser.parse_args(base+['-t', '-1']))
    config = request(parser.parse_args(base+['-t', '-1', '--setParameters', 'r=1']))
    assert config['signal_strength'] == 1. and config['float64']
    assert config['points'] == [{'r': -2.}, {'r': 0.}, {'r': 2.}]


def test_promoted_nuisance_constraint_matches_native_nll(counting_path):
    import jax
    from combine_jax.root_io import load_caching_nll
    from combine_jax.tensor.model import make_model_function
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        mc = workspace.obj('ModelConfig')
        workspace.pdf('bkg_Pdf').setAttribute('ignoreConstraint')
        pois = root.RooArgSet(workspace.var('r'), workspace.var('bkg'))
        mc.SetParametersOfInterest(pois)
        nuis = root.RooArgSet(mc.GetNuisanceParameters()); nuis.remove(pois)
        mc.SetNuisanceParameters(nuis)
        model = from_workspace(workspace, fit_policy='combine_v11_default')
        assert 'bkg_Pdf' not in model.metadata['constraint_names']
        native = root.cacheutils.CachingSimNLL(mc.GetPdf(), workspace.data('data_obs'), mc.GetNuisanceParameters())
        native.setAnalyticBarlowBeeston(True); native.clearConstantZeroPoint(); native.clearZeroPoint()
        function = jax.jit(make_model_function(model, binned_gradient='structural_sequential'))
        names = model.metadata['parameter_names']
        initial = model.arrays['parameter_info__initial']
        for r, bkg in [(0., -.4), (.7, .2), (2., 1.1)]:
            theta = initial.copy(); theta[names.index('r')] = r; theta[names.index('bkg')] = bkg
            for n, value in zip(names, theta): workspace.var(n).setVal(float(value))
            assert float(function(theta, dict(model.arrays))) == pytest.approx(native.getVal(), abs=1e-10)


def test_flat_global_direction_can_use_explicit_estimated_policy(counting_path, monkeypatch):
    from combine_jax.tensor import model as model_module
    with open_workspace(counting_path) as workspace:
        model = from_workspace(workspace)
    names = model.metadata['parameter_names']
    r, lumi = names.index('r'), names.index('lumi')
    def objective(theta, arrays):
        return .5*(theta[r]-1.)**2+.5*theta[lumi]**2
    monkeypatch.setattr(model_module, 'make_model_function', lambda *a, **k: objective)
    driver = MultiDimProfiler(model, conditional_parameters=[('r', 'bkg')],
        edm_validation='exact_hessian', global_edm_validation='estimated')
    result = driver.global_minimum()
    assert result['status'] == 0 and not result['edm_curvature_validated']
    _, result = driver.profile({'r': 2., 'bkg': .2})
    assert result['status'] == 0 and result['edm_curvature_validated']
    assert result['value'] == pytest.approx(.5)
