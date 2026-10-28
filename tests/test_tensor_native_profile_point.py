from pathlib import Path
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.fit_policy import POLICY, check_native_fit_state, combine_v11_fit_context
from combine_jax.tensor.model import from_workspace, evaluate_model_numpy
from combine_jax.tensor.parameter_info import fit_coordinates, model_parameter_info
from combine_jax.tensor.profile_point import configured_profile_model, apply_native_profile_point
from test_bin_statistics import template_path


CARD=Path(__file__).parent/'cards/counting.txt'


@pytest.mark.parametrize('value',[-1.,.5,1.])
def test_profile_view_preserves_likelihood_and_native_state(counting_path,value):
    root=load_caching_nll()
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        info=model_parameter_info(model);index=info.metadata['names'].index('lumi')
        profile=configured_profile_model(model,{'lumi':value})
        point=model_parameter_info(profile)
        assert profile.structure_key==configured_profile_model(model,{'lumi':0.}).structure_key
        assert profile.metadata['provenance']==model.metadata['provenance']
        assert 'profile_point' not in model.metadata
        assert info.arrays['initial'][index]==0. and info.arrays['lower'][index]<info.arrays['upper'][index]
        for name,array in model.arrays.items():
            if not name.startswith(('parameter_info__','scalars__')):
                assert profile.arrays[name] is array
        assert index not in fit_coordinates(point)['free_indices']
        assert point.arrays['initial'][index]==point.arrays['lower'][index]==point.arrays['upper'][index]==value
        apply_native_profile_point(root,workspace,profile)
        check_native_fit_state(workspace,profile)
        with combine_v11_fit_context(workspace):
            config=workspace.obj('ModelConfig')
            native=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('data_obs'),config.GetNuisanceParameters())
            native.clearConstantZeroPoint();native.clearZeroPoint()
            expected=evaluate_model_numpy(model,point.arrays['initial'])
            actual=evaluate_model_numpy(profile,point.arrays['initial'])
            assert abs(actual-expected)<1e-12 and abs(actual-native.getVal())<1e-10
            del native


@pytest.mark.parametrize('fixed,match',[
    ({},'at least one'),({'missing':0.},'Unknown fixed'),({'lumi':float('nan')},'outside bounds'),
    ({'r':-1.},'outside bounds'),({'lumi':100.},'stored workspace bounds'),
])
def test_invalid_profile_configuration_is_explicit(counting_path,fixed,match):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        with pytest.raises(ValueError,match=match):configured_profile_model(model,fixed)


def test_imported_constants_cannot_move_and_policy_is_required(counting_path):
    with open_workspace(counting_path) as workspace:
        legacy=from_workspace(workspace,CARD)
        with pytest.raises(ValueError,match='explicit fit policy'):
            configured_profile_model(legacy,{'lumi':1.})
        workspace.var('lumi').setConstant(True)
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        with pytest.raises(ValueError,match='Unknown fixed parameters'):
            configured_profile_model(model,{'lumi':1.})
        workspace.var('bkg').setAttribute('optimizeBounds',False)
        workspace.var('bkg').setRange(0.,0.)
        model=from_workspace(workspace,CARD,fit_policy=POLICY)
        assert 'bkg' in model_parameter_info(model).metadata['names']
        with pytest.raises(ValueError,match='Cannot change imported fixed'):
            configured_profile_model(model,{'bkg':1.})


def test_native_profile_boundary_handles_absent_nuisances_and_background_config():
    root=load_caching_nll();workspace=root.RooWorkspace('profile_boundary')
    workspace.factory('r[1,0,10]')
    for name in ('ModelConfig','ModelConfig_bonly'):
        config=root.RooStats.ModelConfig(name,workspace)
        config.SetParametersOfInterest(root.RooArgSet(workspace.var('r')))
        getattr(workspace,'import')(config)
    view=SimpleNamespace(metadata={'profile_point':{'fixed':{'r':1.2}}})
    apply_native_profile_point(root,workspace,view)
    assert workspace.var('r').isConstant() and workspace.var('r').getVal()==1.2
    assert not workspace.obj('ModelConfig').GetNuisanceParameters()
    assert not workspace.obj('ModelConfig_bonly').GetNuisanceParameters()


@pytest.mark.parametrize('kind',['counting','automcstats'])
def test_actual_combine_profile_cli(counting_path,template_path,tmp_path,kind):
    path=template_path if kind=='automcstats' else counting_path
    card=Path(path).parent/'card.txt' if kind=='automcstats' else CARD
    with open_workspace(path) as workspace:
        model=from_workspace(workspace,card,fit_policy=POLICY)
        model.save(tmp_path/'model')
    info=model_parameter_info(model)
    name='r' if kind=='automcstats' else 'lumi';value=1.2 if name=='r' else .5
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({kind:dict(workspace=str(path),model=str(tmp_path/'model'),
        configuration={},dataset='data_obs',hashes={})}))
    output=tmp_path/'profile.json'
    command=[sys.executable,'scripts/benchmark_combine_fit.py','--manifest',str(manifest),'--case',kind,
        '--fixed-parameter',name,str(value),'--directory',str(tmp_path/'fit'),'--output',str(output)]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1500:]+completed.stderr[-1500:]
    result=json.loads(output.read_text())
    assert result['status']=='measured_fit_pending_agreement' and result['preflight']['status']=='passed'
    assert result['profile_point']['fixed']=={name:value}
    assert result['preflight']['initial_state']['fixed_real_values'][name]==value
    assert result['fit']['values'][info.metadata['names'].index(name)]==value
    assert result['fit']['errors'][info.metadata['names'].index(name)]==0.
    assert result['fit']['edm_target_met'] and result['fit']['roofit_evaluation_errors']==0
    assert result['evaluation_counts']['actual_nll_evaluations']>0
    assert abs(evaluate_model_numpy(model,result['fit']['values'])-result['fit']['canonical_nll'])<1e-10
    import jax
    from combine_jax.tensor.profiling import prepare_profile_scan
    function,coordinates,arrays=prepare_profile_scan(model,name,[value],stopping_criterion='edm',
        edm_validation='exact_hessian',max_iterations=2000)
    fitted=jax.device_get(function(coordinates,arrays))
    assert fitted['status'][0]==0 and fitted['edm_curvature_validated'][0]
    native_values=np.asarray(result['fit']['values']);sigma=np.asarray(result['fit']['errors'])
    free=fit_coordinates(info,fixed={name:value})['free_indices']
    delta=abs(fitted['values'][0]-native_values)
    assert np.all(delta[free]<=np.maximum(.01*abs(native_values[free]),.01*sigma[free]))
    nll_delta=float(fitted['value'][0]-result['fit']['canonical_nll'])
    assert abs(nll_delta)<=2e-4
    assert fitted['values'][0,info.metadata['names'].index(name)]==value
    jax.clear_caches()
    print('NATIVE_PROFILE_CLI',json.dumps(dict(case=kind,parameter=name,fixed_value=value,
        edm=result['fit']['edm'],nll=result['fit']['canonical_nll'],counts=result['evaluation_counts'],
        tensor_minus_native_nll=nll_delta,max_parameter_delta_over_sigma=float(np.max(delta[free]/sigma[free])))),flush=True)
