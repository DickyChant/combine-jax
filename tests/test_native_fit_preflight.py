from pathlib import Path
import json
import subprocess
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fit_preflight import run_preflight
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.fit_policy import POLICY, mass_cli_arguments
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.model import evaluate_model_numpy
from combine_jax.tensor.parameter_info import model_parameter_info
from test_bin_statistics import template_path


@pytest.mark.parametrize('mass_option',[True,False])
def test_actual_cli_preflight_preserves_mass_and_effective_bounds(counting_path,tmp_path,mass_option):
    root=load_caching_nll()
    with open_workspace(counting_path) as workspace:
        if not workspace.var('MH'):
            getattr(workspace,'import')(root.RooRealVar('MH','MH',125.))
        workspace.var('MH').setVal(125.);workspace.var('MH').setConstant(True)
        workspace.var('lumi').setRange(-.04,.04)
        # HHVV has four nonconstant variables with equal bounds. The benchmark
        # explicitly freezes them; the effective fixed state must be identical.
        workspace.var('bkg').setAttribute('optimizeBounds',False)
        workspace.var('bkg').setRange(0.,0.)
        model=from_workspace(workspace,Path(__file__).parent/'cards/counting.txt',fit_policy=POLICY)
        workspace.writeToFile(str(tmp_path/'input.root'))
    state=model_parameter_info(model).metadata['fit_policy']
    command=[sys.executable,str(Path(sys.prefix)/'bin/combine'),'-M','MultiDimFit','-d',str(tmp_path/'input.root'),
        '--algo','none','--saveWorkspace','-n','.test','-D','data_obs','--freezeParameters','bkg']
    if mass_option:command+=mass_cli_arguments(state)
    report={'dataset':'data_obs'}
    if mass_option:
        run_preflight(command,tmp_path/'preflight',model,open_workspace,report,lambda:None)
        assert report['preflight']['status']=='passed'
        assert report['preflight']['initial_state']['fixed_real_values']['MH']==125.
    else:
        with pytest.raises(ValueError,match='fit state differs'):
            run_preflight(command,tmp_path/'preflight',model,open_workspace,report,lambda:None)
        assert report['preflight']['status']=='failed'


@pytest.mark.parametrize('kind',['counting','automcstats','named_counting'])
def test_native_benchmark_preflight_and_postcheck(counting_path,template_path,tmp_path,kind):
    root=load_caching_nll()
    path=template_path if kind=='automcstats' else counting_path
    card=Path(path).parent/'card.txt' if kind=='automcstats' else Path(__file__).parent/'cards/counting.txt'
    dataset='test_asimov' if kind=='named_counting' else 'data_obs'
    with open_workspace(path) as workspace:
        if kind=='named_counting':
            original=workspace.data('data_obs');alternate=original.emptyClone(dataset)
            for i in range(original.numEntries()):
                row=original.get(i);alternate.add(row,1.5*original.weight())
            assert not getattr(workspace,'import')(alternate)
            path=tmp_path/'named.root';workspace.writeToFile(str(path))
        model=from_workspace(workspace,card,data=dataset,fit_policy=POLICY)
        model.save(tmp_path/'model')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps({kind:dict(workspace=str(path),model=str(tmp_path/'model'),
        configuration={},dataset=dataset,hashes={})}))
    output=tmp_path/'fit.json'
    command=[sys.executable,'scripts/benchmark_combine_fit.py','--manifest',str(manifest),'--case',kind,
        '--directory',str(tmp_path/'fit'),'--output',str(output)]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1000:]+completed.stderr[-1000:]
    result=json.loads(output.read_text())
    assert result['status']=='measured_fit_pending_agreement'
    assert result['preflight']['status']=='passed' and result['fit']['edm_target_met']
    assert result['dataset']==result['preflight']['dataset']['name']==dataset
    assert result['preflight']['dataset']['identity']==model.metadata['provenance']['dataset']['identity']
    assert result['evaluation_counts']['actual_nll_evaluations']>0
    assert result['evaluation_counts']['numerical_gradient_evaluations']>0
    assert abs(evaluate_model_numpy(model,result['fit']['values'])-result['fit']['canonical_nll'])<1e-10
    assert result['fit']['roofit_evaluation_errors']==0
    print('NATIVE_POLICY_BENCHMARK_SMOKE',json.dumps(dict(case=kind,edm=result['fit']['edm'],nll=result['fit']['canonical_nll'],
        counts=result['evaluation_counts'])),flush=True)
