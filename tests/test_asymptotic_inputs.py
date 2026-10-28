"""Blinding, mode consistency and real workspace/neutral CLI boundaries."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from combine_jax.methods.contracts import validate_expected_model
from combine_jax.methods.runner import run_expected_model


def model_record(mode='prefit'):
    return SimpleNamespace(metadata={
        'asymptotic_asimov': dict(poi='r', signal_strength=0., mode=mode,
                                 dataset='asimov', dataset_identity={'sha256':'test'}),
        'provenance': dict(dataset=dict(name='asimov', identity={'sha256':'test'}))})


@pytest.mark.parametrize('mode,run,no_fit,accepted', [
    ('prefit',None,False,True), ('conditional_background_fit',None,False,True),
    ('prefit','blind',False,True), ('prefit','expected',True,True),
    ('conditional_background_fit','expected',False,True),
    ('conditional_background_fit','blind',False,False),
    ('conditional_background_fit',None,True,False), ('prefit','expected',False,False)])
def test_prepared_mode_cannot_silently_change(mode, run, no_fit, accepted):
    model = model_record(mode)
    if accepted:
        assert validate_expected_model(model, 'r', run=run, no_fit_asimov=no_fit)['mode'] == mode
    else:
        with pytest.raises(ValueError, match='requested Asimov mode'):
            validate_expected_model(model, 'r', run=run, no_fit_asimov=no_fit)


@pytest.mark.parametrize('change', ['missing','signal','identity','name','mode','discrete'])
def test_runner_rejects_invalid_artifacts_before_fitting(change):
    model = model_record()
    record = model.metadata['asymptotic_asimov']
    if change == 'missing':
        del model.metadata['asymptotic_asimov']
    elif change == 'signal':
        record['signal_strength'] = 1.
    elif change == 'identity':
        record['dataset_identity'] = {'sha256':'different'}
    elif change == 'name':
        record['dataset'] = 'data_obs'
    elif change == 'mode':
        record['mode'] = 'observed'
    else:
        model.metadata['provenance']['discrete_choices'] = {'pdf_choice': {}}
    with pytest.raises((ValueError, NotImplementedError)):
        run_expected_model(model, 'r')


def test_limit_rejects_fitter_bound_removal_policy():
    model = model_record()
    model.metadata['parameter_info'] = {'fit_policy': {'nuisance_bounds_removed':['shape']}}
    with pytest.raises(ValueError,match='preserves workspace nuisance bounds'):
        validate_expected_model(model,'r')


@pytest.mark.parametrize('plateau', [0., .6])
def test_unresolved_diagnostic_does_not_replace_limit_or_baseline(monkeypatch, plateau):
    from combine_jax.tensor import limit_profile
    from combine_jax.methods.asymptotic import expected_limits
    from combine_jax.methods.root_output import audit_crossings
    class Profiler:
        def __init__(self, *args, **kwargs):
            self.r_max=10.;self.timings={};self.compiler_memory={};self.records=[]
        def compile(self): pass
        def global_minimum(self): return 0.
        def __call__(self,mu):
            self.records.append(dict(status=0,function_evaluations=1,gradient_evaluations=1,iteration=1))
            return plateau+.5*mu*mu
        def diagnose_global_baseline(self):
            return dict(used_for_limits=False,status='complete',nll=0.)
    monkeypatch.setattr(limit_profile,'LimitProfiler',Profiler)
    reference=expected_limits(Profiler(),0.,10.)
    result=run_expected_model(model_record(),'r')
    for name in ('limits','trace','minimum_nll','crossing_reasons'):
        assert result[name]==reference[name]
    if plateau:
        assert result['unresolved_baseline_diagnostic']['nll']==0.
        assert audit_crossings(result)[0]['limit'] is None
    else:
        assert 'unresolved_baseline_diagnostic' not in result


def test_fixed_discrete_requires_opt_in_and_matching_preparation():
    model = model_record()
    choices = {'pdf_choice': {'index': 2, 'selected': 'power', 'correction': 1.}}
    model.metadata['provenance']['discrete_choices'] = choices
    with pytest.raises(ValueError, match='matching preparation'):
        validate_expected_model(model, 'r', discrete_policy='fixed')
    record = model.metadata['asymptotic_asimov']
    record.update(discrete_policy='fixed', fixed_discrete_choices=choices.copy())
    with pytest.raises(NotImplementedError, match='not validated'):
        validate_expected_model(model, 'r')
    assert validate_expected_model(model, 'r', discrete_policy='fixed') is record
    record['fixed_discrete_choices'] = {'pdf_choice': {'index': 1}}
    with pytest.raises(ValueError, match='matching preparation'):
        validate_expected_model(model, 'r', discrete_policy='fixed')


@pytest.mark.parametrize('options', [
    ['--run','observed'], ['--run','both'], ['--mass','125'],
    ['--setParameters','MX=2000'], ['--data','different']])
def test_cli_rejects_invalid_mode_or_preparation_options(tmp_path, options):
    result = subprocess.run([sys.executable,'-m','combine_jax.cli','-M','AsymptoticLimits',
        '--model','does-not-exist','--output',str(tmp_path/'result.json'),*options],
        capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    assert 'No such file' not in result.stderr
    assert not (tmp_path/'result.json').exists()


def test_hhvv_conditional_rejects_unverified_data_before_any_fit(counting_path, tmp_path, monkeypatch):
    from combine_jax.root_io import open_workspace
    from combine_jax.methods.workspace import prepare_expected_workspace, digest
    from combine_jax.tensor import model as importer
    source = tmp_path/'tagged.root'
    # Synthetic fixture tagged with the HHVV POI name to exercise the guard.
    with open_workspace(counting_path) as workspace:
        workspace.factory('r_hhvv[1,0,10]')
        workspace.writeToFile(str(source))
    def forbidden(*args, **kwargs):
        pytest.fail('Unverified HHVV data reached the likelihood importer')
    monkeypatch.setattr(importer, 'from_workspace', forbidden)
    with pytest.raises(ValueError, match='verified Asimov input report'):
        prepare_expected_workspace(source,Path('tests/cards/counting.txt'),tmp_path/'missing',run='expected')
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps(dict(workspace_sha256=digest(source),
        generation_kind='combine_prefit_asimov', dataset=dict(name='data_obs',identity={'sha256':'wrong'}))))
    with pytest.raises(ValueError, match='identity mismatch'):
        prepare_expected_workspace(source,Path('tests/cards/counting.txt'),tmp_path/'tampered',
                                   run='expected',asimov_report=proof)


def test_workspace_cli_and_root_free_benchmark_match_native(counting_path,tmp_path,root):
    output, prepared = tmp_path/'limit.json', tmp_path/'prepared'
    command = [sys.executable,'-m','combine_jax.cli','limit','--workspace',str(counting_path),
        '--datacard','tests/cards/counting.txt','--poi','r','--workdir',str(prepared),
        '--output',str(output)]
    result = subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    measured = json.loads(output.read_text())
    assert measured['dataset']['mode'] == 'prefit'  # Workspace default is blind.
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.parameter_info import model_parameter_info
    info = model_parameter_info(TensorBundle.load(prepared/'model'))
    assert 'fit_policy' not in info.metadata
    assert np.isfinite(info.arrays['lower']).all() and np.isfinite(info.arrays['upper']).all()
    bench = tmp_path/'benchmark.json'
    result = subprocess.run([sys.executable,'scripts/benchmark_tensor.py','--operation','limit',
        '--backend','cpu','--model',str(prepared/'model'),'--poi','r','--output',str(bench)],
        capture_output=True,text=True,timeout=120)
    assert result.returncode == 0, result.stdout+result.stderr
    capture = json.loads(bench.read_text())
    assert capture['root_imported'] is False
    np.testing.assert_allclose(capture['limits'],measured['limits'],rtol=1e-12)
    native = subprocess.run(['combine','-M','AsymptoticLimits','-d',str(counting_path),
        '--run','blind','-m','125','-v','0'],cwd=tmp_path,capture_output=True,text=True,timeout=60)
    assert native.returncode == 0, native.stdout+native.stderr
    handle = root.TFile.Open(str(next(tmp_path.glob('higgsCombine*.root'))))
    expected = sorted((float(row.quantileExpected),float(row.limit)) for row in handle.Get('limit'))
    handle.Close()
    assert len(expected) == 5 and all(p > 0 for p, _ in expected)
    np.testing.assert_allclose(capture['limits'],[value for _,value in expected],rtol=.01,atol=0.)
    np.testing.assert_allclose(capture['quantiles'],[p for p,_ in expected],rtol=0.,atol=1e-7)
