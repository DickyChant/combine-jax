"""Public-package-only import/fit/scan transport, dataset policy and exact curves."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from test_discrete_masks import multipdf_path


def cli(*args):
    return subprocess.run([sys.executable,'-m','combine_jax.cli',*map(str,args)],
        capture_output=True,text=True,env=dict(os.environ),timeout=180)


@pytest.fixture(scope='module')
def workspace_path(tmp_path_factory):
    from combine_jax.root_io import load_caching_nll
    r=load_caching_nll();path=tmp_path_factory.mktemp('public_workflow')/'input.root'
    w=r.RooWorkspace('w')
    for text in ('x[0.5,0,1]','mu[1,0.1,4]','Uniform::shape(x)',
                 'expr::rate("12*@0",mu)','SUM::channel(rate*shape)'):
        w.factory(text)
    w.var('mu').setError(.2)
    category=r.RooCategory('CMS_channel','');category.defineType('one')
    sim=r.RooSimultaneous('sim','',category);sim.addPdf(w.pdf('channel'),'one')
    pdf=r.RooSimultaneousOpt(sim);pdf.SetName('model_s')
    getattr(w,'import')(pdf,r.RooFit.RecycleConflictNodes())
    cols=r.RooArgSet(w.var('x'),w.cat('CMS_channel'))
    mc=r.RooStats.ModelConfig('ModelConfig',w);mc.SetPdf(w.pdf('model_s'));mc.SetObservables(cols)
    mc.SetParametersOfInterest(r.RooArgSet(w.var('mu')));mc.SetNuisanceParameters(r.RooArgSet())
    mc.SetGlobalObservables(r.RooArgSet());getattr(w,'import')(mc)
    w.saveSnapshot('start',w.allVars())
    data=r.RooDataSet('data_obs','',cols)
    for i in range(13):
        w.var('x').setVal((i+.5)/13);data.add(cols)
    getattr(w,'import')(data)
    weight=r.RooRealVar('weight','',1,0,100);acol=r.RooArgSet(cols);acol.add(weight)
    asimov=r.RooDataSet('toy_asimov','',acol,r.RooFit.WeightVar(weight));asimov.add(acol,12.)
    f=r.TFile.Open(str(path),'RECREATE');w.Write();f.mkdir('toys').cd();asimov.Write();f.Close()
    return path


@pytest.mark.parametrize('dataset,kind,total',[('data_obs','observed',13.),('toys/toy_asimov','asimov',12.)])
def test_public_import_scan_matches_poisson(workspace_path,tmp_path,dataset,kind,total):
    from combine_jax.tensor.bundle import TensorBundle,file_hash
    original=file_hash(workspace_path);model=tmp_path/'model'
    result=cli('import',workspace_path,'--data',dataset,'--analysis','synthetic',
        '--snapshot','start','--poi','mu','-o',model)
    assert result.returncode==0,result.stdout+result.stderr
    assert file_hash(workspace_path)==original
    m=TensorBundle.load(model)
    storage=m.metadata['workflow_import']['density_storage']
    assert storage['enabled'] and storage['model_array_bytes_after']==sum(a.nbytes for a in m.arrays.values())
    assert m.metadata['workflow_import']['dataset']==dict(name=dataset,kind=kind,analysis='synthetic',
        observed_data_accessed=kind=='observed',authorization='synthetic fixture')
    output=tmp_path/'scan.json'
    result=cli('scan',model,'--device','cpu','--set','mu=2','--poi','mu','--points','.7','1','1.4',
        '--threads','2','--host-memory-limit-gib','30','-o',output)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(output.read_text());assert report['all_fits_converged']
    assert report['cpu_threads']['requested']==2 and len(report['cpu_affinity'])==2
    assert report['cpu_threads']['environment']['OMP_NUM_THREADS']=='2'
    assert '--xla_cpu_multi_thread_eigen=true' in report['environment']['XLA_FLAGS']
    assert report['host_memory_limit_gib']==30.
    assert report['resources']['rss_limit_mib']==30*1024
    assert report['compiler_static_guard_bytes']==8*2**30
    fit=report['fits'][0];values=np.load(fit['capture'])['values']
    np.testing.assert_allclose(values,[total/12.],atol=.004,rtol=0)
    for row,value in zip(report['fits'][1:],[.7,1.,1.4],strict=True):
        # Compare to the saved approximate EDM minimum, not an exact-denominator assumption.
        expected=12*(value-values[0])-total*np.log(value/values[0])
        np.testing.assert_allclose(row['deltaNLL'],expected,atol=1e-10,rtol=0)
    assert len(report['compilations'])==2
    assert all(c['allocation_bytes']<8*2**30 for c in report['compilations'])
    assert report['dataset']['name']==dataset
    assert not any('/private_tools/' in str(v) for v in report.values())
    if kind=='asimov':
        fitted=tmp_path/'fixed.json'
        result=cli('fit',model,'--set','mu=1.2','--freeze','mu','-o',fitted)
        assert result.returncode==0,result.stdout+result.stderr
        fit_report=json.loads(fitted.read_text())
        assert fit_report['all_fits_converged'] and fit_report['fits'][0]['fixed']=={'mu':1.2}
        np.testing.assert_array_equal(np.load(fit_report['fits'][0]['capture'])['values'],[1.2])


def test_blind_import_refuses_observed_before_opening(tmp_path):
    result=cli('import',tmp_path/'nonexistent.root','--data','data_obs','--analysis','hhvv','-o',tmp_path/'model')
    assert result.returncode!=0 and 'Observed data are prohibited' in result.stderr


def test_public_scan_reports_named_iteration_failure(workspace_path,tmp_path):
    model=tmp_path/'model'
    result=cli('import',workspace_path,'--data','toys/toy_asimov','--analysis','synthetic',
        '--snapshot','start','--poi','mu','-o',model)
    assert result.returncode==0,result.stdout+result.stderr
    output=tmp_path/'failed.json'
    result=cli('scan',model,'--set','mu=2','--poi','mu','--points','1','--max-iterations','1','-o',output)
    assert result.returncode!=0
    assert 'refit: iteration cap reached (iterations 1/1)' in result.stderr
    assert 'corrected EDM' in result.stderr and '0.0001' in result.stderr
    report=json.loads(output.read_text())
    assert report['status']=='fit_failed' and report['failures'][0]['task']=='refit'
    assert report['failures'][0]['max_iterations']==1


def test_public_default_cap_and_invalid_denominator_details():
    import argparse
    from combine_jax.workflow_cli import add_commands
    from combine_jax.tensor.workflow import workflow_failures
    parser=argparse.ArgumentParser();add_commands(parser.add_subparsers(dest='command'))
    for command in ('fit','scan'):
        extra=['--poi','kl','--points','1'] if command=='scan' else []
        assert parser.parse_args([command,'model','-o','result.json',*extra]).max_iterations==10000
        assert parser.parse_args([command,'model','-o','result.json',*extra]).host_memory_limit_gib==11.
    report=dict(status='invalid_global_baseline',fits=[],global_baseline_check=dict(valid=False,
        lower_points=[dict(task='point_002',deltaNLL=-.003,roundoff_allowance=1e-6)]))
    failure=workflow_failures(report)[0]
    assert failure['task']=='point_002' and failure['kind']=='denominator'
    assert '0.003 below the unrestricted fit' in failure['message']


@pytest.mark.parametrize('limit',['0','-1','nan','inf','1e308'])
def test_public_host_limit_rejects_invalid_budget_before_loading(limit,tmp_path):
    from combine_jax.tensor.workflow import run_model
    result=cli('fit',tmp_path/'nonexistent','-o',tmp_path/'result.json',
        '--host-memory-limit-gib',limit)
    assert result.returncode!=0 and '--host-memory-limit-gib' in result.stderr
    with pytest.raises(ValueError,match='positive and finite'):
        run_model(tmp_path/'nonexistent',tmp_path/'result.json',host_memory_limit_gib=float(limit))
    assert not (tmp_path/'result.fits').exists()


@pytest.mark.parametrize('seed', [1.99,2.])
def test_fixed_scan_recovers_a_false_global_minimum_and_preserves_freezes(tmp_path,seed):
    """Two basins: the negative one meets EDM but is disproved by mu=+2.

    Gaussian data measure mu**2+eta=5.25; a weak Gaussian constraint prefers
    mu=+2. Freeze eta in the fit, not the import, so recovery must retain it.
    """
    from combine_jax.root_io import load_caching_nll
    r=load_caching_nll();path=tmp_path/'two_basins.root';w=r.RooWorkspace('w')
    for text in ('x[5.25,-20,20]','mu[-2,-3,3]','eta[1.25,-2,2]',
                 'expr::mean("@0*@0+@1",mu,eta)','Gaussian::shape(x,mean,width[1])',
                 'SUM::channel(rate[1]*shape)','CMS_channel[one=0]',
                 'Gaussian::constraint(mu,global_mu[2],constraint_width[10])',
                 'PROD::constrained_channel(channel,constraint)',
                 'SIMUL::model(CMS_channel,one=constrained_channel)'):
        assert w.factory(text)
    w.var('mu').setError(.2);w.var('eta').setError(.2)
    cols=r.RooArgSet(w.var('x'),w.cat('CMS_channel'))
    mc=r.RooStats.ModelConfig('ModelConfig',w);mc.SetPdf(w.pdf('model'));mc.SetObservables(cols)
    mc.SetParametersOfInterest(r.RooArgSet(w.var('mu')));mc.SetNuisanceParameters(r.RooArgSet(w.var('eta')))
    mc.SetGlobalObservables(r.RooArgSet(w.var('global_mu')));getattr(w,'import')(mc)
    data=r.RooDataSet('data_obs','',cols);data.add(cols);getattr(w,'import')(data)
    w.writeToFile(str(path));model=tmp_path/'model'
    result=cli('import',path,'--analysis','synthetic','--data','data_obs','--poi','mu','-o',model)
    assert result.returncode==0,result.stdout+result.stderr
    output=tmp_path/'scan.json'
    result=cli('scan',model,'--freeze','eta','--poi','mu','--points','-2',str(seed),'2.5',
        '--max-iterations','20','-o',output)
    report=json.loads(output.read_text())
    assert 'Global recovery from point_001;' in result.stdout
    if seed==2.:
        # Existing BFGS curvature-readiness guard cannot certify this numerically
        # stationary restart (nonzero 1e-48 derivative, no representable step).
        # Recovery must expose failure, never accept it or re-zero the grid.
        assert result.returncode!=0 and 'recovery_refit: iteration cap' in result.stderr
        assert report['global_recovery']['recovered_status']==1
        assert report['selected_global_baseline_task']=='refit'
        assert not report['all_fits_converged'] and not report['global_baseline_check']['valid']
        assert report['global_recovery']['nll_improvement']==0
        return
    assert result.returncode==0,result.stdout+result.stderr
    assert report['all_fits_converged']
    assert report['fits'][0]['mu']<0 and report['fits'][0]['status']==0
    assert not report['initial_global_baseline_check']['valid']
    assert report['selected_global_baseline_task']=='recovery_refit'
    recovery=report['global_recovery']
    assert recovery['seed_task']=='point_001' and recovery['nll_improvement']>.07
    assert recovery['fixed']=={'eta':1.25} and recovery['attempt_limit']==1
    assert report['global_baseline_check']['valid'] and not report['failures']
    assert len(report['fits'])==5 and len(report['compilations'])==2
    recovered=report['fits'][-1]
    assert recovered['task']=='recovery_refit' and abs(recovered['mu']-2)<.004
    capture=np.load(recovered['capture']);names=capture['parameter_names'].tolist()
    assert capture['values'][names.index('eta')]==1.25
    assert next(row for row in report['fits'] if row['task']=='point_001')['deltaNLL']<-.07
    for row in report['fits']:assert row['deltaNLL_baseline_task']=='refit'
    for row in report['scan_points']:
        assert row['baseline_task']=='recovery_refit'
        assert row['deltaNLL']==row['raw_nll']-recovered['value']


def test_published_preset_contains_no_paths():
    from combine_jax.workspace_import import hig_kl_options
    options=hig_kl_options()
    assert options['mass']==125.38 and options['ranges']=={'kl':[-15.,26.]}
    assert options['poi']=='kl' and 'kl' not in options['parameters']
    assert options['runtime_defines']['MINIMIZER_multiMin_maskChannels']==2
    assert all(x not in json.dumps(options) for x in ('/afs/','/eos/','/tmp/','private_tools'))


def test_public_density_compaction_is_default_and_byte_identical_at_26_points(multipdf_path,tmp_path):
    import jax
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import make_model_function
    models=[]
    for compact in (False,True):
        output=tmp_path/str(compact)
        result=cli('import',multipdf_path,'--analysis','synthetic','--data','data_obs',
            '--freeze','multi_slope,multi_c1,multi_c2','--discrete','profile','-o',output,
            *([] if compact else ['--no-compact-density']))
        assert result.returncode==0,result.stdout+result.stderr
        models.append(TensorBundle.load(output))
    before,after=models
    receipt=after.metadata['workflow_import']['density_storage']
    assert receipt['enabled'] and receipt['changed'] and receipt['removed_bytes']>0
    assert set(receipt['transformations'])=={'binned','discrete_channels'}
    assert receipt['removed_bytes']==sum(t['removed_float64_bytes'] for t in receipt['transformations'].values())
    assert not before.metadata['workflow_import']['density_storage']['enabled']
    assert receipt['model_array_bytes_before']==sum(a.nbytes for a in before.arrays.values())
    assert receipt['model_array_bytes_after']==sum(a.nbytes for a in after.arrays.values())
    theta=before.arrays['parameter_info__initial']
    points=np.vstack([theta,theta+np.random.default_rng(7319).uniform(-.01,.01,(25,len(theta)))])
    compiled=[jax.jit(jax.value_and_grad(make_model_function(m,binned_gradient='structural_sequential')))
              for m in models]
    for choice in (0,1):
        arrays=[dict(m.arrays,binned__discrete_indices=np.array([choice],dtype=np.int32)) for m in models]
        for point in points:
            results=[jax.device_get(f(point,a)) for f,a in zip(compiled,arrays)]
            for left,right in zip(*results):
                assert np.asarray(left).tobytes()==np.asarray(right).tobytes()


@pytest.fixture(scope='module')
def shared_workspace_path(tmp_path_factory):
    """Same Gaussian data/model as the logged native-v10 shared-shape fixture.

    Standalone construction: no private campaign helpers or full-model inputs.
    Native evidence: reports/hig23006_shared_discrete_native_equal_truth.json.
    """
    import math
    from combine_jax.root_io import load_caching_nll
    r=load_caching_nll();path=tmp_path_factory.mktemp('public_shared')/'input.root'
    w=r.RooWorkspace('w')
    for text in ('kl[1,-15,26]','x[0,-4,4]','CMS_channel[a=0,b=1,c=2]',
                 'mean_shared[0.3,-3,3]','mean_independent[0.1,-3,3]',
                 'unit_width[1]','expr::rate("40*exp(0.02*(@0-1))",kl)'):
        w.factory(text)
    for label in ('a','b','c'):
        w.factory(f'Gaussian::own_{label}(x,mean_{label}[0.2,-3,3],width_{label}[1,0.5,1.6])')
        mean='mean_shared' if label in ('a','b') else 'mean_independent'
        w.factory(f'Gaussian::alternative_{label}(x,{mean},unit_width)')
        index=r.RooCategory('pdfindex_'+label,'')
        pdf=r.RooMultiPdf('choice_'+label,'',index,r.RooArgList(w.pdf('own_'+label),w.pdf('alternative_'+label)))
        getattr(w,'import')(pdf,r.RooFit.RecycleConflictNodes())
        w.factory(f'SUM::channel_{label}(rate*choice_{label})')
    w.factory('SIMUL::model(CMS_channel,a=channel_a,b=channel_b,c=channel_c)')
    categories=r.RooArgSet('discreteParams')
    for label in ('b','a','c'):categories.add(w.cat('pdfindex_'+label))
    getattr(w,'import')(categories,'discreteParams')
    nuisances=r.RooArgSet()
    for name in ['mean_shared','mean_independent']+[n+'_'+c for c in ('a','b','c') for n in ('mean','width')]:
        v=w.var(name);v.setError(.1 if name.startswith('width') else .3);nuisances.add(v)
    w.var('kl').setError(.5)
    mc=r.RooStats.ModelConfig('ModelConfig',w);mc.SetPdf(w.pdf('model'))
    mc.SetObservables(r.RooArgSet(w.var('x'),w.cat('CMS_channel')))
    mc.SetParametersOfInterest(r.RooArgSet(w.var('kl')));mc.SetNuisanceParameters(nuisances)
    mc.SetGlobalObservables(r.RooArgSet());getattr(w,'import')(mc)
    snapshot=r.RooArgSet(w.allVars());snapshot.add(w.allCats());w.saveSnapshot('MultiDimFit',snapshot)
    weight=r.RooRealVar('weight','',0.,1e6);columns=r.RooArgSet(w.var('x'),w.cat('CMS_channel'),weight)
    data=r.RooDataSet('toy_asimov','',columns,r.RooFit.WeightVar(weight))
    centers=[-4.+(i+.5)*.1 for i in range(80)]
    for label,mean in zip(('a','b','c'),(-.8,-.8,.4),strict=True):
        w.cat('CMS_channel').setLabel(label);weights=[math.exp(-.5*(x-mean)**2) for x in centers]
        for x,value in zip(centers,weights,strict=True):
            w.var('x').setVal(x);data.add(columns,40.*value/sum(weights))
    w.loadSnapshot('MultiDimFit')
    f=r.TFile.Open(str(path),'RECREATE');w.Write();f.mkdir('toys').cd();data.Write();f.Close()
    return path


@pytest.mark.parametrize('candidate_batch_size',[1,4])
def test_public_shared_profile_scan_preserves_native_endpoint_and_category_restarts(shared_workspace_path,tmp_path,candidate_batch_size):
    model=tmp_path/'model'
    result=cli('import',shared_workspace_path,'--analysis','synthetic','--data','toys/toy_asimov',
        '--snapshot','MultiDimFit','--poi','kl','--discrete','profile',
        '--discrete-profile-mode','native-batched','-o',model)
    assert result.returncode==0,result.stdout+result.stderr
    imported=json.loads((model/'import.json').read_text())
    assert imported['discrete_profile_mode']=='native_batched'
    assert imported['runtime_defines']['MINIMIZER_multiMin_maskChannels']==2
    output=tmp_path/'scan.json'
    result=cli('scan',model,'--poi','kl','--points','0','1','2','--candidate-batch-size',candidate_batch_size,'-o',output)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(output.read_text());assert report['all_fits_converged']
    assert report['discrete_profile_mode']=='native_batched'
    assert report['global_baseline_check']['valid']
    first=report['fits'][0]
    np.testing.assert_array_equal(first['discrete_indices'],[1,1,1])
    assert first['discrete_iteration']==4
    # Pinned Combine v10.3.3 /ROOT 6.30.07 physical endpoint for this exact fixture.
    # Its intermediate category trajectory is deliberately not claimed identical.
    assert abs(first['value']-172.8597206031253)<1e-6
    assert report['fit_options']['binned_gradient']=='structural_sequential'
    assert report['fit_options']['edm_tolerance']==1e-4
    assert report['selected_global_baseline_task']=='profile_recovery_refit'
    assert report['profile_global_recovery']['seed_task']=='point_001'
    assert next(r for r in report['fits'] if r['task']=='point_001')['deltaNLL']<0
    baseline=next(r for r in report['fits'] if r['task']==report['selected_global_baseline_task'])
    for row in report['fits']:
        assert abs(row['native_rank_value']-row['raw_value']-row['masking_offset'])<1e-12
        if row['task'].startswith('point_'):
            np.testing.assert_array_equal(row['initial_discrete_indices'],first['discrete_indices'])
    for row in report['scan_points']:
        assert row['deltaNLL']==row['raw_nll']-baseline['value']
    assert len(report['scan_points'])==3
    assert all(c['allocation_bytes']<8*2**30 for c in report['compilations'])


def test_native_batched_import_requires_profile_selection(tmp_path):
    result=cli('import',tmp_path/'absent.root','--analysis','synthetic','--data','toys/toy_asimov',
        '--discrete-profile-mode','native-batched','-o',tmp_path/'model')
    assert result.returncode!=0 and 'requires discrete profiling' in result.stderr


def test_invalid_candidate_width_is_rejected_before_model_load(tmp_path):
    result=cli('fit',tmp_path/'absent','--candidate-batch-size','0','-o',tmp_path/'result.json')
    assert result.returncode!=0 and 'positive integer' in result.stderr


def test_native_batched_import_rejects_different_native_masking(tmp_path):
    result=cli('import',tmp_path/'absent.root','--analysis','synthetic','--data','toys/toy_asimov',
        '--discrete','profile','--discrete-profile-mode','native-batched',
        '--X-rtd','MINIMIZER_multiMin_maskChannels=1','-o',tmp_path/'model')
    assert result.returncode!=0 and 'requires MINIMIZER_multiMin_maskChannels=2' in result.stderr
