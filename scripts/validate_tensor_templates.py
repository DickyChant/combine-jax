"""Hash-bound HHVV template/profile gate plus a complete autoMCStats card gate."""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time
import traceback

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.constraints import make_constraint_function,evaluate_constraints
from combine_jax.tensor.counting import make_counting_function,evaluate_counting
from combine_jax.tensor.templates import pack_templates,make_template_function,evaluate_templates
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy
from combine_jax.root_io import open_workspace,load_caching_nll,REFERENCE_FLAGS


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual,reference,tolerance=1e-6):
    delta=np.abs(actual-reference)
    return dict(maximum_absolute_error=float(np.max(delta,initial=0.)),
        maximum_scaled_error=float(np.max(delta/np.maximum(1.,np.abs(reference)),initial=0.)),
        passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<tolerance)))


def hhvv(report,directory,save):
    previous_path=Path('reports/tensor_hhvv_likelihood_terms.json')
    previous=json.loads(previous_path.read_bytes())
    assert previous['status']=='constraint_and_counting_values_checked'
    for p,h in previous['bundles'].items():assert sha(p)==h
    assert sha(previous['validation_arrays'])==previous['validation_arrays_sha256']
    assert sha(previous['scalar_report'])==previous['scalar_report_sha256']
    assert sha(previous['saved_native_report'])==previous['saved_native_report_sha256']
    scalar_report=json.loads(Path(previous['scalar_report']).read_bytes())
    for p,h in scalar_report['tensor_sources'].items():assert sha(p)==h
    assert sha(scalar_report['export'])==scalar_report['export_sha256']
    assert sha(Path(scalar_report['bundle'])/'model.json')==scalar_report['bundle_metadata_sha256']
    assert sha(Path(scalar_report['bundle'])/'arrays.npz')==scalar_report['bundle_arrays_sha256']
    with Path(scalar_report['export']).open('rb') as stream:payload=pickle.load(stream)
    native=json.loads(Path(previous['saved_native_report']).read_bytes())
    assert native['source']==payload['source']==previous['source']
    references=np.load(previous['validation_arrays'],allow_pickle=False)
    scalar=TensorBundle.load(scalar_report['bundle'])
    constraints=TensorBundle.load(previous['constraint_bundle'])
    counting=TensorBundle.load(previous['channel_bundle'])
    nodes=payload['nodes'];rows=[]
    for name,pdf,axes,coordinates,weights,storage,masked in payload['channels']:
        if nodes[pdf][0]=='RooRealSumPdf':
            rows.append(dict(name=name,pdf=pdf,observables=axes,coordinates=coordinates,weights=weights,masked=masked))
    assert len(rows)==182
    before=time.perf_counter();templates=pack_templates(nodes,rows,scalar,profiles=payload['profiles'])
    report['hhvv']['template_import_seconds']=time.perf_counter()-before
    templates.save(directory/'hhvv_templates');templates=TensorBundle.load(directory/'hhvv_templates')
    profile_names=templates.metadata['profile_names']
    assert profile_names==references['profile_names'].tolist()==constraints.metadata['profile_names']
    assert counting.metadata['names']==references['channel_names'].tolist()
    assert constraints.metadata['names']==references['constraint_names'].tolist()
    sf=make_function(scalar);tf=make_template_function(templates)
    cf=make_constraint_function(constraints);chf=make_counting_function(counting)
    arrays=jax.device_put(dict(scalars=dict(scalar.arrays),templates=dict(templates.arrays),
        constraints=dict(constraints.arrays),counting=dict(counting.arrays)))
    def evaluate(theta,buffers):
        values=sf(theta,buffers['scalars'])
        channels,profiles,_=tf(values,buffers['templates'])
        return channels,profiles,cf(values,profiles,buffers['constraints']),chf(values,buffers['counting'])
    point=references['points'][0]
    before=time.perf_counter();lowered=jax.jit(evaluate).lower(point,arrays)
    report['hhvv']['lowering_seconds']=time.perf_counter()-before
    before=time.perf_counter();compiled=lowered.compile()
    report['hhvv']['compile_seconds']=time.perf_counter()-before
    report['hhvv'].update(template_channels=182,counting_channels=178,remaining_channels=99,profiles=len(profile_names),
        histograms=len(templates.metadata['histogram_names']),folded_histograms=len(templates.metadata['folded_histograms']),
        morph_batches=len(templates.metadata['schedule'])-1,buffer_bytes=sum(a.nbytes for a in templates.arrays.values()),
        closed_over_arrays=len(jax.make_jaxpr(evaluate)(point,arrays).consts),structure_key=templates.structure_key,
        prior_native_terms_report=str(previous_path),prior_native_terms_report_sha256=sha(previous_path),
        native_channel_report=previous['saved_native_report'],native_channel_report_sha256=previous['saved_native_report_sha256'],
        source=payload['source'])
    assert report['hhvv']['closed_over_arrays']==0
    matrices={n:[] for n in ('native_channels','jax_channels','numpy_channels','native_profiles','jax_profiles','numpy_profiles',
        'native_constraints','jax_constraints','numpy_constraints','native_counting','jax_counting','numpy_counting')}
    for i,case in enumerate(native['cases']):
        point=np.array(case['theta']);np.testing.assert_array_equal(point,references['points'][i])
        channels,profiles,penalties,counting_values=map(np.asarray,compiled(point,arrays))
        sv=evaluate_numpy(scalar,point);np_channels,np_profiles,_=evaluate_templates(templates,sv)
        np_penalties=evaluate_constraints(constraints,sv,np_profiles);np_counting=evaluate_counting(counting,sv)
        native_map={c['name']:c['combine'] for c in case['channels']}
        ref_channels=np.array([native_map[r['name']] for r in rows]);ref_profiles=references['profile_values'][i]
        ref_penalties=references['native_constraints'][i];ref_counting=references['native_channels'][i]
        checks=dict(channels=error(channels,ref_channels),profiles=error(profiles,ref_profiles,1e-10),
            constraints=error(penalties,ref_penalties),counting=error(counting_values,ref_counting),
            numpy_channels=error(np_channels,ref_channels),numpy_profiles=error(np_profiles,ref_profiles,1e-10),
            numpy_constraints=error(np_penalties,ref_penalties),numpy_counting=error(np_counting,ref_counting))
        report['hhvv']['cases'].append(dict(index=i,**checks));save()
        for kind,values in [('channels',(ref_channels,channels,np_channels)),('profiles',(ref_profiles,profiles,np_profiles)),
                            ('constraints',(ref_penalties,penalties,np_penalties)),('counting',(ref_counting,counting_values,np_counting))]:
            for backend,value in zip(('native','jax','numpy'),values,strict=True):matrices[backend+'_'+kind].append(value)
        print('HHVV',i,{k:v['maximum_absolute_error'] for k,v in checks.items()},flush=True)
    validation=directory/'hhvv_validation.npz'
    np.savez_compressed(validation,**{k:np.array(v) for k,v in matrices.items()},points=references['points'],
        channel_names=np.array([r['name'] for r in rows]),profile_names=np.array(profile_names),
        constraint_names=references['constraint_names'],counting_names=references['channel_names'])
    report['hhvv'].update(validation_arrays=str(validation),validation_arrays_sha256=sha(validation))
    assert len(report['hhvv']['cases'])==26 and all(v['passed'] for c in report['hhvv']['cases'] for k,v in c.items() if k!='index')


def automcstats(report,directory,save):
    manifest_path=Path('build/gpu_exports/manifest_random.json');manifest=json.loads(manifest_path.read_bytes())
    reference=manifest['models']['automcstats'];path=Path('build/gpu_exports/automcstats_source/workspace.root')
    assert sha(path)==reference['source']['sha256']
    out=report['automcstats'];root=load_caching_nll()
    assert str(root.gROOT.GetVersion())==manifest['root_version']==report['root_version']
    with open_workspace(path) as workspace:
        model=from_workspace(workspace,path.parent/'card.txt');model.save(directory/'automcstats')
        model=TensorBundle.load(directory/'automcstats');fn=make_model_function(model)
        arrays=jax.device_put(dict(model.arrays));point=model.arrays['scalars__parameter_initial']
        functions={}
        for name,function in [('value',fn),('gradient',jax.grad(fn)),('hessian',jax.hessian(fn))]:
            before=time.perf_counter();functions[name]=jax.jit(function).lower(point,arrays).compile()
            out[name+'_lower_compile_seconds']=time.perf_counter()-before
        out['closed_over_arrays']=len(jax.make_jaxpr(fn)(point,arrays).consts)
        names=model.metadata['parameter_names'];assert names==reference['reference']['parameters']
        native=root.cacheutils.CachingSimNLL(workspace.pdf('model_s'),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        native.setAnalyticBarlowBeeston(True);native.clearZeroPoint();native.clearConstantZeroPoint()
        root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
        for i,case in enumerate(reference['reference']['cases']):
            theta=np.array(case['theta'])
            for n,v in zip(names,theta,strict=True):workspace.var(n).setVal(float(v))
            root.RooAbsReal.clearEvalErrorLog();ref=float(native.getVal())
            nll=float(functions['value'](theta,arrays));g=np.array(functions['gradient'](theta,arrays));fd=np.array(case['combine_fd_gradient'])
            row=dict(index=i,nll=nll,native_nll=ref,nll_absolute_error=abs(nll-ref),
                numpy_absolute_error=abs(float(evaluate_model_numpy(model,theta))-ref),
                native_saved_absolute_error=abs(ref-case['combine_nll']),gradient=g.tolist(),native_fd_gradient=fd.tolist(),
                gradient_relative_error=(np.abs(g-fd)/np.maximum(1e-8,np.abs(fd))).tolist(),
                old_ad_gradient_max_error=float(np.max(np.abs(g-np.array(case['jax_gradient'])))),
                root_evaluation_errors=int(root.RooAbsReal.numEvalErrors()))
            if i<3:
                h=np.array(functions['hessian'](theta,arrays));previous=np.array(case['hessian']);fdh=np.array(case['fd_hessian'])
                row.update(hessian=h.tolist(),old_ad_hessian_max_error=float(np.max(np.abs(h-previous))),fd_hessian_max_error=float(np.max(np.abs(h-fdh))))
            out['cases'].append(row);save()
        native.setAnalyticBarlowBeeston(False);root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.PrintErrors)
    out.update(source=reference['source'],manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
        parameter_names=names,structure_key=model.structure_key,channels=model.metadata['channel_names'])
    assert len(out['cases'])==26 and out['closed_over_arrays']==0
    assert all(c['nll_absolute_error']<1e-6 and c['numpy_absolute_error']<1e-6 and c['native_saved_absolute_error']<1e-6
        and max(c['gradient_relative_error'])<1e-4 and c['old_ad_gradient_max_error']<1e-10
        and c.get('old_ad_hessian_max_error',0)<1e-10 and c.get('fd_hessian_max_error',0)<1e-6
        and c['root_evaluation_errors']==0 for c in out['cases'])


def main():
    path=Path('reports/tensor_templates.json');directory=Path('build/tensor_templates_validation')
    assert not path.exists() and not directory.exists(),'Preserve previous evidence'
    directory.mkdir();started=time.perf_counter()
    report=dict(status='running',scope='Complete autoMCStats card plus HHVV 182 template channels, 74 computed profiles, 178 counting channels and 2124 constraints. The other 99 HHVV channels remain pending; no full HHVV NLL/AD/fit or GPU performance claim.',
        hhvv=dict(cases=[]),automcstats=dict(cases=[]),script_sha256=sha(__file__),jax_version=jax.__version__,
        combine_version=version('higgs-combine-tool'),root_version=str(load_caching_nll().gROOT.GetVersion()),
        backend=jax.default_backend(),float64=bool(jax.config.x64_enabled),runtime_flags=REFERENCE_FLAGS,
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temp=path.with_suffix('.json.tmp');temp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temp.replace(path)
    save()
    try:
        hhvv(report,directory,save);automcstats(report,directory,save)
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report.update(status='passed',artifacts={str(p):sha(p) for p in sorted(directory.glob('*/*'))})
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['elapsed_seconds'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
