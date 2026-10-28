"""HHVV 50 additional channels, 410-channel subtotal and a complete binned card."""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import resource
import subprocess
import sys
import time
import traceback

import jax
import numpy as np

from combine_jax.root_io import open_workspace,load_caching_nll,REFERENCE_FLAGS
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.binned_channels import supports_binned,pack_binned_channels,make_binned_function,evaluate_binned_channels
from combine_jax.tensor.model import compose_binned,from_workspace,make_model_function,evaluate_model_numpy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(actual,reference):
    delta=np.abs(actual-reference)
    return dict(max_absolute_error=float(np.max(delta,initial=0.)),passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<1e-6)))


def hhvv(report,directory,save):
    out=report['hhvv'];scalar_report_path=Path('reports/tensor_hhvv_scalars.json')
    scalar_report=json.loads(scalar_report_path.read_bytes());assert scalar_report['status']=='scalar_values_checked'
    assert sha(scalar_report['export'])==scalar_report['export_sha256']
    assert sha(Path(scalar_report['bundle'])/'model.json')==scalar_report['bundle_metadata_sha256']
    assert sha(Path(scalar_report['bundle'])/'arrays.npz')==scalar_report['bundle_arrays_sha256']
    with Path(scalar_report['export']).open('rb') as stream:payload=pickle.load(stream)
    source_path=Path('reports/hhvv_random_values.json');source=json.loads(source_path.read_bytes())
    template_path=Path('reports/tensor_templates.json');template_report=json.loads(template_path.read_bytes())
    assert template_report['status']=='passed' and source['source']==payload['source']==template_report['hhvv']['source']
    for p,h in template_report['artifacts'].items():assert sha(p)==h
    term_path=Path(template_report['hhvv']['prior_native_terms_report'])
    assert sha(term_path)==template_report['hhvv']['prior_native_terms_report_sha256']
    term_report=json.loads(term_path.read_bytes())
    for p,h in term_report['bundles'].items():assert sha(p)==h
    assert sha(term_report['validation_arrays'])==term_report['validation_arrays_sha256']
    refs=np.load(term_report['validation_arrays'],allow_pickle=False)
    scalars=TensorBundle.load(scalar_report['bundle']);templates=TensorBundle.load('build/tensor_templates_validation/hhvv_templates')
    constraints=TensorBundle.load(term_report['constraint_bundle']);counting=TensorBundle.load(term_report['channel_bundle'])
    assert scalars.metadata['parameter_names']==source['parameter_names']
    nodes=payload['nodes'];rows=[]
    for name,pdf,axes,coords,weights,storage,masked in payload['channels']:
        if nodes[pdf][0]!='RooAddPdf' or all(nodes[n][0]=='RooUniform' for n in nodes[pdf][2]['pdfs']):continue
        if supports_binned(nodes,pdf):rows.append(dict(name=name,pdf=pdf,observables=axes,coordinates=coords,weights=weights,masked=masked))
    assert len(rows)==50
    before=time.perf_counter();binned=pack_binned_channels(nodes,rows,scalars)
    model=compose_binned(scalars,constraints,counting,binned,templates=templates,offset=payload['offset'],
        provenance=dict(source=payload['source'],scope='410/459 HHVV channels only; other 49 excluded explicitly',all_card_channels_imported=False))
    out['import_seconds']=time.perf_counter()-before
    model.save(directory/'hhvv_subset');model=TensorBundle.load(directory/'hhvv_subset')
    sf=make_function(scalars);bf=make_binned_function(binned);nll=make_model_function(model)
    arrays=jax.device_put(dict(model.arrays));sa=jax.device_put(dict(scalars.arrays));ba=jax.device_put(dict(binned.arrays))
    def evaluate(theta,arrays,sa,ba):return nll(theta,arrays),bf(sf(theta,sa),ba)
    point=np.array(source['initial']);before=time.perf_counter();lowered=jax.jit(evaluate).lower(point,arrays,sa,ba)
    out['lowering_seconds']=time.perf_counter()-before;before=time.perf_counter();compiled=lowered.compile();out['compile_seconds']=time.perf_counter()-before
    out.update(channels=[r['name'] for r in rows],subtotal_channels=model.metadata['channel_names'],remaining_channels=49,
        density_batches=len(binned.metadata['program']),nll_batches=len(binned.metadata['nll_program']),
        fast_batches=len(binned.metadata['histograms']['schedule']),folded_queries=len(binned.metadata['folded_queries']),
        binned_buffer_bytes=sum(a.nbytes for a in binned.arrays.values()),structure_key=model.structure_key,
        closed_over_arrays=len(jax.make_jaxpr(evaluate)(point,arrays,sa,ba).consts),source=payload['source'],
        native_report=str(source_path),native_report_sha256=sha(source_path),template_report=str(template_path),template_report_sha256=sha(template_path),
        scalar_report=str(scalar_report_path),scalar_report_sha256=sha(scalar_report_path),term_report=str(term_path),term_report_sha256=sha(term_path))
    assert len(out['subtotal_channels'])==410 and out['closed_over_arrays']==0
    tables={k:[] for k in ('native_channels','jax_channels','numpy_channels','native_subtotal','jax_subtotal','numpy_subtotal')}
    for i,case in enumerate(source['cases']):
        point=np.array(case['theta']);np.testing.assert_array_equal(point,refs['points'][i])
        subtotal,values=compiled(point,arrays,sa,ba);subtotal=float(subtotal);values=np.asarray(values)
        native_map={c['name']:c['combine'] for c in case['channels']}
        expected=np.array([native_map[r['name']] for r in rows])
        expected_subtotal=sum(native_map[n] for n in out['subtotal_channels'])+np.sum(refs['native_constraints'][i])+payload['offset']
        eager=evaluate_binned_channels(binned,evaluate_numpy(scalars,point));eager_subtotal=float(evaluate_model_numpy(model,point))
        row=dict(index=i,channels=errors(values,expected),numpy_channels=errors(eager,expected),
            subtotal=errors(np.array(subtotal),np.array(expected_subtotal)),numpy_subtotal=errors(np.array(eager_subtotal),np.array(expected_subtotal)))
        out['cases'].append(row);save()
        for k,v in [('native_channels',expected),('jax_channels',values),('numpy_channels',eager),
            ('native_subtotal',expected_subtotal),('jax_subtotal',subtotal),('numpy_subtotal',eager_subtotal)]:tables[k].append(v)
        print('HHVV',i,row,flush=True)
    validation=directory/'hhvv_validation.npz'
    np.savez_compressed(validation,**{k:np.array(v) for k,v in tables.items()},points=refs['points'],channel_names=np.array(out['channels']),subtotal_channels=np.array(out['subtotal_channels']))
    out.update(validation_arrays=str(validation),validation_arrays_sha256=sha(validation))
    assert len(out['cases'])==26 and all(c[k]['passed'] for c in out['cases'] for k in ('channels','numpy_channels','subtotal','numpy_subtotal'))


def complete_card(report,directory,save):
    out=report['complete_card'];path=directory/'fixture/workspace.root'
    reference_path=directory/'native_reference.json'
    # Creating and then reopening this fixture in the HHVV process crashed in
    # ROOT. A fresh native process passed the unchanged existing harness. Keep
    # construction, native reference measurement and tensor import isolated.
    for label,arguments in [('build_fixture',['scripts/build_binned_fixture.py',str(directory/'fixture')]),
        ('native_reference',['scripts/validate_binned_native_reference.py',str(path),str(reference_path)])]:
        log=directory/(label+'_process.log')
        with log.open('w') as stream:run=subprocess.run([sys.executable,*arguments],stdout=stream,stderr=subprocess.STDOUT)
        out[label+'_process']=dict(exit_code=run.returncode,log=str(log),log_sha256=sha(log));save();run.check_returncode()
    reference=json.loads(reference_path.read_bytes())
    assert reference['status']=='passed' and reference['workspace_sha256']==sha(path)
    assert reference['worker_sha256']==report['native_reference_worker_sha256']
    with open_workspace(path) as workspace:
        assert reference['max_nll_absolute_error']<1e-6 and reference['max_gradient_relative_error']<1e-4 and reference['max_hessian_absolute_error']<1e-6
        model=from_workspace(workspace,path.parent/'card.txt');model.save(directory/'complete_model');model=TensorBundle.load(directory/'complete_model')
        f=make_model_function(model);arrays=jax.device_put(dict(model.arrays));point=model.arrays['scalars__parameter_initial']
        functions={}
        for name,fn in [('value',f),('gradient',jax.grad(f)),('hessian',jax.hessian(f))]:
            before=time.perf_counter();functions[name]=jax.jit(fn).lower(point,arrays).compile();out[name+'_lower_compile_seconds']=time.perf_counter()-before
        out.update(parameter_names=model.metadata['parameter_names'],channels=model.metadata['channel_names'],
            structure_key=model.structure_key,closed_over_arrays=len(jax.make_jaxpr(f)(point,arrays).consts),
            native_reference=str(reference_path),native_reference_sha256=sha(reference_path),workspace=str(path),workspace_sha256=sha(path))
        assert out['parameter_names']==reference['parameters'] and out['closed_over_arrays']==0
        for i,case in enumerate(reference['cases']):
            point=np.array(case['theta']);value=float(functions['value'](point,arrays));gradient=np.array(functions['gradient'](point,arrays));fd=np.array(case['combine_fd_gradient'])
            row=dict(index=i,theta=point.tolist(),nll=value,native_nll=case['combine_nll'],nll_absolute_error=abs(value-case['combine_nll']),
                numpy_absolute_error=abs(float(evaluate_model_numpy(model,point))-case['combine_nll']),gradient=gradient.tolist(),
                gradient_relative_error=(np.abs(gradient-fd)/np.maximum(np.abs(fd),1e-8)).tolist(),old_ad_gradient_max_error=float(np.max(np.abs(gradient-np.array(case['jax_gradient'])))))
            if i<3:
                h=np.array(functions['hessian'](point,arrays));row.update(hessian=h.tolist(),
                    old_ad_hessian_max_error=float(np.max(np.abs(h-np.array(case['hessian'])))),fd_hessian_max_error=float(np.max(np.abs(h-np.array(case['fd_hessian'])))))
            out['cases'].append(row);save()
    assert len(out['cases'])==26 and all(c['nll_absolute_error']<1e-6 and c['numpy_absolute_error']<1e-6 and max(c['gradient_relative_error'])<1e-4
        and c['old_ad_gradient_max_error']<1e-10 and c.get('old_ad_hessian_max_error',0)<1e-10 and c.get('fd_hessian_max_error',0)<1e-6 for c in out['cases'])


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/tensor_binned_channels.json')
    parser.add_argument('--directory',default='build/tensor_binned_channels_validation_v2')
    args=parser.parse_args();path=Path(args.output);directory=Path(args.directory)
    assert not path.exists() and not directory.exists(),'Preserve prior evidence'
    directory.mkdir();started=time.perf_counter();root=load_caching_nll()
    report=dict(status='running',scope='Complete new binned card NLL/AD plus 50 HHVV binned channels and a 410-channel subtotal with all constraints and analytic profiles. The other 49 channels and full HHVV NLL/AD/fits remain pending.',
        hhvv=dict(cases=[]),complete_card=dict(cases=[]),script_sha256=sha(__file__),fixture_builder_sha256=sha('scripts/build_binned_fixture.py'),
        native_reference_worker_sha256=sha('scripts/validate_binned_native_reference.py'),
        jax_version=jax.__version__,combine_version=version('higgs-combine-tool'),root_version=str(root.gROOT.GetVersion()),
        float64=bool(jax.config.x64_enabled),backend=jax.default_backend(),runtime_flags=REFERENCE_FLAGS,
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    try:
        hhvv(report,directory,save);complete_card(report,directory,save)
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report.update(status='passed',artifacts={str(p):sha(p) for p in sorted(directory.glob('*/*')) if p.is_file()})
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['elapsed_seconds'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
