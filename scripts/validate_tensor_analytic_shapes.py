"""Native 26-point gates for a complete parametric card and 434 HHVV channels."""
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
from combine_jax.tensor.binned_channels import pack_binned_channels,make_binned_function,evaluate_binned_channels
from combine_jax.tensor.model import compose_binned,from_workspace,make_model_function,evaluate_model_numpy


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual,reference):
    delta=np.abs(actual-reference)
    return dict(max_absolute_error=float(np.max(delta,initial=0.)),passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<1e-6)))


def hhvv(report,directory,save):
    out=report['hhvv'];prior_path=Path('reports/tensor_binned_channels.json');prior=json.loads(prior_path.read_bytes())
    assert prior['status']=='passed';base=prior['hhvv']
    for field in ('native_report','scalar_report','term_report','template_report'):assert sha(base[field])==base[field+'_sha256']
    source=json.loads(Path(base['native_report']).read_bytes());sr=json.loads(Path(base['scalar_report']).read_bytes());terms=json.loads(Path(base['term_report']).read_bytes())
    assert sha(sr['export'])==sr['export_sha256'] and sha(terms['validation_arrays'])==terms['validation_arrays_sha256']
    with Path(sr['export']).open('rb') as stream:payload=pickle.load(stream)
    assert source['source']==payload['source']==base['source']
    refs=np.load(terms['validation_arrays'],allow_pickle=False);scalars=TensorBundle.load(sr['bundle'])
    templates=TensorBundle.load('build/tensor_templates_validation/hhvv_templates')
    constraints=TensorBundle.load(terms['constraint_bundle']);counting=TensorBundle.load(terms['channel_bundle'])
    input_artifacts={str(p):sha(p) for folder in (sr['bundle'],'build/tensor_templates_validation/hhvv_templates',terms['constraint_bundle'],terms['channel_bundle'])
        for p in (Path(folder)/'model.json',Path(folder)/'arrays.npz')}
    historical={**prior['artifacts'],**terms['bundles'],str(Path(sr['bundle'])/'model.json'):sr['bundle_metadata_sha256'],str(Path(sr['bundle'])/'arrays.npz'):sr['bundle_arrays_sha256']}
    tr=json.loads(Path(base['template_report']).read_bytes());historical.update(tr['artifacts'])
    assert all(historical[p]==digest for p,digest in input_artifacts.items())
    nodes=payload['nodes'];rows=[];old_names=set(base['channels'])
    for name,pdf,axes,coords,weights,storage,masked in payload['channels']:
        if name in old_names or name.startswith('WW_lvqq'):
            rows.append(dict(name=name,pdf=pdf,observables=axes,coordinates=coords,weights=weights,masked=masked))
    assert len(rows)==74 and sum(r['name'].startswith('WW_lvqq') for r in rows)==24
    before=time.perf_counter();binned=pack_binned_channels(nodes,rows,scalars)
    model=compose_binned(scalars,constraints,counting,binned,templates=templates,offset=payload['offset'],
        provenance=dict(source=payload['source'],scope='434/459 HHVV channels; remaining 25 excluded explicitly',all_card_channels_imported=False))
    out['import_seconds']=time.perf_counter()-before;model.save(directory/'hhvv_subset')
    sf=make_function(scalars);bf=make_binned_function(binned);nll=make_model_function(model)
    arrays=jax.device_put(dict(model.arrays));sa=jax.device_put(dict(scalars.arrays));ba=jax.device_put(dict(binned.arrays))
    def evaluate(theta,arrays,sa,ba):return nll(theta,arrays),bf(sf(theta,sa),ba)
    point=np.array(source['initial']);before=time.perf_counter();lowered=jax.jit(evaluate).lower(point,arrays,sa,ba)
    out['lowering_seconds']=time.perf_counter()-before;before=time.perf_counter();compiled=lowered.compile();out['compile_seconds']=time.perf_counter()-before
    out.update(channels=[r['name'] for r in rows],subtotal_channels=model.metadata['channel_names'],
        remaining_channels=sorted(set(source['cases'][0]['channels'][i]['name'] for i in range(459))-set(model.metadata['channel_names'])),
        density_batches=len(binned.metadata['program']),analytic_batches=sum(b['operation'].startswith('analytic_') for b in binned.metadata['program']),
        nll_batches=len(binned.metadata['nll_program']),fast_batches=len(binned.metadata['histograms']['schedule']),
        folded_queries=len(binned.metadata['folded_queries']),buffer_bytes=sum(a.nbytes for a in binned.arrays.values()),structure_key=model.structure_key,
        closed_over_arrays=len(jax.make_jaxpr(evaluate)(point,arrays,sa,ba).consts),source=payload['source'],
        prior_report=str(prior_path),prior_report_sha256=sha(prior_path),input_artifacts=input_artifacts,
        export=sr['export'],export_sha256=sr['export_sha256'])
    assert len(out['subtotal_channels'])==434 and len(out['remaining_channels'])==25 and out['closed_over_arrays']==0
    tables={k:[] for k in ('native_channels','jax_channels','numpy_channels','native_subtotal','jax_subtotal','numpy_subtotal')}
    for i,case in enumerate(source['cases']):
        point=np.array(case['theta']);np.testing.assert_array_equal(point,refs['points'][i])
        subtotal,values=compiled(point,arrays,sa,ba);subtotal=float(subtotal);values=np.asarray(values)
        native_map={c['name']:c['combine'] for c in case['channels']};expected=np.array([native_map[r['name']] for r in rows])
        expected_subtotal=sum(native_map[n] for n in out['subtotal_channels'])+np.sum(refs['native_constraints'][i])+payload['offset']
        eager=evaluate_binned_channels(binned,evaluate_numpy(scalars,point));eager_subtotal=float(evaluate_model_numpy(model,point))
        row=dict(index=i,channels=error(values,expected),numpy_channels=error(eager,expected),
            subtotal=error(np.array(subtotal),np.array(expected_subtotal)),numpy_subtotal=error(np.array(eager_subtotal),np.array(expected_subtotal)))
        out['cases'].append(row);save();print('HHVV',i,row,flush=True)
        for k,v in [('native_channels',expected),('jax_channels',values),('numpy_channels',eager),('native_subtotal',expected_subtotal),('jax_subtotal',subtotal),('numpy_subtotal',eager_subtotal)]:tables[k].append(v)
    path=directory/'hhvv_validation.npz'
    np.savez_compressed(path,**{k:np.array(v) for k,v in tables.items()},points=refs['points'],channel_names=np.array(out['channels']),subtotal_channels=np.array(out['subtotal_channels']))
    out.update(validation_arrays=str(path),validation_arrays_sha256=sha(path))
    assert len(out['cases'])==26 and all(c[k]['passed'] for c in out['cases'] for k in ('channels','numpy_channels','subtotal','numpy_subtotal'))


def parametric_card(report,directory,save):
    out=report['parametric_card'];path=Path('build/parametric_fixture/workspace.root');ref_path=directory/'native_reference.json';log=directory/'native_reference.log'
    with log.open('w') as stream:
        run=subprocess.run([sys.executable,'scripts/validate_analytic_native_reference.py',str(path),str(ref_path)],stdout=stream,stderr=subprocess.STDOUT)
    out['native_process']=dict(exit_code=run.returncode,log=str(log),log_sha256=sha(log));save();run.check_returncode()
    reference=json.loads(ref_path.read_bytes());assert reference['status']=='passed' and reference['workspace_sha256']==sha(path)
    assert reference['worker_sha256']==report['native_reference_worker_sha256']
    with open_workspace(path) as workspace:model=from_workspace(workspace,path.parent/'card.txt')
    model.save(directory/'parametric_model');model=TensorBundle.load(directory/'parametric_model')
    f=make_model_function(model);arrays=jax.device_put(dict(model.arrays));point=model.arrays['scalars__parameter_initial'];functions={}
    for name,fn in [('value',f),('gradient',jax.grad(f)),('hessian',jax.hessian(f))]:
        before=time.perf_counter();functions[name]=jax.jit(fn).lower(point,arrays).compile();out[name+'_lower_compile_seconds']=time.perf_counter()-before
    out.update(parameter_names=model.metadata['parameter_names'],channels=model.metadata['channel_names'],structure_key=model.structure_key,
        closed_over_arrays=len(jax.make_jaxpr(f)(point,arrays).consts),native_reference=str(ref_path),native_reference_sha256=sha(ref_path),
        workspace=str(path),workspace_sha256=sha(path),card=str(path.parent/'card.txt'),card_sha256=sha(path.parent/'card.txt'))
    assert out['parameter_names']==reference['parameters'] and out['closed_over_arrays']==0
    for i,case in enumerate(reference['cases']):
        point=np.array(case['theta']);value=float(functions['value'](point,arrays));gradient=np.array(functions['gradient'](point,arrays));fd=np.array(case['combine_fd_gradient'])
        row=dict(index=i,theta=point.tolist(),nll=value,native_nll=case['combine_nll'],nll_absolute_error=abs(value-case['combine_nll']),
            numpy_absolute_error=abs(float(evaluate_model_numpy(model,point))-case['combine_nll']),gradient=gradient.tolist(),
            gradient_relative_error=(np.abs(gradient-fd)/np.maximum(np.abs(fd),1e-8)).tolist(),old_ad_gradient_max_error=float(np.max(np.abs(gradient-np.array(case['jax_gradient'])))))
        if i<3:
            h=np.array(functions['hessian'](point,arrays));expected=np.array(case['fd_hessian']);delta=np.abs(h-expected)
            row.update(hessian=h.tolist(),old_ad_hessian_max_error=float(np.max(np.abs(h-np.array(case['hessian'])))),
                fd_hessian_max_error=float(np.max(delta)),fd_hessian_max_scaled_error=float(np.max(delta/np.maximum(np.abs(expected),1.))))
        out['cases'].append(row);save()
    assert len(out['cases'])==26 and all(c['nll_absolute_error']<1e-6 and c['numpy_absolute_error']<1e-6 and max(c['gradient_relative_error'])<1e-4
        and c['old_ad_gradient_max_error']<1e-8 and c.get('old_ad_hessian_max_error',0)<1e-8 and c.get('fd_hessian_max_scaled_error',0)<1e-6 for c in out['cases'])


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/tensor_analytic_shapes.json')
    parser.add_argument('--directory',default='build/tensor_analytic_validation_v2');args=parser.parse_args()
    path=Path(args.output);directory=Path(args.directory);assert not path.exists() and not directory.exists(),'Preserve prior evidence'
    directory.mkdir();started=time.perf_counter();root=load_caching_nll()
    report=dict(status='running',scope='24 additional WW lvqq channels, 434-channel HHVV subtotal with all constraints/profiles, and the existing complete two-channel parametric card. Full HHVV values/AD/fits and GPU performance remain pending.',
        hhvv=dict(cases=[]),parametric_card=dict(cases=[]),script_sha256=sha(__file__),native_reference_worker_sha256=sha('scripts/validate_analytic_native_reference.py'),
        jax_version=jax.__version__,combine_version=version('higgs-combine-tool'),root_version=str(root.gROOT.GetVersion()),
        float64=bool(jax.config.x64_enabled),backend=jax.default_backend(),runtime_flags=REFERENCE_FLAGS,
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    try:
        hhvv(report,directory,save);parametric_card(report,directory,save)
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report.update(status='passed',artifacts={str(p):sha(p) for p in sorted(directory.glob('*/*')) if p.is_file()})
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['elapsed_seconds'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
