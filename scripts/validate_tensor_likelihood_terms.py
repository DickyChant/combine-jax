"""Check every HHVV constraint and uniform channel at the saved 26 native points.

Native CachingSimNLL supplies profiled values. This validates the new consumers,
not a tensor implementation of the profile solver or the full HHVV likelihood.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import pickle
import resource
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from combine_jax.graph import walk
from combine_jax.root_io import open_workspace,load_caching_nll,REFERENCE_FLAGS
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.constraints import pack_constraints,make_constraint_function,evaluate_constraints
from combine_jax.tensor.counting import pack_counting_channels,make_counting_function,evaluate_counting
from combine_jax.tensor.jax_scalar import make_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual,reference):
    delta=np.abs(actual-reference)
    return dict(maximum_absolute_error=float(np.max(delta,initial=0.)),
        maximum_scaled_error=float(np.max(delta/np.maximum(1.,np.abs(reference)),initial=0.)),
        finite=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))),
        passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<1e-6)))


def main():
    path=Path('reports/tensor_hhvv_likelihood_terms.json')
    outdir=Path('build/tensor_hhvv_likelihood_terms')
    assert not path.exists() and not outdir.exists(),'Preserve prior evidence'
    outdir.mkdir()
    started=time.perf_counter()
    report=dict(status='running',scope='All 2124 constraints and 178 uniform channels only; native profile values are explicit runtime inputs. Other 281 channels and tensor profile solver remain pending.',
        cases=[],script_sha256=sha(__file__),jax_version=jax.__version__,float64=bool(jax.config.x64_enabled),runtime_flags=REFERENCE_FLAGS)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    save()
    try:
        scalar_report_path=Path('reports/tensor_hhvv_scalars.json')
        previous=json.loads(scalar_report_path.read_bytes())
        assert previous['status']=='scalar_values_checked'
        assert sha(previous['export'])==previous['export_sha256']
        for p,h in previous['tensor_sources'].items():assert sha(p)==h
        with Path(previous['export']).open('rb') as stream:payload=pickle.load(stream)
        scalar=TensorBundle.load(previous['bundle'])
        native_report_path=Path('reports/hhvv_random_values.json')
        native_saved=json.loads(native_report_path.read_bytes())
        assert native_saved['status']=='passed' and native_saved['source']==payload['source']
        assert native_saved['parameter_names']==scalar.metadata['parameter_names']
        workspace_path=Path('inputs/workspace_MX2000.root')
        assert sha(workspace_path)==payload['source']['sha256']
        report.update(scalar_report=str(scalar_report_path),scalar_report_sha256=sha(scalar_report_path),
            saved_native_report=str(native_report_path),saved_native_report_sha256=sha(native_report_path),
            source=payload['source'],tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
        nodes=payload['nodes'];profiles=sorted(n for p in payload['profiles'] for n in p.parameters)
        rows=[]
        for name,pdf,axes,coordinates,weights,storage,masked in payload['channels']:
            cls,_,attrs=nodes[pdf]
            if cls=='RooAddPdf' and all(nodes[n][0]=='RooUniform' for n in attrs['pdfs']):
                rows.append(dict(name=name,pdf=pdf,observables=axes,weights=weights,masked=masked))
        assert len(rows)==178 and len(payload['constraints'])==2124 and len(profiles)==74
        channels=pack_counting_channels(nodes,rows,scalar)
        root=load_caching_nll()
        report.update(root_version=str(root.gROOT.GetVersion()),combine_version=previous['combine_version'])
        assert report['root_version']==previous['root_version']
        native_constraints=[];jax_constraints=[];numpy_constraints=[]
        native_channels=[];jax_channels=[];numpy_channels=[];profile_values=[]
        with open_workspace(workspace_path) as workspace:
            for n,settings in payload['source']['configuration'].items():
                workspace.var(n).setVal(settings['value']);workspace.var(n).setConstant(settings['constant'])
            model=workspace.obj('ModelConfig');raw=walk(model.GetPdf())
            def integral(name,axes):
                return float(raw[name].createIntegral(root.RooArgSet(*(raw[n] for n in axes))).getVal())
            constraints=pack_constraints(nodes,payload['constraints'],payload['constraint_axes'],scalar,
                profile_names=profiles,fixed_normalization=integral)
            constraints.save(outdir/'constraints');channels.save(outdir/'channels')
            report.update(constraints=len(constraints.metadata['names']),channels=len(rows),profile_inputs=len(profiles),
                constraint_batches=len(constraints.metadata['schedule']),
                constraint_types=dict(Counter(nodes[n][0] for n in payload['constraints'])),
                folded_normalizations=len(constraints.metadata['folded_normalizations']),
                constraint_bundle=str(outdir/'constraints'),channel_bundle=str(outdir/'channels'),
                constraint_buffer_bytes=sum(a.nbytes for a in constraints.arrays.values()),
                channel_buffer_bytes=sum(a.nbytes for a in channels.arrays.values()))
            sf=make_function(scalar);cf=make_constraint_function(constraints);chf=make_counting_function(channels)
            arrays=jax.device_put(dict(scalars=dict(scalar.arrays),constraints=dict(constraints.arrays),channels=dict(channels.arrays)))
            def evaluate(theta,pvalues,buffers):
                values=sf(theta,buffers['scalars'])
                return cf(values,pvalues,buffers['constraints']),chf(values,buffers['channels'])
            point=np.array(native_saved['initial']);pvalues=np.array([workspace.var(n).getVal() for n in profiles])
            before=time.perf_counter();lowered=jax.jit(evaluate).lower(point,pvalues,arrays)
            report['lowering_seconds']=time.perf_counter()-before
            before=time.perf_counter();compiled=lowered.compile();report['compile_seconds']=time.perf_counter()-before
            report['closed_over_arrays']=len(jax.make_jaxpr(evaluate)(point,pvalues,arrays).consts)
            assert report['closed_over_arrays']==0
            reference=root.cacheutils.CachingSimNLL(model.GetPdf(),workspace.data('data_obs'),model.GetNuisanceParameters())
            reference.setAnalyticBarlowBeeston(True);reference.clearZeroPoint();reference.clearConstantZeroPoint()
            normsets={n:root.RooArgSet(*(raw[a] for a in payload['constraint_axes'].get(n,()))) for n in payload['constraints']}
            scalar_exec=jax.jit(sf)
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
            save()
            for index,case in enumerate(native_saved['cases']):
                point=np.array(case['theta'])
                for n,v in zip(scalar.metadata['parameter_names'],point,strict=True):workspace.var(n).setVal(float(v))
                root.RooAbsReal.clearEvalErrorLog()
                full_native=float(reference.getVal())
                pvalues=np.array([workspace.var(n).getVal() for n in profiles])
                expected=np.array([-float(raw[n].getLogValFast() if nodes[n][0].startswith('Simple') else raw[n].getLogVal(normsets[n])) for n in payload['constraints']])
                vals=compiled(point,pvalues,arrays)
                con=np.array(vals[0].block_until_ready());chn=np.array(vals[1].block_until_ready())
                saved={c['name']:c['combine'] for c in case['channels']}
                expected_channels=np.array([saved[c['name']] for c in rows])
                scalar_values=np.array(scalar_exec(point,arrays['scalars']))
                econ=evaluate_constraints(constraints,scalar_values,pvalues)
                echan=evaluate_counting(channels,scalar_values)
                row=dict(index=index,constraints=error(con,expected),channels=error(chn,expected_channels),
                    numpy_constraints=error(econ,expected),numpy_channels=error(echan,expected_channels),
                    native_full_nll=full_native,saved_native_full_absolute_error=abs(full_native-case['combine_nll']),
                    root_evaluation_errors=int(root.RooAbsReal.numEvalErrors()))
                report['cases'].append(row);save()
                native_constraints.append(expected);jax_constraints.append(con);numpy_constraints.append(econ)
                native_channels.append(expected_channels);jax_channels.append(chn);numpy_channels.append(echan);profile_values.append(pvalues)
                print('case',index,'constraint error',row['constraints']['maximum_absolute_error'],
                      'channel error',row['channels']['maximum_absolute_error'],'native full error',row['saved_native_full_absolute_error'],flush=True)
            reference.setAnalyticBarlowBeeston(False)
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.PrintErrors)
        destination=outdir/'validation.npz'
        np.savez_compressed(destination,constraint_names=np.array(payload['constraints']),channel_names=np.array([r['name'] for r in rows]),
            profile_names=np.array(profiles),profile_values=np.array(profile_values),points=np.array([c['theta'] for c in native_saved['cases']]),
            native_constraints=np.array(native_constraints),jax_constraints=np.array(jax_constraints),numpy_constraints=np.array(numpy_constraints),
            native_channels=np.array(native_channels),jax_channels=np.array(jax_channels),numpy_channels=np.array(numpy_channels))
        report.update(validation_arrays=str(destination),validation_arrays_sha256=sha(destination),
            bundles={str(p):sha(p) for p in sorted(outdir.glob('*/*'))})
        assert len(report['cases'])==26
        assert all(all(c[k]['passed'] for k in ('constraints','channels','numpy_constraints','numpy_channels'))
                   and c['root_evaluation_errors']==0 and c['saved_native_full_absolute_error']<1e-6 for c in report['cases'])
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report['status']='constraint_and_counting_values_checked'
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['elapsed_seconds'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
