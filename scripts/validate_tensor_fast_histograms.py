"""Every HHVV fast-histogram bin and full integral at 26 native reference points."""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import resource
import time
import traceback

import jax
import numpy as np

from combine_jax.graph import walk
from combine_jax.root_io import open_workspace,load_root
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import evaluate_numpy
from combine_jax.tensor.fast_histograms import (FAST_CLASSES,pack_fast_histograms,pack_fast_queries,
    make_fast_function,make_query_function,evaluate_fast_histograms,evaluate_fast_queries)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def comparison(actual,reference,absolute=1e-12,relative=1e-10):
    delta=np.abs(actual-reference);tolerance=absolute+relative*np.abs(reference)
    return dict(max_absolute_error=float(np.max(delta,initial=0.)),
        max_relative_error=float(np.max(delta/np.maximum(np.abs(reference),1e-300),initial=0.)),
        max_tolerance_ratio=float(np.max(delta/tolerance,initial=0.)),
        passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<=tolerance)))


def main():
    path=Path('reports/tensor_fast_histograms.json');directory=Path('build/tensor_fast_histograms_validation')
    assert not path.exists() and not directory.exists(),'Preserve prior evidence'
    directory.mkdir();started=time.perf_counter()
    report=dict(status='running',scope='All 731 HHVV fast histogram PDFs, every stored bin and full default-range integral at 26 points. This operator gate does not add complete channel NLL/gradient/Hessian/fit or GPU benchmark claims.',
        cases=[],script_sha256=sha(__file__),combine_version=version('higgs-combine-tool'),jax_version=jax.__version__,
        float64=bool(jax.config.x64_enabled),backend=jax.default_backend(),tolerances=dict(absolute=1e-12,relative=1e-10),
        tensor_sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))})
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    save()
    try:
        spath=Path('reports/tensor_hhvv_scalars.json');previous=json.loads(spath.read_bytes())
        assert previous['status']=='scalar_values_checked'
        for p,h in previous['tensor_sources'].items():assert sha(p)==h
        assert sha(previous['export'])==previous['export_sha256']
        assert sha(Path(previous['bundle'])/'model.json')==previous['bundle_metadata_sha256']
        assert sha(Path(previous['bundle'])/'arrays.npz')==previous['bundle_arrays_sha256']
        with Path(previous['export']).open('rb') as stream:payload=pickle.load(stream)
        native_path=Path('reports/hhvv_random_values.json');native=json.loads(native_path.read_bytes())
        assert native['status']=='passed' and native['source']==payload['source']
        workspace_path=Path('inputs/workspace_MX2000.root');assert sha(workspace_path)==payload['source']['sha256']
        scalar=TensorBundle.load(previous['bundle']);assert scalar.metadata['parameter_names']==native['parameter_names']
        nodes=payload['nodes'];names=sorted(n for n,(cls,_,_) in nodes.items() if cls in FAST_CLASSES)
        before=time.perf_counter();histograms=pack_fast_histograms(nodes,names,scalar,provenance=dict(source=payload['source'],export_sha256=previous['export_sha256']))
        queries=pack_fast_queries(histograms,[dict(name=n,pdf=n,mode='integral') for n in names])
        report['import_seconds']=time.perf_counter()-before
        histograms.save(directory/'histograms');queries.save(directory/'queries')
        sf=make_function(scalar);hf=make_fast_function(histograms);qf=make_query_function(queries)
        arrays=jax.device_put(dict(s=dict(scalar.arrays),h=dict(histograms.arrays),q=dict(queries.arrays)))
        def evaluate(theta,arrays):
            bins=hf(sf(theta,arrays['s']),arrays['h'])
            return bins,qf(bins,arrays['q'])
        point=np.array(native['initial']);before=time.perf_counter();lowered=jax.jit(evaluate).lower(point,arrays)
        report['lowering_seconds']=time.perf_counter()-before
        before=time.perf_counter();compiled=lowered.compile();report['compile_seconds']=time.perf_counter()-before
        report.update(histograms=len(names),bins=len(histograms.arrays['initial_bins']),
            classes=dict(Counter(nodes[n][0] for n in names)),batches=len(histograms.metadata['schedule']),
            folded_histograms=len(histograms.metadata['folded_histograms']),
            buffer_bytes=sum(a.nbytes for a in histograms.arrays.values()),query_buffer_bytes=sum(a.nbytes for a in queries.arrays.values()),
            structure_key=histograms.structure_key,closed_over_arrays=len(jax.make_jaxpr(evaluate)(point,arrays).consts),
            source=payload['source'],scalar_report=str(spath),scalar_report_sha256=sha(spath),native_points_report=str(native_path),native_points_report_sha256=sha(native_path))
        assert report['histograms']==731 and report['closed_over_arrays']==0
        root=load_root();report['root_version']=str(root.gROOT.GetVersion())
        assert report['root_version']==previous['root_version'] and report['combine_version']==previous['combine_version']
        assert root.gInterpreter.Declare(r'''
          #include "RooAbsPdf.h"
          #include "RooRealVar.h"
          #include "RooArgSet.h"
          namespace combine_jax_tensor_fast_check {
          std::vector<double> sample(std::vector<RooAbsPdf*> const& pdfs,
              std::vector<std::vector<RooRealVar*>> const& variables,
              std::vector<std::vector<std::vector<double>>> const& centers) {
            std::vector<double> result;
            for (size_t p=0; p<pdfs.size(); ++p) {
              RooArgSet norm; size_t size=1;
              for (size_t d=0; d<variables[p].size(); ++d) { norm.add(*variables[p][d]); size*=centers[p][d].size(); }
              for (size_t i=0; i<size; ++i) {
                size_t cell=i;
                for (int d=variables[p].size()-1; d>=0; --d) {
                  variables[p][d]->setVal(centers[p][d][cell%centers[p][d].size()]);
                  cell/=centers[p][d].size();
                }
                result.push_back(pdfs[p]->getVal(&norm));
              }
            }
            return result;
          }
          }
        ''')
        tables={k:[] for k in ('native_bins','jax_bins','numpy_bins','native_integrals','jax_integrals','numpy_integrals')}
        geometry=histograms.metadata['histograms'];volumes=[];centers=[]
        for h in geometry:
            edges=[histograms.arrays[k] for k in h['edges']];bounds=histograms.arrays[h['bounds']]
            centers.append([((e[:-1]+e[1:])/2).tolist() for e in edges])
            lengths=[np.maximum(0.,np.minimum(e[1:],hi)-np.maximum(e[:-1],lo)) for e,(lo,hi) in zip(edges,bounds,strict=True)]
            volumes.append(np.prod(np.array(np.meshgrid(*lengths,indexing='ij')),axis=0).reshape(-1))
        with open_workspace(workspace_path) as workspace:
            for n,settings in payload['source']['configuration'].items():
                workspace.var(n).setVal(settings['value']);workspace.var(n).setConstant(settings['constant'])
            raw=walk(workspace.obj('ModelConfig').GetPdf())
            pdfs=[raw[n] for n in names];variables=[[workspace.var(n) for n in h['axes']] for h in geometry]
            # Sampling stored tails can lie beyond a current default range; ROOT
            # accepts in-range HHVV grid centers. Reject instead of changing ranges.
            for vs,cs in zip(variables,centers,strict=True):
                for v,c in zip(vs,cs,strict=True):assert min(c)>=v.getMin() and max(c)<=v.getMax(),v.GetName()
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.CountErrors)
            for i,case in enumerate(native['cases']):
                point=np.array(case['theta'])
                for n,v in zip(scalar.metadata['parameter_names'],point,strict=True):workspace.var(n).setVal(float(v))
                root.RooAbsReal.clearEvalErrorLog()
                ref=np.asarray(root.combine_jax_tensor_fast_check.sample(pdfs,variables,centers)).copy()
                ref_integrals=np.array([np.sum(ref[h['offset']:h['offset']+h['size']]*v) for h,v in zip(geometry,volumes,strict=True)])
                bins,integrals=map(np.asarray,compiled(point,arrays))
                eager=evaluate_fast_histograms(histograms,evaluate_numpy(scalar,point));eager_integrals=evaluate_fast_queries(queries,eager)
                worst=int(np.argmax(np.abs(bins-ref)));owner=next(h for h in geometry if h['offset']<=worst<h['offset']+h['size'])
                row=dict(index=i,bins=comparison(bins,ref),numpy_bins=comparison(eager,ref),
                    integrals=comparison(integrals,ref_integrals),numpy_integrals=comparison(eager_integrals,ref_integrals),
                    worst_absolute_bin=dict(name=owner['name'],index=worst-owner['offset'],actual=float(bins[worst]),native=float(ref[worst])),
                    root_evaluation_errors=int(root.RooAbsReal.numEvalErrors()))
                report['cases'].append(row);save()
                for name,values in [('native_bins',ref),('jax_bins',bins),('numpy_bins',eager),('native_integrals',ref_integrals),('jax_integrals',integrals),('numpy_integrals',eager_integrals)]:tables[name].append(values)
                print('case',i,'bin error',row['bins'],'integral error',row['integrals'],flush=True)
            root.RooAbsReal.setEvalErrorLoggingMode(root.RooAbsReal.PrintErrors)
        numeric=directory/'validation.npz'
        np.savez_compressed(numeric,**{k:np.array(v) for k,v in tables.items()},names=np.array(names),
            offsets=np.array([h['offset'] for h in geometry]),sizes=np.array([h['size'] for h in geometry]),points=np.array([c['theta'] for c in native['cases']]))
        report.update(validation_arrays=str(numeric),validation_arrays_sha256=sha(numeric),artifacts={str(p):sha(p) for p in sorted(directory.glob('*/*'))})
        assert len(report['cases'])==26 and all(c['root_evaluation_errors']==0 and all(c[k]['passed'] for k in ('bins','numpy_bins','integrals','numpy_integrals')) for c in report['cases'])
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report['status']='passed'
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['elapsed_seconds'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
