"""Fixed Asimov channel/PDF probes in separately loaded Combine stacks; no fits."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stack', choices=('v10', 'v11'), required=True)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--points', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--backend', choices=('combine', 'legacy', 'cpu'), default='combine')
    p.add_argument('--plain-flags', action='store_true', help='Disable ADDNLL optimized dispatch in a fresh process')
    p.add_argument('--cached-pdf-values', action='store_true', help='Use native makeCachingPdf for per-process arrays')
    args = p.parse_args()
    assert not args.output.exists()
    if args.stack == 'v11':
        from combine_jax.root_io import load_root
        root = load_root()
    else:
        import ROOT as root
        base = Path('/afs/cern.ch/work/s/sqian/crab_inference_dev/software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4')
        root.gInterpreter.AddIncludePath(str(base/'src'))
        assert root.gSystem.Load(str(base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so')) >= 0
        assert root.gROOT.GetVersion().startswith('6.30')
    loaded = sorted(set(l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libHiggsAnalysisCombinedLimit.so' in l))
    assert len(loaded) == 1
    if args.stack == 'v10':
        assert Path(loaded[0]).resolve() == (base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so').resolve()
    root.gROOT.SetBatch(True)
    assert root.gInterpreter.Declare(r'''
    #include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"
    #include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"
    #include "RooDataHist.h"
    #include "RooHistPdf.h"
    namespace vv4q_probe {
    std::vector<double> values(RooAbsPdf &pdf, RooAbsData &data, RooArgSet &obs) {
      std::vector<double> result; result.reserve(data.numEntries());
      for (int i=0; i<data.numEntries(); ++i) {
        obs.assignValueOnly(*data.get(i)); result.push_back(pdf.getVal(&obs));
      }
      return result;
    }
    std::vector<double> columns(RooAbsData &data, std::string name) {
      std::vector<double> result; result.reserve(data.numEntries());
      for (int i=0; i<data.numEntries(); ++i) {
        const auto *row=data.get(i);
        result.push_back(name=="__weight__" ? data.weight() : row->getRealValue(name.c_str()));
      }
      return result;
    }
    }
    ''')
    plan = json.loads(args.points.read_text())
    flags = {k: 0 if args.plain_flags else v for k,v in plan['runtime_flags'].items()}
    for k,v in flags.items(): root.runtimedef.set(k,v)
    wh = root.TFile.Open(str(args.workspace), 'READ'); ws = wh.Get('w')
    dh = root.TFile.Open(str(args.data), 'READ'); data = dh.Get('toys/toy_asimov')
    assert ws and data
    ws.var('MX').setVal(4000.); assert ws.var('MX').isConstant()
    mc = ws.obj('ModelConfig'); pdf = mc.GetPdf(); category = pdf.indexCat()
    variables = [ws.var(n) for n in plan['parameter_names']]
    for v,x in zip(variables,plan['cases'][0]['point']): v.setVal(float(x))
    total = root.cacheutils.CachingSimNLL(pdf,data,mc.GetNuisanceParameters())
    total.clearConstantZeroPoint(); total.clearZeroPoint()
    report = dict(status='running',script_sha256=sha(__file__),stack=args.stack,
        root_version=root.gROOT.GetVersion(),library_sha256=sha(loaded[0]),
        workspace_sha256=sha(args.workspace),data_sha256=sha(args.data),points_sha256=sha(args.points),
        backend=args.backend,cached_pdf_values=args.cached_pdf_values,runtime_flags=flags,channels={},cases=[],
        scope='Identical fixed Asimov dataset; no fit. Per-process densities are diagnostic, not additive NLL terms.')
    arrays = {}; channels = {}; held=[]; caches={}
    def save(): args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    for item in category:
        name = str(item.first); channel = pdf.getPdf(name)
        assert channel.ClassName() == 'RooAddPdf'
        obs = channel.getObservables(mc.GetObservables())
        reduced = data.reduce(root.RooFit.Cut(f'{category.GetName()}=={category.GetName()}::{name}'),root.RooFit.SelectVars(obs))
        held.extend([obs,reduced])
        if args.backend == 'combine':
            nll = root.cacheutils.CachingAddNLL('probe_'+name,'',channel,reduced,True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
        else:
            nll = channel.createNLL(reduced,root.RooFit.Extended(True),root.RooFit.Offset(False),root.RooFit.EvalBackend(args.backend))
        channels[name]=(channel,obs,reduced,nll)
        components = list(channel.pdfList())
        # Also isolate each 3D morpher's stored RooHistPdf input.
        for q in list(components):
            if q.ClassName()=='FastVerticalInterpHistPdf3D':
                components.extend(s for s in q.servers() if s.ClassName()=='RooHistPdf')
        report['channels'][name]=dict(pdf=channel.GetName(),class_name=channel.ClassName(),
            rows=reduced.numEntries(),sum_weights=float(reduced.sumEntries()),
            observables=[v.GetName() for v in obs],
            pdfs=[dict(name=q.GetName(),class_name=q.ClassName()) for q in components])
        channels[name] += (components,)
        if args.cached_pdf_values:
            for q in components:
                cache=root.cacheutils.makeCachingPdf(q,obs)
                cache.setIncludeZeroWeights(True)
                caches[name,q.GetName()]=cache
        arrays[name+'__weights']=np.asarray(root.vv4q_probe.columns(reduced,'__weight__')).copy()
        for o in obs: arrays[name+'__coord__'+o.GetName()]=np.asarray(root.vv4q_probe.columns(reduced,o.GetName())).copy()
    for point in plan['cases']:
        if args.backend != 'combine' and point['index'] not in (0,1,23): continue
        for v,x in zip(variables,point['point']): v.setVal(float(x))
        row=dict(index=point['index'],total=float(total.getVal()),channels={})
        for name,(channel,obs,reduced,nll,components) in channels.items():
            entry=dict(nll=float(nll.getVal()),coefficients=[float(c.getVal()) for c in channel.coefList()])
            row['channels'][name]=entry
            if point['index'] in (0,1,23):
                for q in components:
                    key=f"point{point['index']}__{name}__{q.GetName()}"
                    values=(caches[name,q.GetName()].eval(reduced) if args.cached_pdf_values
                            else root.vv4q_probe.values(q,reduced,obs))
                    arrays[key]=np.asarray(values).copy()
                    assert np.isfinite(arrays[key]).all(),key
        if args.backend=='combine': row['constraints']=row['total']-sum(c['nll'] for c in row['channels'].values())
        report['cases'].append(row); save()
        print('case',point['index'],row['total'],flush=True)
    target=args.output.with_suffix('.npz'); np.savez_compressed(target,**arrays)
    report.update(status='completed',arrays=str(target),arrays_sha256=sha(target));save()


if __name__=='__main__': main()
