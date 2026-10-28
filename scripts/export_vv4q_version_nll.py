"""No-fit cross-version NLL probes, on identical v10 background Asimov rows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stack',choices=('v10','v11','v11_official'),required=True)
    parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--points',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gradient-index',type=int,help='Optional five-point finite-difference reference; no fit')
    args=parser.parse_args()
    assert not args.output.exists()
    if args.stack=='v11':
        from combine_jax.root_io import load_root
        root=load_root()
    else:
        import ROOT as root
        base=Path('/afs/cern.ch/work/s/sqian/crab_inference_dev') / (
            'software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4' if args.stack=='v10'
            else 'software_v11/combine_v11.0.0_el9_amd64_gcc13/CMSSW_16_0_0')
        arch='el9_amd64_gcc12' if args.stack=='v10' else 'el9_amd64_gcc13'
        root.gInterpreter.AddIncludePath(str(base/'src'))
        intended=base/'lib'/arch/'libHiggsAnalysisCombinedLimit.so'
        assert root.gSystem.Load(str(intended))>=0
        assert root.gROOT.GetVersion().startswith('6.30' if args.stack=='v10' else '6.36')
    root.gROOT.SetBatch(True)
    assert root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    plan=json.loads(args.points.read_text())
    for flag,value in plan['runtime_flags'].items():root.runtimedef.set(flag,value)
    loaded=sorted(set(line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libHiggsAnalysisCombinedLimit.so' in line))
    assert len(loaded)==1
    if args.stack!='v11':
        assert Path(loaded[0]).resolve()==intended.resolve()
        assert sha(loaded[0])==sha(intended)
    report=dict(status='running',stack=args.stack,root_version=root.gROOT.GetVersion(),
        script_sha256=sha(__file__),workspace_sha256=sha(args.workspace),data_sha256=sha(args.data),
        points_sha256=sha(args.points),library=loaded[0],library_sha256=sha(loaded[0]),
        runtime_flags=plan['runtime_flags'],cases=[],scope='Fixed Asimov points only. No fit and no observed data access.')
    report.update(cmssw_version=os.environ.get('CMSSW_VERSION'),scram_arch=os.environ.get('SCRAM_ARCH'))
    def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
    save()
    handle=root.TFile.Open(str(args.workspace),'READ');ws=handle.Get('w')
    dh=root.TFile.Open(str(args.data),'READ');data=dh.Get('toys/toy_asimov')
    assert data and ws
    ws.var('MX').setVal(4000.)
    assert ws.var('MX').isConstant()
    mc=ws.obj('ModelConfig')
    nll=root.cacheutils.CachingSimNLL(mc.GetPdf(),data,mc.GetNuisanceParameters())
    nll.clearConstantZeroPoint();nll.clearZeroPoint()
    variables=[ws.var(name) for name in plan['parameter_names']]
    for point in plan['cases']:
        for v,x in zip(variables,point['point']):v.setVal(float(x))
        start=time.perf_counter();value=float(nll.getVal())
        report['cases'].append(dict(index=point['index'],nll=value,evaluation_seconds=time.perf_counter()-start))
        save()
    if args.gradient_index is not None:
        point=next(p for p in plan['cases'] if p['index']==args.gradient_index)
        for v,x in zip(variables,point['point']):v.setVal(float(x))
        steps=[2e-4*max(float(v.getError()),1.) for v in variables]
        report['gradient']=dict(point_index=args.gradient_index,parameter_names=plan['parameter_names'],steps=steps,values=[])
        for v,x,h in zip(variables,point['point'],steps):
            assert v.getMin() < x-2*h and x+2*h < v.getMax()
            values=[]
            for shift in (-2,-1,1,2):
                v.setVal(float(x+shift*h));values.append(float(nll.getVal()))
            v.setVal(float(x))
            report['gradient']['values'].append((values[0]-8*values[1]+8*values[2]-values[3])/(12*h))
            save()
    report['status']='completed';save()
    print(args.stack,report['cases'][0]['nll'],len(report['cases']),flush=True)
    # Keep both read-only ROOT handles alive until process teardown.


if __name__=='__main__':main()
