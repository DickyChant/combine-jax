"""Independent stored-bin reconstruction of every VV4q one-dimensional morpher."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stack',choices=('v10','v11'),required=True)
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();assert not args.output.exists()
    if args.stack=='v11':
        from combine_jax.root_io import load_root
        root=load_root()
    else:
        import ROOT as root
        base=Path('/afs/cern.ch/work/s/sqian/crab_inference_dev/software/combine_hhvv_v10.3.3_el9_amd64_gcc12/CMSSW_14_1_0_pre4')
        root.gInterpreter.AddIncludePath(str(base/'src'))
        assert root.gSystem.Load(str(base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'))>=0
        assert root.gROOT.GetVersion().startswith('6.30')
    assert root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"')
    libraries=sorted(set(l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libHiggsAnalysisCombinedLimit.so' in l))
    assert len(libraries)==1
    if args.stack=='v10':assert Path(libraries[0]).resolve()==(base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so').resolve()
    f=root.TFile.Open(str(args.workspace),'READ');w=f.Get('w');w.var('MX').setVal(4000.)
    report=dict(status='running',stack=args.stack,root=root.gROOT.GetVersion(),script_sha256=sha(__file__),
        workspace_sha256=sha(args.workspace),library_sha256=sha(libraries[0]),pdfs={})
    for m in w.allPdfs():
        if m.ClassName()!='FastVerticalInterpHistPdf':continue
        assert m.smoothRegion()==1
        obs=m.getObservables(w.obj('ModelConfig').GetObservables());assert len(obs)==1
        x=next(iter(obs));templates=[];raw=[];coordinates=[];volumes=[];measured=[]
        for n,pdf in enumerate(m.funcList()):
            d=pdf.dataHist();values=[];coords=[];widths=[]
            for i in range(d.numEntries()):
                row=d.get(i);x.setVal(row.getRealValue(x.GetName()));values.append(d.weight());coords.append(x.getVal());widths.append(d.binVolume())
                if n==0:measured.append(m.getVal(obs))
            raw.append(values)
            if n==0:coordinates=coords;volumes=np.array(widths)
            else:np.testing.assert_array_equal(coordinates,coords);np.testing.assert_array_equal(volumes,widths)
        raw=np.array(raw);predictions={}
        for convention in ('density','mass'):
            ts=raw/raw.sum(axis=1)[:,None]
            if convention=='density':ts=ts/volumes*volumes.mean()
            ts=ts.astype(np.float32).astype(np.float64);ts/=np.sum(ts*volumes,axis=1)[:,None]
            transformed=np.log(ts) if m.smoothAlgo()<0 else ts
            nominal=transformed[0];value=nominal.copy()
            for i,c in enumerate(m.coefList()):
                t=float(c.getVal());step=np.sign(t) if abs(t)>=1 else .125*t*(t*t*(3*t*t-10)+15)
                hi=transformed[2*i+1]-nominal;lo=transformed[2*i+2]-nominal
                value+=.5*t*((hi-lo)+step*(hi+lo))
            value=np.exp(value) if m.smoothAlgo()<0 else np.maximum(value,1e-9)
            value/=np.sum(value*volumes)
            predictions[convention]=dict(max_abs=float(max(abs(value-measured))),
                max_relative=float(max(abs(value-measured)/np.maximum(abs(np.array(measured)),1e-300))))
        report['pdfs'][m.GetName()]=dict(class_name=m.ClassName(),observable=x.GetName(),bins=len(volumes),
            smooth_algo=int(m.smoothAlgo()),smooth_region=float(m.smoothRegion()),
            widths_min=float(volumes.min()),widths_max=float(volumes.max()),predictions=predictions,
            coefficients={c.GetName():float(c.getVal()) for c in m.coefList()},
            stored_values_sha256=hashlib.sha256(raw.astype('<f8').tobytes()).hexdigest())
    report['status']='completed';args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
