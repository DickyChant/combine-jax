"""Export stored histogram weights and scalar densities, without a likelihood or fit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


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
    libraries=sorted(set(l.split()[-1] for l in Path('/proc/self/maps').read_text().splitlines() if 'libHiggsAnalysisCombinedLimit.so' in l))
    assert len(libraries)==1
    if args.stack=='v10':assert Path(libraries[0]).resolve()==(base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so').resolve()
    assert root.gInterpreter.Declare(r'''
    #include "RooHistPdf.h"
    #include "RooDataHist.h"
    #include "TH3.h"
    #include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
    namespace vv4q_hist {
    std::vector<double> stored(RooHistPdf &pdf, RooArgSet &obs, RooAbsPdf &morph) {
      auto &data=pdf.dataHist(); std::vector<double> out;
      for(int i=0;i<data.numEntries();++i){
        const auto *row=data.get(i); obs.assignValueOnly(*row);
        out.push_back(data.weight()); out.push_back(data.binVolume());
        for(auto *v:obs) out.push_back(row->getRealValue(v->GetName()));
        out.push_back(pdf.getVal(&obs)); out.push_back(morph.getVal(&obs));
      }return out;
    }
    std::vector<double> histogram(TH3 &h) {
      std::vector<double> out;
      for(int x=1;x<=h.GetNbinsX();++x)for(int y=1;y<=h.GetNbinsY();++y)for(int z=1;z<=h.GetNbinsZ();++z){
        out.push_back(h.GetXaxis()->GetBinCenter(x));out.push_back(h.GetYaxis()->GetBinCenter(y));out.push_back(h.GetZaxis()->GetBinCenter(z));
        out.push_back(h.GetXaxis()->GetBinWidth(x)*h.GetYaxis()->GetBinWidth(y)*h.GetZaxis()->GetBinWidth(z));
        out.push_back(h.GetBinContent(x,y,z));
      }return out;
    }
    }
    ''')
    f=root.TFile.Open(str(args.workspace),'READ');w=f.Get('w');w.var('MX').setVal(4000.)
    mc=w.obj('ModelConfig');report=dict(status='running',stack=args.stack,root=root.gROOT.GetVersion(),
        script_sha256=sha(__file__),workspace_sha256=sha(args.workspace),library_sha256=sha(libraries[0]),channels={})
    arrays={}; held=[]
    for channel in ('ch10','ch4','ch5','ch9'):
        morph=w.pdf('shapeBkg_nonRes_'+channel)
        pdf=next(s for s in morph.servers() if s.ClassName()=='RooHistPdf')
        obs=morph.getObservables(mc.GetObservables())
        data=pdf.dataHist()
        arr=np.asarray(root.vv4q_hist.stored(pdf,obs,morph)).copy().reshape(-1,7)
        arrays[channel+'__stored']=arr
        h=pdf.createHistogram('probe_'+channel,morph.x(),root.RooFit.YVar(morph.y()),root.RooFit.ZVar(morph.z()))
        h.SetDirectory(0); held.append(h)
        arrays[channel+'__histogram']=np.asarray(root.vv4q_hist.histogram(h)).copy().reshape(-1,5)
        # Plain NumPy density directly from stored RooDataHist bin weights and volumes.
        density=arr[:,0]/arr[:,0].sum()/arr[:,1]
        report['channels'][channel]=dict(pdf=pdf.GetName(),morph=morph.GetName(),
            observables=[v.GetName() for v in obs],stored_columns=['weight','volume']+[v.GetName() for v in obs]+['hist_pdf','morph_pdf'],
            histogram_columns=[morph.x().GetName(),morph.y().GetName(),morph.z().GetName(),'volume','content'],
            interpolation_order=int(pdf.getInterpolationOrder()),conditional=bool(morph.conditional()),
            smooth_algo=int(morph.smoothAlgo()),smooth_region=float(morph.smoothRegion()),
            raw_weight_sum=float(arr[:,0].sum()),rows=len(arr),
            hist_pdf_vs_raw_max_abs=float(np.max(abs(arr[:,-2]-density))),
            morph_vs_raw_max_abs=float(np.max(abs(arr[:,-1]-density))),
            morph_vs_raw_l1_probability=float(np.sum(abs(arr[:,-1]-density)*arr[:,1])))
        coefficients=[s for s in morph.servers() if s.ClassName()=='RooRealVar' and s.GetName() not in [v.GetName() for v in obs]]
        report['channels'][channel]['coefficients']={s.GetName():float(s.getVal()) for s in coefficients}
        report['channels'][channel]['templates']=[]
        for i,template in enumerate(s for s in morph.servers() if s.ClassName()=='RooHistPdf'):
            report['channels'][channel]['templates'].append(template.GetName())
            arrays[f'{channel}__template{i}']=np.asarray(root.vv4q_hist.stored(template,obs,morph)).copy().reshape(-1,7)
        # At this workspace nominal point only TurnOn=+1 is nonzero, so the
        # interpolation equals that Up template (after native normalization/floor).
        nonzero=[(i,float(s.getVal())) for i,s in enumerate(coefficients) if s.getVal()!=0]
        assert len(nonzero)==1 and nonzero[0][1]==1., nonzero
        idx=2*nonzero[0][0]+1
        target=arrays[f'{channel}__template{idx}']
        np.testing.assert_array_equal(target[:,2:5],arr[:,2:5])
        weights=target[:,0];volumes=target[:,1]
        candidates={
            'stored_density':weights/volumes,
            'stored_bin_mass_as_density':weights,
        }
        checks={}
        for label,raw in candidates.items():
            pred=raw/np.sum(raw*volumes)
            if morph.smoothAlgo()>=0:pred=np.maximum(pred,1e-9)
            pred/=np.sum(pred*volumes)
            checks[label]=dict(max_abs=float(np.max(abs(pred-arr[:,-1]))),
                l1_probability=float(np.sum(abs(pred-arr[:,-1])*volumes)))
        report['channels'][channel]['numpy_turnon_up_checks']=checks
        print(channel,report['channels'][channel],flush=True)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    np.savez_compressed(args.output.with_suffix('.npz'),**arrays)
    report.update(status='completed',arrays_sha256=sha(args.output.with_suffix('.npz')))
    args.output.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
