"""Standalone Combine issue reproducer: a 12-bin 3D template, no analysis inputs.

In a cmsenv: python3 reproduce_variable_bin_volume.py --output-dir /tmp/repro
In the pip environment add --pip-stack. Creates a tiny card/workspace, prints
template, RooHistPdf, FastVerticalInterpHistPdf3D and background Asimov probabilities.
Only synthetic data are used. No combine-jax imports and no fits are required.
"""
import argparse
from array import array
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--pip-stack', action='store_true')
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    import ROOT as R
    if args.pip_stack:
        from cms_combine.cli import runtime_environment
        env = runtime_environment()
        for p in env['ROOT_INCLUDE_PATH'].split(':'): R.gInterpreter.AddIncludePath(p)
        library = next(Path(p)/'libHiggsAnalysisCombinedLimit.so' for p in env['LD_LIBRARY_PATH'].split(':')
                       if (Path(p)/'libHiggsAnalysisCombinedLimit.so').is_file())
    else:
        base = Path(os.environ['CMSSW_BASE'])
        R.gInterpreter.AddIncludePath(str(base/'src'))
        library = base/'lib'/os.environ['SCRAM_ARCH']/'libHiggsAnalysisCombinedLimit.so'
    assert R.gSystem.Load(str(library)) >= 0
    R.gROOT.SetBatch(True)
    R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    edges = [[0.,1.,2.], [0.,1.,2.], [0.,1.,3.,8.]]
    axes = [R.RooRealVar(n,n,.5,e[0],e[-1]) for n,e in zip(('x','y','z'),edges)]
    for x,e in zip(axes,edges): x.setBinning(R.RooBinning(len(e)-1,array('d',e)))
    obs = R.RooArgSet(*axes)
    cells = list(product(range(2),range(2),range(3)))
    weights = [float(i+1) for i in range(12)]
    volumes = [edges[2][iz+1]-edges[2][iz] for ix,iy,iz in cells]
    theta = R.RooRealVar('shape','shape',0.,-3.,3.)
    held = []; templates = R.RooArgList()
    for j in range(3):
        h = R.TH3D('hist'+str(j),'',*(arg for e in edges for arg in (len(e)-1,array('d',e))))
        h.SetDirectory(0)
        for i,cell in enumerate(cells):
            delta = .1*(cell[2]-1)
            value = weights[i]*(1 if j==0 else 1+(1 if j==1 else -1)*delta)
            h.SetBinContent(*(k+1 for k in cell),value)
        d = R.RooDataHist('template'+str(j),'',R.RooArgList(*axes),h)
        p = R.RooHistPdf('histpdf'+str(j),'',obs,d)
        held.extend((h,d,p)); templates.add(p)
    bkg = R.FastVerticalInterpHistPdf3D('background','',*axes,False,templates,R.RooArgList(theta),1.,-1)
    sig = R.RooHistPdf('signal','',obs,held[1])
    synthetic = R.RooDataHist('data_obs','synthetic template only',R.RooArgList(*axes),held[0])
    w = R.RooWorkspace('w')
    for p in (bkg,sig): getattr(w,'import')(p,R.RooFit.RecycleConflictNodes(),R.RooFit.Silence())
    getattr(w,'import')(synthetic,R.RooFit.Silence())
    w.writeToFile(str(directory/'shapes.root'))
    (directory/'card.txt').write_text('''imax 1
jmax 1
kmax *
shapes * ch shapes.root w:$PROCESS
bin ch
observation -1
bin ch ch
process signal background
process 0 1
rate 10 100
shape param 0 1
''')
    run = subprocess.run(['text2workspace.py','card.txt','-o','workspace.root'],cwd=directory,
                         capture_output=True,text=True)
    (directory/'text2workspace.log').write_text(run.stdout+run.stderr); run.check_returncode()
    f = R.TFile.Open(str(directory/'workspace.root'),'READ'); ws=f.Get('w')
    mc=ws.obj('ModelConfig'); ws.var('r').setVal(0.)
    data=R.asimovutils.asimovDatasetNominal(mc,0.,-1)
    getattr(ws,'import')(data,R.RooFit.Rename('asimov_bonly'))
    ws.var('r').setConstant(False); ws.var('r').setVal(0.)
    ws.writeToFile(str(directory/'asimov.root'))
    # Print direct independent template mass and the two native densities.
    values=[]
    for i,cell in enumerate(cells):
        for x,e,k in zip(axes,edges,cell): x.setVal((e[k]+e[k+1])/2)
        values.append(dict(cell=cell,volume=volumes[i],template_probability=weights[i]/sum(weights),
            histpdf_probability=float(templates[0].getVal(obs))*volumes[i],
            morpher_probability=float(bkg.getVal(obs))*volumes[i]))
    asimov=[]
    for i in range(data.numEntries()):
        row=data.get(i)
        asimov.append(dict(coordinates=[row.getRealValue(n) for n in ('x','y','z')],weight=float(data.weight())))
    flags=['ADDNLL_RECURSIVE','ADDNLL_GAUSSNLL','ADDNLL_HISTNLL','ADDNLL_CBNLL',
        'ADDNLL_ROOREALSUM_FACTOR','ADDNLL_ROOREALSUM_NONORM','ADDNLL_ROOREALSUM_BASICINT',
        'ADDNLL_ROOREALSUM_KEEPZEROS','ADDNLL_PRODNLL','ADDNLL_HFNLL','ADDNLL_HISTFUNCNLL','ADDNLL_ROOREALSUM_CHEAPPROD']
    for flag in flags: R.runtimedef.set(flag,1)
    nll=R.cacheutils.CachingSimNLL(mc.GetPdf(),data,mc.GetNuisanceParameters())
    nll.clearConstantZeroPoint(); nll.clearZeroPoint()
    cases=[]
    points=[(0.,0.)]+[(.2+i*.19,-1.8+i*.15) for i in range(25)]
    for r,t in points:
        ws.var('r').setVal(r); ws.var('shape').setVal(t)
        value=float(nll.getVal()); grad=[]
        for name in ('r','shape'):
            v=ws.var(name); saved=v.getVal(); step=1e-4
            if saved==0 and name=='r': step=1e-5  # Temporarily allow negative signal for a symmetric derivative.
            oldmin=v.getMin(); v.setMin(-10.)
            ys=[]
            for shift in (-2,-1,1,2): v.setVal(saved+shift*step); ys.append(float(nll.getVal()))
            v.setVal(saved); v.setMin(oldmin)
            grad.append((ys[0]-8*ys[1]+8*ys[2]-ys[3])/(12*step))
        cases.append(dict(point=[r,t],nll=value,gradient=grad))
    sha=lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    report=dict(root_version=R.gROOT.GetVersion(),cmssw=os.environ.get('CMSSW_VERSION'),
        library=str(library.resolve()),library_sha256=sha(library),script_sha256=sha(__file__),
        workspace_sha256=sha(directory/'asimov.root'),edges=edges,bins=values,asimov=asimov,cases=cases,
        parameter_names=['r','shape'],scope='Standalone synthetic 3D card, background Asimov, no fits or analysis data.')
    (directory/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'root':report['root_version'],'bins':values,'asimov':asimov},indent=2),flush=True)


if __name__=='__main__': main()
