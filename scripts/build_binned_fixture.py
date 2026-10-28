"""A v11 card combining a 2D morpher and vertically interpolated floating bins."""
from array import array
from pathlib import Path
import subprocess

from combine_jax.root_io import load_root


def build_binned_fixture(directory):
    root=load_root();directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    xedges=[0.,.5,1.5,3.];yedges=[-1.,0.,2.]
    x=root.RooRealVar('bx','',.25,0.,3.);y=root.RooRealVar('by','',.5,-1.,2.)
    x.setBinning(root.RooBinning(3,array('d',xedges)));y.setBinning(root.RooBinning(2,array('d',yedges)))
    morph=root.RooRealVar('morph','',0.,-3.,3.)
    def histogram(name,values):
        hist=root.TH2D(name,'',3,array('d',xedges),2,array('d',yedges));hist.SetDirectory(0)
        for i,value in enumerate(values):hist.SetBinContent(i%3+1,i//3+1,value)
        return hist
    nominal=[2.,5.,3.,7.,4.,6.];histograms=[histogram('signal_nominal',nominal),
        histogram('signal_up',[v*f for v,f in zip(nominal,[1.1,.85,1.2,.9,1.1,1.05],strict=True)]),
        histogram('signal_down',[v*f for v,f in zip(nominal,[.9,1.15,.8,1.1,.9,.95],strict=True)])]
    templates=root.TList()
    for h in histograms:templates.Add(h)
    signal=root.FastVerticalInterpHistPdf2D2('signal','',x,y,False,templates,root.RooArgList(morph),1.,1)
    means=[9.,7.,8.,6.,5.,8.]
    coefficients=[root.RooRealVar(f'bin{i}','',value,.1,20.) for i,value in enumerate(means)]
    central=root.RooParametricHist2D('background_central','',x,y,root.RooArgList(*coefficients),histograms[0])
    held=[];functions=root.RooArgList(central)
    for label,scale in [('up',1.1),('down',.9)]:
        scaled=[]
        for i,c in enumerate(coefficients):
            f=root.RooFormulaVar(f'background_{label}_{i}',f'{scale+.01*i}*@0',root.RooArgList(c));held.append(f);scaled.append(f)
        pdf=root.RooParametricHist2D(f'background_{label}','',x,y,root.RooArgList(*scaled),histograms[0]);held.append(pdf);functions.add(pdf)
    background=root.VerticalInterpPdf('background','',functions,root.RooArgList(morph),1.,2)
    observations=histogram('observations',[6.,9.,8.,7.,10.,12.])
    data=root.RooDataHist('data_obs','',root.RooArgList(x,y),observations)
    workspace=root.RooWorkspace('w')
    for obj in (signal,background):getattr(workspace,'import')(obj,root.RooFit.RecycleConflictNodes(),root.RooFit.Silence())
    getattr(workspace,'import')(data,root.RooFit.Silence())
    workspace.writeToFile(str(directory/'shapes.root'))
    card=directory/'card.txt'
    card.write_text('''imax 1
jmax 1
kmax *
shapes * ch shapes.root w:$PROCESS
bin ch
observation -1
bin ch ch
process signal background
process 0 1
rate 8 43
lumi lnN 1.05 1.05
morph param 0 1
'''+''.join(f'bin{i} param {value} 1\n' for i,value in enumerate(means)))
    output=directory/'workspace.root'
    run=subprocess.run(['text2workspace.py',str(card),'-o',str(output)],cwd=directory,capture_output=True,text=True)
    (directory/'text2workspace.log').write_text(run.stdout+run.stderr)
    # TList::Clear inspects its elements even when it does not own them. Unlink
    # them while the Python-owned histograms are still alive, before locals die.
    templates.Clear('nodelete')
    run.check_returncode()
    return output


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('directory',nargs='?',default='build/tensor_binned_fixture')
    print(build_binned_fixture(parser.parse_args().directory))
