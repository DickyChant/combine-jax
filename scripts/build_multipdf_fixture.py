"""A two-channel v11 card with discrete alternatives and channel-mask controls."""

from pathlib import Path
import subprocess

import numpy as np

from combine_jax.root_io import load_root


def build_multipdf_fixture(directory):
    root=load_root();directory=Path(directory).resolve();directory.mkdir(parents=True,exist_ok=True)
    workspace=root.RooWorkspace("w")
    slope=root.RooRealVar("multi_slope","",-.02,-.06,.02);slope.setError(.005)
    c0=root.RooConstVar("multi_c0","",1.)
    c1=root.RooRealVar("multi_c1","",1.2,.2,3.);c1.setError(.1)
    c2=root.RooRealVar("multi_c2","",.8,.2,3.);c2.setError(.1)
    rng=np.random.default_rng(7710)
    for channel in ("choice","control"):
        x=root.RooRealVar("mass_"+channel,"",125.,100.,180.);x.setBins(20)
        mean=root.RooConstVar("signal_mean_"+channel,"",125.)
        sigma=root.RooConstVar("signal_sigma_"+channel,"",4.)
        signal=root.RooGaussian("signal_"+channel,"",x,mean,sigma)
        if channel=="choice":
            category=root.RooCategory("pdf_index","PDF choice")
            expo=root.RooExponential("alternative_exp","",x,slope)
            bern=root.RooBernstein("alternative_bern","",x,root.RooArgList(c0,c1,c2))
            background=root.RooMultiPdf("background_choice","",category,root.RooArgList(expo,bern))
            background.setCorrectionFactor(.7)
        else:
            background=root.RooUniform("background_control","",root.RooArgSet(x))
        norm=root.RooArgSet(x)
        data=root.RooDataHist("data_"+channel,"",root.RooArgList(x))
        for coordinate in np.arange(102.,180.,4.):
            x.setVal(float(coordinate))
            expected=4*(8*signal.getVal(norm)+60*background.getVal(norm))
            data.add(norm,float(rng.poisson(expected)))
        getattr(workspace,"import")(signal,root.RooFit.RecycleConflictNodes())
        getattr(workspace,"import")(background,root.RooFit.RecycleConflictNodes())
        getattr(workspace,"import")(data)
    workspace.writeToFile(str(directory/"shapes.root"))
    card=directory/"card.txt"
    card.write_text("""imax 2
jmax 1
kmax *
------------
shapes * * shapes.root w:$PROCESS_$CHANNEL
shapes data_obs * shapes.root w:data_$CHANNEL
------------
bin choice control
observation -1 -1
------------
bin choice choice control control
process signal background signal background
process 0 1 0 1
rate 8 60 8 60
------------
lumi lnN 1.025 - 1.025 -
background_norm lnN - 1.10 - 1.10
pdf_index discrete
multi_slope flatParam
multi_c1 flatParam
multi_c2 flatParam
""")
    path=directory/"workspace.root"
    result=subprocess.run(["text2workspace.py",str(card),"--channel-masks","-o",str(path)],cwd=directory,text=True,capture_output=True)
    (directory/"text2workspace.log").write_text(result.stdout+result.stderr);result.check_returncode()
    return path


if __name__=="__main__":print(build_multipdf_fixture("build/multipdf_fixture"))
