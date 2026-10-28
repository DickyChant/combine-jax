"""Build a v11 signal-plus-background card with two analytic shape families."""

from pathlib import Path
import subprocess

import numpy as np

from combine_jax.root_io import load_root


def build_parametric_fixture(directory):
    root = load_root()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    workspace = root.RooWorkspace("w")
    mean_shift = root.RooRealVar("mean_shift", "", 0., -5., 5.)
    width_shift = root.RooRealVar("width_shift", "", 0., -5., 5.)
    mean = root.RooFormulaVar("signal_mean", "125+0.3*@0", root.RooArgList(mean_shift))
    sigma = root.RooFormulaVar("signal_sigma", "2.5*exp(0.1*@0)", root.RooArgList(width_shift))
    slope = root.RooRealVar("slope", "", -.02, -.08, .04)
    c0 = root.RooConstVar("bern_zero", "", 1.)
    c1 = root.RooRealVar("bern_one", "", 1.2, .1, 3.)
    c2 = root.RooRealVar("bern_two", "", .8, .1, 3.)
    rng = np.random.default_rng(1124)
    for channel, signal_yield, background_yield in (("expo", 12., 80.), ("bern", 8., 65.)):
        x = root.RooRealVar(f"mass_{channel}", "", 125., 100., 180.)
        x.setBins(40)
        signal = root.RooGaussian(f"signal_{channel}", "", x, mean, sigma)
        background = (root.RooExponential(f"background_{channel}", "", x, slope) if channel == "expo"
                      else root.RooBernstein(f"background_{channel}", "", x, root.RooArgList(c0, c1, c2)))
        observables = root.RooArgSet(x)
        data = root.RooDataHist(f"data_{channel}", "", root.RooArgList(x))
        for coordinate in np.arange(101., 180., 2.):
            x.setVal(float(coordinate))
            expected = 2*(signal_yield*signal.getVal(observables)+background_yield*background.getVal(observables))
            data.add(observables, float(rng.poisson(expected)))
        getattr(workspace, "import")(signal, root.RooFit.RecycleConflictNodes())
        getattr(workspace, "import")(background, root.RooFit.RecycleConflictNodes())
        getattr(workspace, "import")(data)
    workspace.writeToFile(str(directory / "shapes.root"))
    card = directory / "card.txt"
    card.write_text("""imax 2
jmax 1
kmax *
------------
shapes * * shapes.root w:$PROCESS_$CHANNEL
shapes data_obs * shapes.root w:data_$CHANNEL
------------
bin expo bern
observation -1 -1
------------
bin expo expo bern bern
process signal background signal background
process 0 1 0 1
rate 12 80 8 65
------------
lumi lnN 1.025 - 1.025 -
background_norm lnN - 1.10 - 1.10
mean_shift param 0 1
width_shift param 0 1
slope param -0.02 0.005
bern_one param 1.2 0.2
bern_two param 0.8 0.2
""")
    path = directory / "workspace.root"
    run = subprocess.run(["text2workspace.py", str(card), "-o", str(path)],
                         cwd=directory, text=True, capture_output=True)
    (directory / "text2workspace.log").write_text(run.stdout+run.stderr)
    run.check_returncode()
    return path


if __name__ == "__main__":
    print(build_parametric_fixture("build/parametric_fixture"))
