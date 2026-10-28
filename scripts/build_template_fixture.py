"""Create a small v11 template workspace exercising all autoMCStats bin types."""

from pathlib import Path
import subprocess

from combine_jax.root_io import load_root


def build_template_fixture(directory, *, threshold=10):
    root = load_root()
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    templates = directory / "templates.root"
    with root.TFile(str(templates), "RECREATE") as output:
        entries = {
            "signal": ([6., 3., 2., 1., 0.], [.6, 0., .3, 0., 0.]),
            "background": ([30., 3., .1, 8., 0.], [3., 2., 1., 0., 0.]),
            "data_obs": ([37., 7., 2., 7., 0.], [0.]*5),
        }
        for name, (values, errors) in entries.items():
            variations = [(name, values)]
            if name != "data_obs":
                variations += [(name+"_shapeUp", [v*f for v, f in zip(values, [1.07, 1.2, .8, 1.1, 1.])]),
                               (name+"_shapeDown", [v*f for v, f in zip(values, [.92, .8, 1.2, .9, 1.])])]
            for label, contents in variations:
                hist = root.TH1D(label, "", 5, 0., 5.)
                hist.SetDirectory(0)
                for index, (value, error) in enumerate(zip(contents, errors, strict=True), 1):
                    hist.SetBinContent(index, value)
                    hist.SetBinError(index, error)
                output.WriteObject(hist, label)
    card = directory / "card.txt"
    card.write_text(f"""imax 1
jmax 1
kmax *
------------
shapes * ch templates.root $PROCESS $PROCESS_$SYSTEMATIC
------------
bin ch
observation -1
------------
bin ch ch
process signal background
process 0 1
rate -1 -1
------------
lumi lnN 1.05 -
background_norm lnN - 1.10
shape shape 1 1
ch autoMCStats {threshold} 1 1
""")
    workspace = directory / "workspace.root"
    run = subprocess.run(["text2workspace.py", str(card), "-o", str(workspace)],
                         cwd=directory, capture_output=True, text=True)
    (directory / "text2workspace.log").write_text(run.stdout + run.stderr)
    run.check_returncode()
    return workspace


if __name__ == "__main__":
    print(build_template_fixture("build/template_fixture"))
