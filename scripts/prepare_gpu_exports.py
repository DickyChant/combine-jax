"""Export GPU inputs locally with the certified Combine v11 / ROOT 6.40 stack."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import subprocess

from build_template_fixture import build_template_fixture
from combine_jax.likelihood import from_workspace
from combine_jax.root_io import open_workspace, load_root
from combine_jax.validation import validate_workspace
from hhvv_export_cache import save_export


def source_identity(path, configuration=None):
    with Path(path).open("rb") as handle:
        return dict(sha256=hashlib.file_digest(handle, "sha256").hexdigest(), configuration=configuration or {})


def main():
    directory = Path("build/gpu_exports")
    directory.mkdir(parents=True, exist_ok=True)
    counting = directory/"counting.root"
    subprocess.run(["text2workspace.py", "tests/cards/counting.txt", "-o", str(counting)], check=True)
    templates = build_template_fixture(directory/"automcstats_source")
    manifest = dict(combine_version=version("higgs-combine-tool"),
                    root_version=str(load_root().gROOT.GetVersion()), models={})
    for label, path in (("counting", counting), ("automcstats", templates)):
        with open_workspace(path) as workspace:
            likelihood = from_workspace(workspace)
            source = source_identity(path)
            save_export(likelihood, directory/(label+".pkl"), source)
            reference = validate_workspace(workspace, random_points=25, seed=1819)
        assert reference["max_nll_absolute_error"] < 1e-6
        assert reference["max_gradient_relative_error"] < 1e-4
        manifest["models"][label] = dict(source=source, cache=label+".pkl", reference=reference)
        print(label, "validated and exported", flush=True)
    nominal = json.loads(Path("reports/hhvv_nominal.json").read_text())
    assert nominal["status"] == "passed"
    source = source_identity("inputs/workspace_MX2000.root", nominal["configuration"])
    shutil.copyfile("build/hhvv_export.pkl", directory/"hhvv.pkl")
    manifest["models"]["hhvv"] = dict(source=source, cache="hhvv.pkl", reference=dict(
        cases=[dict(theta=nominal["initial"], combine_nll=nominal["combine_nll"])],
        nominal_report="reports/hhvv_nominal.json",
        derivative_scope="GPU versus same-node CPU AD; full Combine FD gradient gate remains in the main goal"))
    (directory/"manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
