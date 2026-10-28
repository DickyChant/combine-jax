"""Build a v11 fixture and save reproducible absolute NLL/derivative comparisons."""

import json
from pathlib import Path
import subprocess

from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_counting


def main():
    output = Path("reports/counting.json")
    output.parent.mkdir(exist_ok=True)
    Path("build").mkdir(exist_ok=True)
    command = ["text2workspace.py", "tests/cards/counting.txt", "-o", "build/counting.root"]
    subprocess.run(command, check=True)
    with open_workspace("build/counting.root") as workspace:
        report = validate_counting(workspace)
    report["workspace_command"] = command
    output.write_text(json.dumps(report, indent=2) + "\n")
    print({k: v for k, v in report.items() if k.startswith("max_")})
    assert report["max_nll_absolute_error"] < 1e-6
    assert report["max_channel_absolute_error"] < 1e-6
    assert report["max_gradient_relative_error"] < 1e-4
    assert report["max_hessian_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
