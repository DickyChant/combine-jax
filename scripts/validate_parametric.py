"""Log channel/global NLL and derivative agreement for the v11 parametric card."""

import json
from pathlib import Path

from build_parametric_fixture import build_parametric_fixture
from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_workspace


def main():
    path = build_parametric_fixture("build/parametric_fixture")
    with open_workspace(path) as workspace:
        report = validate_workspace(workspace, seed=1820)
    Path("reports/parametric.json").write_text(json.dumps(report, indent=2)+"\n")
    print({k: v for k, v in report.items() if k.startswith("max_")}, flush=True)
    assert report["max_nll_absolute_error"] < 1e-6
    assert report["max_channel_absolute_error"] < 1e-6
    assert report["max_gradient_relative_error"] < 1e-4
    assert report["max_hessian_scaled_error"] < 1e-6


if __name__ == "__main__":
    main()
