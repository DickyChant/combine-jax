"""Save generic autoMCStats comparisons with and without analytic profiling."""

import json
from pathlib import Path

from build_template_fixture import build_template_fixture
from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_workspace


def main():
    path = build_template_fixture("build/template_fixture")
    reports = {}
    for profile in [False, True]:
        with open_workspace(path) as workspace:
            report = validate_workspace(workspace, seed=1819, analytic_barlow_beeston=profile)
        reports["profiled" if profile else "explicit"] = report
    Path("reports/templates.json").write_text(json.dumps(reports, indent=2)+"\n")
    for label, report in reports.items():
        print(label, {k: v for k, v in report.items() if k.startswith("max_")}, flush=True)
        assert report["max_nll_absolute_error"] < 1e-6
        assert report["max_gradient_relative_error"] < 1e-4
        assert report["max_hessian_absolute_error"] < 1e-6


if __name__ == "__main__":
    main()
