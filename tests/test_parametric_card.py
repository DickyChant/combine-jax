import importlib.util
from pathlib import Path

from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_workspace


def test_full_parametric_card_against_combine(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/build_parametric_fixture.py"
    spec = importlib.util.spec_from_file_location("build_parametric_fixture", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = module.build_parametric_fixture(tmp_path)
    with open_workspace(path) as workspace:
        report = validate_workspace(workspace, random_points=25, seed=1820)
    assert len(report["cases"]) == 26
    assert len(report["cases"][0]["channels"]) == 2
    assert report["max_nll_absolute_error"] < 1e-6
    assert report["max_channel_absolute_error"] < 1e-6
    assert report["max_gradient_relative_error"] < 1e-4
    assert report["max_hessian_scaled_error"] < 1e-6
