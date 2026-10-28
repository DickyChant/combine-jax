import numpy as np

from combine_jax.likelihood import from_workspace
from combine_jax.root_io import open_workspace
from combine_jax.validation import validate_counting


def test_full_and_channel_nll_gradients_hessians(counting_path):
    with open_workspace(counting_path) as workspace:
        report = validate_counting(workspace)
    assert report["max_nll_absolute_error"] < 1e-6
    assert report["max_channel_absolute_error"] < 1e-6
    assert report["max_gradient_relative_error"] < 1e-4
    assert report["max_hessian_absolute_error"] < 1e-6


def test_stable_parameters_global_observables_and_offsets(counting_path):
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
        assert likelihood.parameter_names == ("bkg", "lumi", "r")
        assert likelihood.constraints == ("bkg_Pdf", "lumi_Pdf")
        assert len(likelihood.channels) == 2
        zeroed = likelihood.at_zero(likelihood.initial)
    # Evaluation remains valid after ROOT file and graph ownership have ended.
    assert abs(float(zeroed.value(likelihood.initial))) < 1e-14
    point = likelihood.initial + np.array([.3, -.7, .5])
    np.testing.assert_allclose(zeroed.value_and_grad(point)[1], likelihood.value_and_grad(point)[1], rtol=1e-12)
    np.testing.assert_allclose(zeroed.hessian(point), likelihood.hessian(point), rtol=1e-12)
