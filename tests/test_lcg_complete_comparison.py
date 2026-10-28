from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from compare_lcg_complete_benchmark import derivative_errors


def arrays():
    base = dict(parameter_names=np.array(["a", "b", "fixed"]),
        point=np.array([0., 1., 2.]), parameter_scales=np.array([1., 2., 3.]),
        value=np.array(123.), vg_value=np.array(123.), gradient=np.array([1., -2., 0.]),
        validation_points=np.tile([0., 1., 2.], (26, 1)),
        validation_labels=np.array([str(i) for i in range(26)]),
        validation_values=np.full(26, 123.), validation_vg_values=np.full(26, 123.),
        validation_gradients=np.tile([1., -2., 0.], (26, 1)), hessian=np.eye(3))
    return deepcopy(base), deepcopy(base), deepcopy(base)


def test_matching_complete_derivatives():
    c, g, h = arrays()
    result = derivative_errors(c, g, h)
    assert result["passed"] and result["compared_hessian_entries"] == 9


@pytest.mark.parametrize("error", ["value", "vg_value", "gradient", "validation_values",
                                  "validation_vg_values", "validation_gradients", "fixed_hessian"])
def test_numerical_failures_preserve_existing_tolerances(error):
    c, g, h = arrays()
    if error == "fixed_hessian":
        h["hessian"][2, 2] += 2e-6
    elif error in ("gradient", "validation_gradients"):
        g[error].flat[0] += 2e-4
    else:
        g[error].flat[0] += 2e-6
    assert not derivative_errors(c, g, h)["passed"]


@pytest.mark.parametrize("error", ["float32", "nan", "broadcast_hessian", "short_batch", "wrong_names", "wrong_points"])
def test_incomplete_or_misaligned_arrays_rejected(error):
    c, g, h = arrays()
    if error == "float32":
        g["gradient"] = g["gradient"].astype(np.float32)
    elif error == "nan":
        h["hessian"][0, 0] = np.nan
    elif error == "broadcast_hessian":
        h["hessian"] = h["hessian"][:1]
    elif error == "short_batch":
        g["validation_gradients"] = g["validation_gradients"][:25]
    elif error == "wrong_names":
        g["parameter_names"] = g["parameter_names"][::-1]
    elif error == "wrong_points":
        g["validation_points"][2, 1] += 1.
    with pytest.raises(AssertionError):
        derivative_errors(c, g, h)
