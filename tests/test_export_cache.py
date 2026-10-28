import importlib.util
from pathlib import Path

import numpy as np
import pytest

from combine_jax.likelihood import from_workspace
from combine_jax.root_io import open_workspace


def test_benchmark_export_cache_roundtrip(counting_path, tmp_path):
    path = Path(__file__).resolve().parents[1]/"scripts/hhvv_export_cache.py"
    spec = importlib.util.spec_from_file_location("hhvv_export_cache", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    with open_workspace(counting_path) as workspace:
        likelihood = from_workspace(workspace)
    source = dict(test="source identity", configuration={"MX": 2000})
    cache = tmp_path/"export.pkl"
    module.save_export(likelihood, cache, source)
    restored = module.load_export(cache, source)
    point = likelihood.initial+np.array([.3, -.7, .5])
    assert restored.parameter_names == likelihood.parameter_names
    for actual, expected in zip(restored.value_and_grad(point), likelihood.value_and_grad(point), strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(restored.hessian(point), likelihood.hessian(point))
    assert all(not c.data.weights.flags.writeable for c in restored.channels)
    with pytest.raises(ValueError, match="does not match"):
        module.load_export(cache, dict(test="wrong source"))
