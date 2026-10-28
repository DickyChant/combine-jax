import importlib.util
from pathlib import Path

import numpy as np
import pytest

from combine_jax.graph import UnsupportedModelError
from combine_jax.likelihood import from_workspace
from combine_jax.root_io import load_caching_nll,open_workspace
from combine_jax.validation import validate_workspace


@pytest.fixture(scope="module")
def multipdf_path(tmp_path_factory):
    path=Path(__file__).resolve().parents[1]/"scripts/build_multipdf_fixture.py"
    spec=importlib.util.spec_from_file_location("build_multipdf_fixture",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.build_multipdf_fixture(tmp_path_factory.mktemp("multipdf"))


@pytest.mark.parametrize("index,masked", [(0,0),(1,0),(0,1),(1,1),(1,-2)])
def test_multipdf_masked_card_nll_and_derivatives(multipdf_path,index,masked):
    with open_workspace(multipdf_path) as w:
        w.cat("pdf_index").setIndex(index);w.var("mask_choice").setRange(-3.,3.);w.var("mask_choice").setVal(masked)
        # Explicit fixture scales, preserved independently of text2workspace's metadata updates.
        for name,error in (("multi_slope",.005),("multi_c1",.1),("multi_c2",.1)):w.var(name).setError(error)
        likelihood=from_workspace(w)
        assert all(not n.startswith("mask_") for n in likelihood.parameter_names)
        if masked:
            assert not likelihood.discrete_choices
            assert not any(n.startswith("multi_") for n in likelihood.parameter_names)
        elif index==0:
            assert "multi_slope" in likelihood.parameter_names and "multi_c1" not in likelihood.parameter_names
        else:
            assert "multi_slope" not in likelihood.parameter_names and "multi_c1" in likelihood.parameter_names
        report=validate_workspace(w,random_points=8,seed=7711)
    assert report["max_nll_absolute_error"]<1e-6
    assert report["max_channel_absolute_error"]<1e-6
    assert report["max_gradient_relative_error"]<1e-4
    assert report["max_hessian_scaled_error"]<1e-6


def test_discrete_and_mask_choices_are_frozen_and_survive_file_close(multipdf_path):
    with open_workspace(multipdf_path) as w:
        first=from_workspace(w);first_value=float(first.value(first.initial))
        w.cat("pdf_index").setIndex(1);second=from_workspace(w)
        assert first.discrete_choices!=second.discrete_choices
        w.var("mask_choice").setVal(1.);w.var("mask_control").setVal(1.)
        masked=from_workspace(w)
        assert all(c.masked for c in masked.channels)
        assert first_value==float(first.value(first.initial))
    assert np.isfinite(second.value(second.initial))
    assert abs(float(masked.at_zero(masked.initial).value(masked.initial)))<1e-12


def test_mask_controls_cannot_silently_freeze_shape_parameters(multipdf_path):
    root=load_caching_nll()
    with open_workspace(multipdf_path) as w:
        mask=root.RooFormulaVar("bad_mask","@0>0",root.RooArgList(w.var("r")))
        zero=root.RooConstVar("zero_mask","",0.)
        w.pdf("model_s").addChannelMasks(root.RooArgList(mask,zero))
        with pytest.raises(UnsupportedModelError,match="Mask controls also affect"):
            from_workspace(w)


@pytest.mark.parametrize("hessian_chunk_size", [None, 1, 2, 8])
def test_local_discrete_selection_masks_and_constant_terms(multipdf_path, hessian_chunk_size):
    from test_local import compare_local

    with open_workspace(multipdf_path) as w:
        w.cat("pdf_index").setIndex(1)
        likelihood = from_workspace(w)
        local = compare_local(likelihood, [likelihood.initial], hessian_chunk_size=hessian_chunk_size)
        control = next(t for t in local.terms if t.name == "control")
        assert not any(n.startswith("multi_") for n in control.parameter_names)
        assert "multi_slope" not in local.parameter_names
        w.var("mask_choice").setVal(1.)
        w.var("mask_control").setVal(1.)
        for n in ("lumi", "background_norm"):
            w.var(n).setConstant(True)
        constant = from_workspace(w)
        assert not constant.parameter_names
        compare_local(constant, [constant.initial], hessian_chunk_size=hessian_chunk_size)
