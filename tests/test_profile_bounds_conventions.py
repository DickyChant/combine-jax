"""Explicit legacy clipping, without changing modern range rejection."""
from pathlib import Path

import jax
import numpy as np
import pytest

from combine_jax.root_io import load_caching_nll,open_workspace
from combine_jax.likelihood import from_workspace as oracle_import
from combine_jax.tensor.model import from_workspace,make_model_function,evaluate_model_numpy,unpack_model
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.templates import profile_solution,validate_templates
from combine_jax.validation import finite_difference
from test_bin_statistics import template_path


def test_legacy_profile_clipping_matches_native_nll_and_derivatives(template_path,tmp_path):
    root=load_caching_nll()
    # ROOT 6.40 exposes this explicit compatibility switch. It is enabled only
    # in this test and restored; production import does not alter ROOT globals.
    root.RooRealVar.enableSilentClipping(True)
    try:
        with open_workspace(template_path) as w:
            beta=w.var('prop_binch_bin0');beta.setRange(-.03,.03)
            model=from_workspace(w,profile_bounds_policy='clip')
            oracle=oracle_import(w,profile_bounds_policy='clip')
            assert model.metadata['provenance']['profile_bounds_policy']=='clip'
            model.save(tmp_path/'model');model=TensorBundle.load(tmp_path/'model')
            assert unpack_model(model)['templates'].arrays['profile_clip'].all()
            fn=make_model_function(model);arrays=jax.device_put(dict(model.arrays))
            value=jax.jit(fn);grad=jax.jit(jax.grad(fn));hess=jax.jit(jax.hessian(fn))
            config=w.obj('ModelConfig');nll=root.cacheutils.CachingSimNLL(config.GetPdf(),w.data('data_obs'),config.GetNuisanceParameters())
            nll.setAnalyticBarlowBeeston(True);nll.clearZeroPoint();nll.clearConstantZeroPoint()
            def native(point):
                for name,x in zip(model.metadata['parameter_names'],point):w.var(name).setVal(float(x))
                return float(nll.getVal())
            clipped=[]
            for rate in (.5,1.,1.7):
                point=model.arrays['scalars__parameter_initial'].copy()
                point[model.metadata['parameter_names'].index('r')]=rate
                expected=native(point);clipped.append(float(beta.getVal()))
                np.testing.assert_allclose(value(point,arrays),expected,atol=1e-10,rtol=0.)
                np.testing.assert_allclose(evaluate_model_numpy(model,point),expected,atol=1e-10,rtol=0.)
                np.testing.assert_allclose(oracle.value(point),expected,atol=1e-10,rtol=0.)
                np.testing.assert_allclose(grad(point,arrays),finite_difference(native,point,np.full(len(point),2e-4)),atol=1e-8,rtol=1e-4)
                np.testing.assert_allclose(grad(point,arrays),oracle.value_and_grad(point)[1],atol=1e-10,rtol=1e-10)
                np.testing.assert_allclose(hess(point,arrays),oracle.hessian(point),atol=1e-9,rtol=1e-9)
            assert -.03 in clipped and .03 in clipped
            nll.setAnalyticBarlowBeeston(False)
    finally:root.RooRealVar.enableSilentClipping(False)


def test_profile_policy_is_explicit_and_zero_error_still_rejected(template_path):
    assert np.isnan(profile_solution(100.,2.,1.,0.,-.03,.03))
    assert profile_solution(100.,2.,1.,0.,-.03,.03,clip_to_bounds=True)==-.03
    assert np.isnan(profile_solution(10.,0.,10.,0.,-7.,7.,clip_to_bounds=True))
    with open_workspace(template_path) as w:
        with pytest.raises(ValueError,match='bounds policy'):from_workspace(w,profile_bounds_policy='guess')
        with pytest.raises(ValueError,match='bounds policy'):oracle_import(w,profile_bounds_policy='guess')
        model=from_workspace(w)
    t=unpack_model(model)['templates']
    assert 'profile_clip' not in t.arrays
    wrong=dict(t.arrays);wrong['profile_clip']=np.ones(len(t.metadata['profile_names']))
    with pytest.raises(ValueError,match='clipping policy'):validate_templates(TensorBundle(t.metadata,wrong))
