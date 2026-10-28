"""Compare stable real-sum deltas with the live v11 channel, including bin profiles."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from build_template_fixture import build_template_fixture
from cached_root_real_sum import CachedRootRealSum
from caching_nll_diagnostics import channel_nlls
from combine_jax.likelihood import from_workspace
from combine_jax.root_io import load_caching_nll, open_workspace


@pytest.fixture(scope="module")
def real_sum_workspace(tmp_path_factory):
    return build_template_fixture(tmp_path_factory.mktemp("cached_real_sum"))


@pytest.mark.parametrize("profile", (False, True))
def test_cached_real_sum_native_deltas(real_sum_workspace, profile):
    root = load_caching_nll()
    try:
        with open_workspace(real_sum_workspace) as workspace:
            likelihood = from_workspace(workspace, analytic_barlow_beeston=profile)
            model = workspace.obj("ModelConfig")
            parent = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"),
                                                   model.GetNuisanceParameters())
            parent.setAnalyticBarlowBeeston(profile)
            parent.clearConstantZeroPoint()
            parent.clearZeroPoint()
            parts, _ = channel_nlls(root, parent)
            assert len(parts) == 1
            predictor = CachedRootRealSum(root, parts[0])
            initial = likelihood.initial
            variables = [workspace.var(n) for n in likelihood.parameter_names]
            scales = np.array([p.error if p.error > 0 else max(abs(p.value), 1.)
                               for p in likelihood.parameters])
            initial_full = float(parent.getVal())
            baseline = predictor.predict()
            initial_channel = float(parts[0].getVal())
            assert predictor.report["basic_integrals"] == 2
            assert predictor.report["histogram_propagators"] == 1
            assert predictor.report["rows"] == 5
            assert predictor.weights[-1] == 0
            # Native histogram floors remain present even in the empty bin.
            np.testing.assert_allclose(baseline[1][-1], 2e-9, rtol=1e-14, atol=0.)
            parent.setZeroPoint()
            rng = np.random.default_rng(28175)
            points = [initial]
            for _ in range(12):
                theta = initial+rng.uniform(-.4, .4, len(initial))*scales
                low = np.array([p.minimum for p in likelihood.parameters])
                high = np.array([p.maximum for p in likelihood.parameters])
                points.append(np.clip(theta, low, high))
            points += [initial+step*scales for step in (1e-4, 1e-5, 1e-6)]
            for theta in points:
                for variable, value in zip(variables, theta, strict=True):
                    variable.setVal(float(value))
                full = float(parent.getVal())
                channel = float(parts[0].getVal())
                delta = float(predictor.delta(baseline))
                assert abs(delta-channel) < 1e-10
                assert float(parent.getVal()) == full
                # Leave every constraint contribution exactly as evaluated by
                # native Combine, including any analytically profiled bins.
                stable_full = full-channel+delta
                assert abs(stable_full-full) < 1e-10
            for variable, value in zip(variables, initial, strict=True):
                variable.setVal(float(value))
            assert abs(float(parent.getVal())) < 1e-10
            assert abs(float(predictor.delta(baseline))) < 1e-10
            parent.clearZeroPoint()
            assert abs(float(parent.getVal())-initial_full) < 1e-10
            assert abs(float(parts[0].getVal())-initial_channel) < 1e-10
            del predictor, parent
    finally:
        root.runtimedef.set("ADDNLL_ROOREALSUM_BASICINT", 1)


def test_real_sum_native_integrals_without_binned_eligibility():
    root = load_caching_nll()
    x = root.RooRealVar("cached_rs_x", "", -2., 3.)
    slope = root.RooRealVar("cached_rs_slope", "", -.2, -.8, .4)
    signal = root.RooFormulaVar("cached_rs_signal", "", "exp(@0*@1)", root.RooArgList(slope, x))
    flat = root.RooFormulaVar("cached_rs_flat", "", "1+0*@0", root.RooArgList(x))
    ns = root.RooRealVar("cached_rs_ns", "", 2., .1, 6.)
    nb = root.RooConstVar("cached_rs_nb", "", 3.)
    pdf = root.RooRealSumPdf("cached_rs_pdf", "", root.RooArgList(signal, flat), root.RooArgList(ns, nb), True)
    weight = root.RooRealVar("cached_rs_weight", "", 1., 0., 100.)
    columns = root.RooArgSet(x, weight)
    data = root.RooDataSet("cached_rs_data", "", columns, root.RooFit.WeightVar(weight))
    for coordinate, w in ((-1.5, .2), (-.2, 0.), (.7, 2.5), (2.5, 3.)):
        x.setVal(coordinate)
        weight.setVal(w)
        data.add(columns, w)
    nll = root.cacheutils.CachingAddNLL("cached_rs_nll", "", pdf, data, True)
    nll.clearConstantZeroPoint()
    nll.clearZeroPoint()
    predictor = CachedRootRealSum(root, nll)
    baseline = predictor.predict()
    assert predictor.report["basic_integrals"] == 0
    assert predictor.report["histogram_propagators"] == 0
    assert predictor.report["rows"] == 4 and predictor.weights[1] == 0
    nll.setZeroPoint()
    for coefficient, value in ((2., -.2), (3., .1), (.7, -.6), (2.0001, -.20001)):
        ns.setVal(coefficient)
        slope.setVal(value)
        assert abs(float(predictor.delta(baseline))-float(nll.getVal())) < 1e-10
    del predictor, nll


def test_real_sum_rejects_changed_normalization_mode(real_sum_workspace):
    root = load_caching_nll()
    with open_workspace(real_sum_workspace) as workspace:
        model = workspace.obj("ModelConfig")
        parent = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"),
                                               model.GetNuisanceParameters())
        parts, _ = channel_nlls(root, parent)
        predictor = CachedRootRealSum(root, parts[0])
        try:
            root.runtimedef.set("ADDNLL_ROOREALSUM_NONORM", 0)
            with pytest.raises(NotImplementedError, match="NONORM=1"):
                predictor.predict()
            with pytest.raises(NotImplementedError, match="NONORM=1"):
                CachedRootRealSum(root, parts[0])
        finally:
            root.runtimedef.set("ADDNLL_ROOREALSUM_NONORM", 1)
        with pytest.raises(TypeError, match="live CachingAddNLL"):
            CachedRootRealSum(root, object())
        del predictor, parent
