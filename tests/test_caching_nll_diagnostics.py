"""Compare borrowed live channels with independently constructed channel NLLs."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from caching_nll_diagnostics import channel_nlls
from combine_jax.root_io import load_caching_nll, open_workspace


def test_live_channels_match_independent_counting_nlls(counting_path):
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        model = workspace.obj("ModelConfig")
        parent = root.cacheutils.CachingSimNLL(model.GetPdf(), workspace.data("data_obs"),
                                              model.GetNuisanceParameters())
        parent.clearConstantZeroPoint(); parent.clearZeroPoint()
        channels, layout = channel_nlls(root,parent)
        assert len(channels) == 2 and layout["member_offset"] > 0
        split = workspace.data("data_obs").split(model.GetPdf(),True)
        subsets = {str(s.GetName()):s for s in split}
        independent = []
        for label in ("signal_region", "control_region"):
            # The full channel contains constraints; the borrowed obsOnly PDF
            # is the same factorization used by Combine's simultaneous NLL.
            borrowed = next(c for c in channels if str(c.pdf().GetName()) == "pdf_bin"+label+"_obsOnly")
            nll = root.cacheutils.CachingAddNLL("independent_"+label,"",borrowed.pdf(),subsets[label],True)
            nll.clearConstantZeroPoint(); nll.clearZeroPoint()
            independent.append(nll)
        for signal,bkg in ((0.,-.8),(.5,-.2),(1.,0.),(2.,.7)):
            workspace.var("r").setVal(signal); workspace.var("bkg").setVal(bkg)
            full = float(parent.getVal())
            for channel in channels:
                label = str(channel.pdf().GetName()).removeprefix("pdf_bin").removesuffix("_obsOnly")
                other = next(n for n in independent if str(n.GetName()) == "independent_"+label)
                assert float(channel.getVal()) == float(other.getVal())
            assert full == float(parent.getVal())
        parent.setZeroPoint()
        np.testing.assert_array_equal([float(c.getVal()) for c in channels],0.)
        assert abs(float(parent.getVal())) < 1e-12
        # Drop borrowed proxies before their owning simultaneous NLL.
        independent.clear(); channels.clear()
        del other, channel, borrowed, nll, parent


def test_wrong_parent_type_rejected(root):
    from combine_jax.root_io import load_caching_nll
    load_caching_nll()
    with pytest.raises(TypeError, match="CachingSimNLL"):
        channel_nlls(root, root.RooRealVar("diagnostic_wrong", "", 1.))
