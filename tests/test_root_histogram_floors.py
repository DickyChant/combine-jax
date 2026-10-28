"""Verify actual ROOT bins across a known histogram-floor transition."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from root_histogram_floors import floor_view_class


@pytest.mark.parametrize("conditional", (False,True))
def test_raw_floor_transition_and_normalized_root_values(root, conditional):
    view_class = floor_view_class()
    tag="conditional" if conditional else "joint"
    x=root.RooRealVar("floor_x_"+tag,"",.5,0.,2.)
    y=root.RooRealVar("floor_y_"+tag,"",.5,0.,2.)
    a=root.RooRealVar("floor_a_"+tag,"",0.,-1.,1.)
    histograms=[]; templates=root.TList()
    for i,values in enumerate(((0.,1.,2.,3.),(1.,1.,2.,3.),(0.,1.,2.,3.))):
        hist=root.TH2D(f"floor_{tag}_h{i}","",2,0.,2.,2,0.,2.)
        hist.SetDirectory(0)
        for cell,value in zip(np.ndindex(2,2),values,strict=True):
            hist.SetBinContent(cell[0]+1,cell[1]+1,value)
        templates.Add(hist);histograms.append(hist)
    pdf=root.FastVerticalInterpHistPdf2D2("floor_pdf_"+tag,"",x,y,conditional,templates,root.RooArgList(a),1.,1)
    view=view_class(pdf)
    norm=root.RooArgSet(x,y)
    states={}
    for theta in (-.01,0.,1e-10,1e-6,.01):
        a.setVal(theta)
        raw,rebuilt,cached=np.asarray(view.bins(),dtype=float)
        values=[]
        for i,j in np.ndindex(2,2):
            x.setVal(i+.5);y.setVal(j+.5);values.append(float(pdf.getVal(norm)))
        np.testing.assert_array_equal(rebuilt,cached)
        np.testing.assert_array_equal(cached,values)
        states[theta]=bool(raw[0]<1e-9)
    assert states=={-.01:True,0.:True,1e-10:True,1e-6:False,.01:False}
    # Destroy copied PDFs before their borrowed parameter/observable servers.
    del view,pdf
