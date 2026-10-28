"""Explicit ROOT import conventions; numerical models keep float64 arrays.

The v10 policy reproduces ROOT 6.30's histogram conversion at the import boundary,
including its native TH1F/TH2F/TH3F serialization rounding. It never casts model
buffers or numerical kernels to float32. The original native-stack oracle remains
available independently of this correction.
"""
from functools import lru_cache

NATIVE = 'native'
V10_DENSITY = 'v10_density'
CONVENTIONS = (NATIVE, V10_DENSITY)


def check_convention(value):
    if value not in CONVENTIONS:
        raise ValueError(f'Unknown template convention: {value}')


@lru_cache(maxsize=1)
def density_histogram():
    from .root_io import load_root
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "RooHistPdf.h"
      #include "RooRealVar.h"
      #include "TH3.h"
      #include <memory>
      namespace combine_jax_io {
      TH1* densityHistogram(RooHistPdf& pdf, RooArgList const& axes, bool conditional) {
        int dim=axes.size();
        if(dim<1 || dim>3) throw std::invalid_argument("Expected 1-3 histogram axes");
        auto& x=dynamic_cast<RooRealVar&>(*axes.at(0));
        RooCmdArg y=dim>1 ? RooFit::YVar(dynamic_cast<RooRealVar&>(*axes.at(1))) : RooCmdArg::none();
        RooCmdArg z=dim>2 ? RooFit::ZVar(dynamic_cast<RooRealVar&>(*axes.at(2))) : RooCmdArg::none();
        std::unique_ptr<TH1> h(x.createHistogram("combine_jax_density",y,z));
        h->SetDirectory(nullptr);
        RooArgSet norm;
        double averageVolume=1.; std::vector<double> saved;
        for(int i=0;i<dim;++i) {
          auto& a=dynamic_cast<RooRealVar&>(*axes.at(i));
          saved.push_back(a.getVal());
          averageVolume *= (a.getMax()-a.getMin())/a.getBins();
          if(!conditional || i) norm.add(a);
        }
        // ROOT 6.30 fillHistogram scales a PDF density by the constant average
        // volume before storing it in TH*F. New ROOT scales by each bin volume.
        for(int ix=1;ix<=h->GetNbinsX();++ix)
          for(int iy=1;iy<=h->GetNbinsY();++iy)
            for(int iz=1;iz<=h->GetNbinsZ();++iz) {
              x.setVal(h->GetXaxis()->GetBinCenter(ix));
              if(dim>1) dynamic_cast<RooRealVar&>(*axes.at(1)).setVal(h->GetYaxis()->GetBinCenter(iy));
              if(dim>2) dynamic_cast<RooRealVar&>(*axes.at(2)).setVal(h->GetZaxis()->GetBinCenter(iz));
              h->SetBinContent(h->GetBin(ix,iy,iz),pdf.getVal(&norm)*averageVolume);
            }
        for(int i=0;i<dim;++i) dynamic_cast<RooRealVar&>(*axes.at(i)).setVal(saved[i]);
        return h.release();
      }
      }
    '''):
        raise RuntimeError('Cannot compile density-preserving histogram importer')
    return root.combine_jax_io.densityHistogram


def corrected_histograms(pdf, axes, conditional=False):
    """Own temporary native histograms; never alter original templates/caches."""
    from .root_io import load_root
    root = load_root()
    convert = density_histogram()
    histograms = []
    for component in pdf.funcList():
        if (component.ClassName() != 'RooHistPdf' or component.getInterpolationOrder() != 0
                or component.haveUnitNorm() or component.getCdfBoundaries()):
            from .graph import UnsupportedModelError
            raise UnsupportedModelError('Density convention requires fixed, ordinary, order-zero RooHistPdf inputs')
        h = convert(component, root.RooArgList(*axes), conditional)
        root.SetOwnership(h, True)
        histograms.append(h)
    return histograms
