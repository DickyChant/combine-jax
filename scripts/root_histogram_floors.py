"""Read pre-floor morphs from copies of installed Combine histogram PDFs."""

from functools import lru_cache

from combine_jax.root_io import load_caching_nll


@lru_cache(maxsize=1)
def floor_view_class():
    root = load_caching_nll()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/VerticalInterpHistPdf.h"
      #include <stdexcept>
      namespace combine_jax_floor_diagnostic {
      class View : public FastVerticalInterpHistPdf2D2 {
      public:
        View(FastVerticalInterpHistPdf2D2 const& pdf)
          : FastVerticalInterpHistPdf2D2(pdf, (std::string(pdf.GetName())+"_floor_view").c_str()) {}
        std::vector<std::vector<double>> bins() const {
          if (_smoothAlgo < 0) throw std::invalid_argument("Only additive morphs have this floor");
          evaluate(); // Populate the copy's parameter and value caches.
          FastHisto2D raw(_cacheNominal);
          for (unsigned i=0; i<_morphs.size(); ++i) {
            double theta=_morphParams[i]->getVal();
            raw.Meld(_morphs[i].diff, _morphs[i].sum, 0.5*theta, smoothStepFunc(theta));
          }
          FastHisto2D rebuilt(raw);
          rebuilt.CropUnderflows();
          if (_conditional) rebuilt.NormalizeXSlices(); else rebuilt.Normalize();
          std::vector<std::vector<double>> result(3);
          for (unsigned i=0; i<raw.fullsize(); ++i) {
            result[0].push_back(raw[i]);
            result[1].push_back(rebuilt[i]);
            result[2].push_back(_cache[i]);
          }
          return result;
        }
      };
      }
    '''):
        raise RuntimeError("Cannot compile Combine histogram-floor diagnostic")
    return root.combine_jax_floor_diagnostic.View
