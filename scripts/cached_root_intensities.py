"""Stable differences from a live Combine RooAddPdf channel's cached components.

Diagnostic only: ROOT reflection verifies the installed private member types and
provides offsets. Keep the owning simultaneous NLL alive while using this reader.
No JAX quantities or reconstructed mixture coefficients enter the calculation.
"""

import numpy as np

from stable_root_intensities import LD, extended_delta


MEMBERS = (
    ("coeffs_", "vector<RooAbsReal*>"),
    ("pdfs_", "vector<unique_ptr<cacheutils::CachingPdfBase>>"),
    ("data_", "constRooAbsData*"),
    ("weights_", "vector<Double_t>"),
    ("multiPdfs_", "vector<pair<constRooMultiPdf*,cacheutils::CachingPdfBase*>>"),
    ("histErrorPropagators_", "vector<constCMSHistErrorPropagator*>"),
    ("histSums_", "vector<constCMSHistSum*>"),
    ("basicIntegrals_", "int"),
    ("isRooRealSum_", "bool"),
)


def reader(root):
    if not hasattr(root, "combine_jax_cached_diagnostic"):
        if not root.gInterpreter.Declare(r'''
          #include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"
          #include <stdexcept>
          namespace combine_jax_cached_diagnostic {
          template<class T> const T& member(const cacheutils::CachingAddNLL& n, long offset) {
            return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(&n)+offset);
          }
          struct Sample {
            std::vector<double> coefficients, weights;
            std::vector<std::vector<double>> densities;
            std::vector<std::string> names;
          };
          Sample read(cacheutils::CachingAddNLL& n, const std::vector<long>& o) {
            if (o.size()!=9) throw std::runtime_error("Unexpected reflected layout");
            if (member<bool>(n,o[8]) || member<int>(n,o[7]) != 0 ||
                !member<std::vector<std::pair<const RooMultiPdf*,cacheutils::CachingPdfBase*>>>(n,o[4]).empty() ||
                !member<std::vector<const CMSHistErrorPropagator*>>(n,o[5]).empty() ||
                !member<std::vector<const CMSHistSum*>>(n,o[6]).empty())
              throw std::runtime_error("Real-sum, basic-integral, MultiPdf or bin-profile channel is unsupported");
            const auto& coefficients=member<std::vector<RooAbsReal*>>(n,o[0]);
            const auto& pdfs=member<std::vector<std::unique_ptr<cacheutils::CachingPdfBase>>>(n,o[1]);
            const auto* data=member<const RooAbsData*>(n,o[2]);
            if (!data || coefficients.size()!=pdfs.size() || pdfs.empty())
              throw std::runtime_error("Invalid live component layout");
            Sample result;
            result.weights=member<std::vector<double>>(n,o[3]);
            for (std::size_t i=0;i<pdfs.size();++i) {
              if (!coefficients[i] || !pdfs[i]) throw std::runtime_error("Null component");
              const auto& values=pdfs[i]->eval(*data);
              if (values.size()!=result.weights.size()) throw std::runtime_error("Row count mismatch");
              result.coefficients.push_back(coefficients[i]->getVal());
              result.densities.push_back(values);
              result.names.push_back(pdfs[i]->pdf()->GetName());
            }
            return result;
          }
          }
        '''):
            raise RuntimeError("Cannot compile cached-density diagnostic reader")
    return root.combine_jax_cached_diagnostic.read


class CachedRootIntensities:
    def __init__(self, root, nll):
        if not isinstance(nll, root.cacheutils.CachingAddNLL):
            raise TypeError("Expected a live CachingAddNLL")
        if str(nll.pdf().ClassName()) != "RooAddPdf":
            raise TypeError("Only RooAddPdf channels are supported")
        klass = root.TClass.GetClass("cacheutils::CachingAddNLL")
        layout = []
        for name, expected in MEMBERS:
            item = klass.GetDataMember(name)
            if not item or item.GetOffset() < 0:
                raise RuntimeError(f"Missing reflected member {name}")
            actual = str(item.GetFullTypeName())
            if actual.replace("std::", "").replace(" ", "") != expected:
                raise TypeError(f"Unsupported {name} type: {actual}")
            layout.append(dict(name=name, type=actual, offset=int(item.GetOffset())))
        self.nll = nll
        self.read = reader(root)
        self.offsets = root.std.vector("long")([m["offset"] for m in layout])
        self.weights = None
        self.report = dict(name=str(nll.pdf().GetName()), class_name="RooAddPdf", reflected_layout=layout,
            normalization="Actual Combine cached leaf densities and flattened coefficients; long-double accumulation; no sampled-density renormalization")

    def predict(self):
        before = float(self.nll.getVal())
        assert np.isfinite(before)
        sample = self.read(self.nll, self.offsets)
        coefficients = np.array(sample.coefficients, dtype=LD)
        weights = np.array(sample.weights, dtype=LD)
        densities = np.array([np.array(row, dtype=LD) for row in sample.densities], dtype=LD)
        assert densities.shape == (len(coefficients), len(weights))
        assert all(np.isfinite(a).all() for a in (coefficients, weights, densities))
        assert np.all(weights >= 0) and np.all(densities >= 0)
        if self.weights is None:
            self.weights = weights
            self.report.update(rows=len(weights), components=len(coefficients),
                               component_names=[str(n) for n in sample.names])
        else:
            np.testing.assert_array_equal(self.weights, weights)
        expected = np.sum(coefficients, dtype=LD)
        intensities = np.sum(coefficients[:, None]*densities, axis=0, dtype=LD)
        assert expected > 0 and np.isfinite(expected)
        assert np.isfinite(intensities).all() and np.all(intensities >= 0)
        assert np.all((intensities > 0) | (weights == 0))
        assert float(self.nll.getVal()) == before
        return expected, intensities

    def delta(self, baseline):
        expected, intensities = self.predict()
        return extended_delta(expected, baseline[0], intensities, baseline[1], self.weights)
