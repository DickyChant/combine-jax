"""Stable differences from a live Combine RooRealSumPdf channel cache.

This diagnostic reads the same native components and integrals as CachingAddNLL,
after its analytic bin profiling. Profile constraints remain with the caller's
full NLL. Only ADDNLL_ROOREALSUM_NONORM=1 is supported; no JAX values enter here.
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
    ("integrals_", "vector<RooAbsReal*>"),
    ("binWidths_", "vector<Double_t>"),
)


def reader(root):
    if not hasattr(root, "combine_jax_cached_real_sum"):
        if not root.gInterpreter.Declare(r'''
          #include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"
          #include <stdexcept>
          namespace combine_jax_cached_real_sum {
          template<class T> const T& member(const cacheutils::CachingAddNLL& n, long offset) {
            return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(&n)+offset);
          }
          struct Sample {
            std::vector<double> coefficients, weights, integrals, widths;
            std::vector<std::vector<double>> densities;
            std::vector<std::string> names;
            int basic_integrals, histogram_propagators;
          };
          Sample read(cacheutils::CachingAddNLL& n, const std::vector<long>& o) {
            if (o.size()!=11) throw std::runtime_error("Unexpected reflected layout");
            const int basic = member<int>(n,o[7]);
            if (!member<bool>(n,o[8]) || (basic != 0 && basic != 2) ||
                !member<std::vector<std::pair<const RooMultiPdf*,cacheutils::CachingPdfBase*>>>(n,o[4]).empty() ||
                !member<std::vector<const CMSHistSum*>>(n,o[6]).empty())
              throw std::runtime_error("Unsupported non-real-sum, transient integral, MultiPdf or CMSHistSum channel");
            const auto& coefficients=member<std::vector<RooAbsReal*>>(n,o[0]);
            const auto& pdfs=member<std::vector<std::unique_ptr<cacheutils::CachingPdfBase>>>(n,o[1]);
            const auto& integrals=member<std::vector<RooAbsReal*>>(n,o[9]);
            const auto* data=member<const RooAbsData*>(n,o[2]);
            if (!data || coefficients.size()!=pdfs.size() || pdfs.empty() ||
                (basic==0 && integrals.size()!=pdfs.size()))
              throw std::runtime_error("Invalid live component layout");
            Sample result;
            result.basic_integrals=basic;
            result.histogram_propagators=member<std::vector<const CMSHistErrorPropagator*>>(n,o[5]).size();
            result.weights=member<std::vector<double>>(n,o[3]);
            result.widths=member<std::vector<double>>(n,o[10]);
            if (basic==2 && result.widths.size()!=1 && result.widths.size()!=result.weights.size())
              throw std::runtime_error("Unsupported basic-integral widths");
            for (std::size_t i=0;i<pdfs.size();++i) {
              if (!coefficients[i] || !pdfs[i] || (basic==0 && !integrals[i]))
                throw std::runtime_error("Null component");
              const auto& values=pdfs[i]->eval(*data);
              if (values.size()!=result.weights.size()) throw std::runtime_error("Row count mismatch");
              result.coefficients.push_back(coefficients[i]->getVal());
              result.densities.push_back(values);
              if (basic==0) result.integrals.push_back(integrals[i]->getVal());
              result.names.push_back(pdfs[i]->pdf()->GetName());
            }
            return result;
          }
          }
        '''):
            raise RuntimeError("Cannot compile cached real-sum diagnostic reader")
    return root.combine_jax_cached_real_sum.read


class CachedRootRealSum:
    def __init__(self, root, nll):
        if not isinstance(nll, root.cacheutils.CachingAddNLL):
            raise TypeError("Expected a live CachingAddNLL")
        if str(nll.pdf().ClassName()) != "RooRealSumPdf":
            raise TypeError("Only RooRealSumPdf channels are supported")
        # The const-char* overload caches by pointer and is only for C++ string
        # literals. Use the std::string overload for a Python runtime lookup.
        if int(root.runtimedef.get(root.std.string("ADDNLL_ROOREALSUM_NONORM"))) != 1:
            raise NotImplementedError("Requires ADDNLL_ROOREALSUM_NONORM=1")
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
        self.nll, self.root = nll, root
        self.read = reader(root)
        self.offsets = root.std.vector("long")([m["offset"] for m in layout])
        self.weights = None
        self.report = dict(name=str(nll.pdf().GetName()), class_name="RooRealSumPdf",
            reflected_layout=layout, normalization="Actual Combine cache, coefficients and native or basic integrals; long-double accumulation",
            profile_scope="Channel intensities after native analytic profiling; external profile constraints are not included in this channel delta")

    def predict(self):
        if int(self.root.runtimedef.get(self.root.std.string("ADDNLL_ROOREALSUM_NONORM"))) != 1:
            raise NotImplementedError("Requires ADDNLL_ROOREALSUM_NONORM=1")
        # CachingAddNLL runs its own analytic bin minimization before updating
        # the component caches. Do not reconstruct or rerun that solve here.
        before = float(self.nll.getVal())
        assert np.isfinite(before)
        sample = self.read(self.nll, self.offsets)
        coefficients = np.array(sample.coefficients, dtype=LD)
        weights = np.array(sample.weights, dtype=LD)
        densities = np.array([np.array(row, dtype=LD) for row in sample.densities], dtype=LD)
        assert densities.shape == (len(coefficients), len(weights))
        assert all(np.isfinite(a).all() for a in (coefficients, weights, densities))
        assert np.all(weights >= 0) and np.all(densities >= 0)
        basic = int(sample.basic_integrals)
        if basic == 2:
            widths = np.array(sample.widths, dtype=LD)
            assert widths.size in (1, len(weights)) and np.all(widths > 0)
            integrals = np.sum(densities*widths, axis=1, dtype=LD)
        else:
            assert basic == 0
            integrals = np.array(sample.integrals, dtype=LD)
        assert integrals.shape == coefficients.shape and np.isfinite(integrals).all()
        expected = np.sum(coefficients*integrals, dtype=LD)
        intensities = np.sum(coefficients[:, None]*densities, axis=0, dtype=LD)
        assert expected > 0 and np.isfinite(expected)
        assert np.isfinite(intensities).all() and np.all(intensities >= 0)
        assert np.all((intensities > 0) | (weights == 0))
        if self.weights is None:
            self.weights = weights
            self.report.update(rows=len(weights), components=len(coefficients), basic_integrals=basic,
                histogram_propagators=int(sample.histogram_propagators),
                component_names=[str(n) for n in sample.names])
        else:
            np.testing.assert_array_equal(self.weights, weights)
            assert basic == self.report["basic_integrals"], "Integral policy changed after baseline"
        assert float(self.nll.getVal()) == before, "Reader changed the native channel result"
        return expected, intensities

    def delta(self, baseline):
        expected, intensities = self.predict()
        return extended_delta(expected, baseline[0], intensities, baseline[1], self.weights)
