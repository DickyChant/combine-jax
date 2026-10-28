"""Stable extended-mixture NLL differences using actual ROOT event densities.

Unlike flat-bin Poisson differences, sampled continuous densities need not sum
to the expected yield. The extended yield term is therefore kept separately.
"""

import numpy as np

from stable_root_bins import LD
from probe_hhvv_stable_fd import log_remainder


def extended_delta(expected, baseline_expected, intensity, baseline, weights):
    expected, baseline_expected = LD(expected), LD(baseline_expected)
    intensity, baseline, weights = (np.asarray(x, dtype=LD) for x in (intensity, baseline, weights))
    assert intensity.shape == baseline.shape == weights.shape
    assert np.isfinite(expected) and np.isfinite(baseline_expected) and min(expected, baseline_expected) > 0
    assert all(np.all(np.isfinite(x)) and np.all(x >= 0) for x in (intensity, baseline, weights))
    counted = weights > 0
    assert np.all((intensity > 0) | ~counted) and np.all((baseline > 0) | ~counted)
    denominator = np.where(counted, baseline, LD(1))
    ratio = np.where(counted, intensity-baseline, LD(0))/denominator
    linear = (expected-baseline_expected)-np.sum(weights*ratio, dtype=LD)
    return linear+np.sum(weights*log_remainder(ratio), dtype=LD)


class RootIntensities:
    """Direct extended RooAddPdf with observable-independent yield coefficients.

    Recursive RooAddPdf flattening and RooMultiPdf corrections are deliberately
    excluded here. No bin-volume quadrature or translated JAX values are used.
    Callers must exclude channels with externally profiled nuisance parameters.
    """

    def __init__(self, root, pdf, observables, positions, weights):
        assert str(pdf.ClassName()) == "RooAddPdf", "Only direct RooAddPdf mixtures are supported"
        self.functions = list(pdf.pdfList())
        self.coefficients = list(pdf.coefList())
        assert len(self.functions) == len(self.coefficients), "An extended yield coefficient per component is required"
        assert all(str(f.ClassName()) != "RooAddPdf" for f in self.functions), "Recursive RooAddPdf flattening is unsupported here"
        assert not any(str(f.ClassName()) == "RooMultiPdf" for f in pdf.getComponents()), "RooMultiPdf corrections are unsupported here"
        self.observables = list(observables)
        self.axes = root.RooArgSet(*self.observables)
        assert all(not c.dependsOn(self.axes) for c in self.coefficients), "Observable-dependent coefficients are unsupported"
        self.positions = np.asarray(positions)
        self.weights = np.asarray(weights, dtype=LD)
        assert self.positions.shape == (len(self.weights), len(self.observables))
        assert np.all(np.isfinite(self.positions)) and np.all(np.isfinite(self.weights)) and np.all(self.weights >= 0)
        self.report = dict(name=str(pdf.GetName()), class_name=str(pdf.ClassName()),
                           component_classes=[str(f.ClassName()) for f in self.functions],
                           dimensions=len(self.observables), rows=len(self.weights),
                           normalization="ROOT component getVal(observables); sampled densities are never renormalized as bin probabilities")

    def predict(self):
        coefficients = np.array([c.getVal() for c in self.coefficients], dtype=LD)
        expected = np.sum(coefficients, dtype=LD)
        intensities = np.zeros(len(self.positions), dtype=LD)
        for function, coefficient in zip(self.functions, coefficients, strict=True):
            for i, row in enumerate(self.positions):
                for observable, coordinate in zip(self.observables, row, strict=True):
                    observable.setVal(float(coordinate))
                intensities[i] += coefficient*LD(function.getVal(self.axes))
        assert np.isfinite(expected) and expected > 0
        assert np.all(np.isfinite(intensities)) and np.all(intensities >= 0)
        assert np.all((intensities > 0) | (self.weights == 0))
        return expected, intensities

    def delta(self, baseline):
        expected, intensities = self.predict()
        return extended_delta(expected, baseline[0], intensities, baseline[1], self.weights)
