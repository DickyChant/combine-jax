"""Native histogram score diagnostic using the exact normalization identity.

For a full, unconditionally normalized histogram, sum(volume * dp) = 0.
Subtracting a constant times volume from the likelihood's density adjoint
therefore leaves its derivative unchanged while reducing cancellation near an
Asimov point. Finite differences act on a linearization at the reference point,
not on a different likelihood. This module never imports an AD framework.
"""
import numpy as np

from native_cached_deltas import LD


def centered_adjoint(density, observed, component, volumes):
    density, observed, component, volumes = (
        np.asarray(x, dtype=LD) for x in (density, observed, component, volumes))
    if len({a.shape for a in (density, observed, component, volumes)}) != 1:
        raise ValueError('Mismatched score shapes')
    if not all(np.isfinite(a).all() for a in (density, observed, component, volumes)):
        raise ValueError('Nonfinite score inputs')
    if np.any(density < 0) or np.any(observed < 0) or np.any(component < 0) or np.any(volumes <= 0):
        raise ValueError('Invalid score inputs')
    counted = observed > 0
    if np.any(density[counted] <= 0):
        raise ValueError('Nonpositive counted density')
    norm = np.sum(component*volumes, dtype=LD)
    if abs(norm-1) > LD('1e-12'):
        raise ValueError('Histogram does not have unit full-grid normalization')
    ratio = np.where(counted, observed/np.where(counted, density, LD(1)), LD(0))
    center = np.sum(ratio*component, dtype=LD)/norm
    return ratio-center*volumes


def full_histogram_volumes(reader, pdf, data, rows):
    """Require the complete native histogram grid, in actual cached row order."""
    if str(pdf.ClassName()) != 'FastVerticalInterpHistPdf2D2' or pdf.conditional():
        raise NotImplementedError('Only unconditional FastVerticalInterpHistPdf2D2 is certified')
    hist = pdf.cacheNominal()
    if hist.fullsize() != hist.size() or data.numEntries() != rows:
        raise NotImplementedError('Cropped histogram or omitted data rows')
    edges = [np.array(reader.object(hist, 'FastHisto2D', 'binEdges'+axis+'_', 'FastTemplate::AT'))
             for axis in ('X', 'Y')]
    shape = tuple(len(e)-1 for e in edges)
    if shape != (hist.binX(), hist.binY()) or rows != int(np.prod(shape)):
        raise NotImplementedError('Data does not cover the complete histogram grid')
    axes = [pdf.x(), pdf.y()]
    for axis, edge in zip(axes, edges, strict=True):
        if axis.getMin() != edge[0] or axis.getMax() != edge[-1] or np.any(np.diff(edge) <= 0):
            raise NotImplementedError('Restricted or invalid histogram range')
    cells = []
    for row in range(rows):
        values = data.get(row)
        index = [int(np.searchsorted(edge, values.getRealValue(axis.GetName()), side='left')-1)
                 for axis, edge in zip(axes, edges, strict=True)]
        if any(i < 0 or i >= n for i, n in zip(index, shape, strict=True)):
            raise ValueError('Data row outside the histogram')
        cells.append(index[0]*shape[1]+index[1])
    if sorted(cells) != list(range(rows)):
        raise NotImplementedError('Duplicate or missing histogram cells')
    volumes = np.array([hist.GetWidth(i) for i in cells], dtype=LD)
    geometric = (np.diff(edges[0])[:, None]*np.diff(edges[1])[None, :]).ravel()[cells]
    np.testing.assert_allclose(volumes, geometric, rtol=1e-14, atol=0)
    return volumes


class CenteredHistogramScore:
    """First-order NLL changes read entirely from live Combine component caches.

    Deliberately narrow: floating coefficients, non-histogram affected shapes,
    conditional normalization, incomplete grids and non-Gaussian constraints
    are rejected. The underlying delta reader verifies every full native NLL
    change separately; the score linearization itself differs at O(step**2).
    """
    def __init__(self, stable):
        self.stable = stable
        self.geometries = {}

    def reset(self, variable):
        stable = self.stable
        stable.reset(variable)
        self.parts = []
        self.evidence = []
        for channel in stable.selected:
            if channel.real_sum or channel.integral_mode != 0:
                raise NotImplementedError('Only RooAddPdf channels are supported')
            density = channel.baseline[0]
            for coefficient, cache in zip(channel.coefficients, channel.pdfs, strict=True):
                if coefficient.dependsOn(variable):
                    raise NotImplementedError('Floating mixture coefficient')
                pdf = cache.pdf()
                if not pdf.dependsOn(variable):
                    continue
                key = str(pdf.GetName()), str(channel.channel.GetName())
                if key not in self.geometries:
                    self.geometries[key] = full_histogram_volumes(stable.reader, pdf, channel.data, len(density))
                volumes = self.geometries[key]
                values = np.asarray(cache.eval(channel.data), dtype=LD).copy()
                adjoint = centered_adjoint(density, channel.weights, values, volumes)
                self.parts.append((cache, channel.data, LD(coefficient.getVal()), values, volumes, adjoint))
                self.evidence.append(dict(channel=str(channel.channel.pdf().GetName()),
                    component=str(pdf.GetName()), class_name=str(pdf.ClassName()),
                    full_grid_rows=len(values), integral=float(np.sum(values*volumes, dtype=LD)),
                    max_absolute_centered_adjoint=float(np.max(abs(adjoint)))))
        if not self.parts:
            raise NotImplementedError('No supported histogram dependence')
        if any(c.dependsOn(variable) for c in stable.poissons):
            raise NotImplementedError('Poisson dependence is outside this diagnostic')
        self.base_gaussian = stable.base_constraints[0].copy()

    def value(self):
        stable = self.stable
        stable.value()  # Check the full NLL reconstruction independently.
        result = LD(0)
        for cache, data, coefficient, baseline, volumes, adjoint in self.parts:
            values = np.asarray(cache.eval(data), dtype=LD)
            if abs(np.sum(values*volumes, dtype=LD)-1) > LD('1e-12'):
                raise ValueError('Native histogram normalization changed')
            result -= coefficient*np.sum(adjoint*(values-baseline), dtype=LD)
        gaussian, _ = stable.constraints()
        result -= np.sum(2*stable.scales*self.base_gaussian*(gaussian-self.base_gaussian), dtype=LD)
        return float(result)
