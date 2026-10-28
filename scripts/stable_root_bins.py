"""Independent ROOT bin expectations for numerically stable NLL differences.

Only explicitly checked flat-bin extended mixtures are supported. This module
does not evaluate translated JAX nodes or approximate continuous densities.
"""

import itertools

import numpy as np

from combine_jax.discrete import active_dependencies
from probe_hhvv_stable_fd import log_remainder


LD = np.longdouble


def poisson_delta(mean, baseline, observed):
    mean, baseline, observed = (np.asarray(x, dtype=LD) for x in (mean, baseline, observed))
    assert all(np.all(np.isfinite(x)) and np.all(x >= 0) for x in (mean,baseline,observed))
    counted = observed > 0
    assert np.all((mean > 0) | ~counted) and np.all((baseline > 0) | ~counted)
    delta = mean-baseline
    denominator = np.where(counted,baseline,LD(1))
    ratio = np.where(counted,delta,LD(0))/denominator
    stable = ((baseline-observed)/denominator)*delta+observed*log_remainder(ratio)
    return np.sum(np.where(counted,stable,delta),dtype=LD)


def flat_pdf(nodes, name):
    node = nodes[name]
    if node.class_name in ("FastVerticalInterpHistPdf2", "FastVerticalInterpHistPdf2D2",
                           "RooParametricStepFunction", "RooParametricHist2D", "RooUniform"):
        return True
    if node.class_name == "VerticalInterpPdf":
        return all(flat_pdf(nodes, n) for n in node.attrs["functions"])
    return False


class RootBins:
    def __init__(self, root, workspace, likelihood, channel):
        self.node = node = likelihood.nodes[channel.pdf]
        assert node.class_name in ("RooAddPdf", "RooRealSumPdf") and node.attrs["extended"]
        assert not channel.masked and 1 <= len(channel.observables) <= 2
        dependencies = active_dependencies(likelihood.nodes, [channel.pdf])
        assert not any(p.propagator in dependencies for p in likelihood.profiles), "Profiled channel unsupported here"
        self.observables = [workspace.var(n) for n in channel.observables]
        self.axes = root.RooArgSet(*self.observables)
        self.positions = np.column_stack([channel.data.coordinates[n] for n in channel.observables])
        self.weights = np.asarray(channel.data.weights, dtype=LD)
        self.binnings = [o.getBinning() for o in self.observables]
        self.bins = [tuple(b.binNumber(float(x)) for b,x in zip(self.binnings,row,strict=True)) for row in self.positions]
        assert len(set(self.bins)) == len(self.bins)
        assert len(self.bins) == np.prod([b.numBins() for b in self.binnings]), "Dataset must cover every bin"
        self.volumes = np.array([np.prod([LD(b.binWidth(i)) for b,i in zip(self.binnings,row,strict=True)])
                                 for row in self.bins], dtype=LD)
        names = node.attrs["pdfs" if node.class_name == "RooAddPdf" else "functions"]
        if node.class_name == "RooAddPdf":
            assert all(flat_pdf(likelihood.nodes, n) for n in names), "Unsupported non-flat component"
            self.functions = [workspace.pdf(n) for n in names]
        else:
            assert all(likelihood.nodes[n].class_name == "CMSHistErrorPropagator" for n in names)
            self.functions = [workspace.function(n) for n in names]
        self.coefficients = [workspace.function(n) or workspace.var(n) for n in node.attrs["coefs"]]
        assert len(self.functions) == len(self.coefficients)
        assert all(not (set(channel.observables) & active_dependencies(likelihood.nodes,[n])) for n in node.attrs["coefs"])
        self.report = dict(name=channel.name, class_name=node.class_name,
                           component_classes=[str(f.ClassName()) for f in self.functions],
                           dimensions=len(self.observables), bins=len(self.bins),
                           max_flatness_error=0., max_normalization_error=0.)

    def set_position(self, row):
        for observable, x in zip(self.observables,row,strict=True):
            observable.setVal(float(x))

    def component(self, function):
        return LD(function.getVal(self.axes) if self.node.class_name == "RooAddPdf" else function.getVal())

    def check_flatness(self):
        for function in self.functions:
            for bins in self.bins:
                values = []
                for fractions in itertools.product((.25,.75), repeat=len(bins)):
                    self.set_position([b.binLow(i)+q*b.binWidth(i)
                                       for b,i,q in zip(self.binnings,bins,fractions,strict=True)])
                    values.append(self.component(function))
                error = float(max(values)-min(values))
                self.report["max_flatness_error"] = max(self.report["max_flatness_error"],error)
                assert error < 1e-12*max(1.,float(max(abs(x) for x in values)))

    def predict(self):
        means = np.zeros(len(self.positions), dtype=LD)
        for function, coefficient in zip(self.functions,self.coefficients,strict=True):
            values = []
            for row in self.positions:
                self.set_position(row)
                values.append(self.component(function))
            values = np.asarray(values)*self.volumes
            if self.node.class_name == "RooAddPdf":
                norm = values.sum(dtype=LD)
                error = float(abs(norm-1))
                self.report["max_normalization_error"] = max(self.report["max_normalization_error"],error)
                assert error < 1e-10 and np.all(values >= 0)
                values /= norm
            means += LD(coefficient.getVal())*values
        assert np.all(np.isfinite(means)) and np.all(means >= 0), (self.report["name"],means.tolist())
        zero = np.flatnonzero(means == 0)
        assert np.all(self.weights[zero] == 0), "Zero expectation with positive observed weight"
        previous = {entry["index"] for entry in self.report.get("zero_mean_bins",[])}
        self.report["zero_mean_bins"] = [dict(index=int(i),coordinates=self.positions[i].tolist(),
                                              observed_weight=float(self.weights[i]))
                                         for i in sorted(previous | set(zero))]
        return means


def constraint_delta(likelihood, workspace, name, baseline, coordinate):
    constraints = [likelihood.nodes[n] for n in likelihood.constraints
                   if name in active_dependencies(likelihood.nodes,[n])]
    delta = LD(0)
    for node in constraints:
        if node.class_name == "SimpleGaussianConstraint":
            assert node.inputs[0] == name
            mean, sigma = (workspace.function(n) or workspace.var(n) for n in node.inputs[1:])
            assert mean.isConstant() and sigma.isConstant() and sigma.getVal() > 0
            mean, sigma = LD(mean.getVal()), LD(sigma.getVal())
            shift = LD(coordinate)-LD(baseline)
            delta += shift*(LD(baseline)-mean+shift/2)/(sigma*sigma)
        elif node.class_name == "RooPoisson":
            assert node.inputs[1] == name
            observed = workspace.var(node.inputs[0])
            assert observed.isConstant()
            assert likelihood.constraint_axes[node.name] == (name,), "Only mean-variable normalization is supported"
            delta += poisson_delta(coordinate,baseline,LD(observed.getVal()))
        else:
            raise NotImplementedError(f"Unsupported stable constraint: {node.class_name} {node.name}")
    return delta
