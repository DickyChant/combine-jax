"""Diagnostic NLL differences from live Combine caches, without translated nodes.

Only Combine's NONORM extended convention, fast Gaussian/Poisson constraints,
and Poissons normalized over their bare mean variable are supported. Private
members are read through checked ROOT reflection; no
layout offsets are assumed and no native objects or evaluation modes are changed.
"""
import ctypes

import numpy as np

from caching_nll_diagnostics import channel_nlls

LD = np.longdouble


def log_remainder(x):
    """x-log1p(x), with cancellation avoided near zero."""
    x = np.asarray(x, dtype=LD)
    small = abs(x) < LD('.125')
    y = np.where(small, x, LD(0))
    power = y*y
    result = power/2
    for k in range(3, 32):
        power *= -y
        result += power/k
    return np.where(small, result, x-np.log1p(x))


def extended_delta(density, baseline, weights, total, baseline_total):
    """Delta of total - sum(weights*log(unnormalized density))."""
    density, baseline, weights = (np.asarray(x, dtype=LD) for x in (density, baseline, weights))
    if density.shape != baseline.shape or density.shape != weights.shape:
        raise ValueError('Mismatched density/weight shapes')
    if not all(np.isfinite(x).all() for x in (density, baseline, weights)):
        raise ValueError('Nonfinite native density or weight')
    counted = weights != 0
    if np.any(density[counted] <= 0) or np.any(baseline[counted] <= 0):
        raise ValueError('Nonpositive native density at a counted row')
    ratio = (density[counted]-baseline[counted])/baseline[counted]
    return (LD(total)-LD(baseline_total)-np.sum(weights[counted]*ratio, dtype=LD)
            +np.sum(weights[counted]*log_remainder(ratio), dtype=LD))


def poisson_delta(mean, baseline, observed):
    """Combine linear terms per bin before summing, including stationary bins."""
    mean, baseline, observed = (np.asarray(x, dtype=LD) for x in (mean, baseline, observed))
    if mean.shape != baseline.shape or mean.shape != observed.shape:
        raise ValueError('Mismatched Poisson shapes')
    if not all(np.isfinite(x).all() and np.all(x >= 0) for x in (mean, baseline, observed)):
        raise ValueError('Invalid Poisson arguments')
    counted = observed > 0
    if np.any(mean[counted] <= 0) or np.any(baseline[counted] <= 0):
        raise ValueError('Nonpositive Poisson mean')
    delta = mean-baseline
    denominator = np.where(counted, baseline, LD(1))
    ratio = np.where(counted, delta/denominator, LD(0))
    return np.sum(np.where(counted, (baseline-observed)*ratio+observed*log_remainder(ratio), delta), dtype=LD)


class Members:
    """Read-only borrowed members with installed type/offset verification."""
    def __init__(self, root):
        self.root = root
        self.layout = {}

    def address(self, obj, klass, member, expected):
        key = (klass, member)
        if key not in self.layout:
            definition = self.root.TClass.GetClass(klass)
            field = definition.GetDataMember(member) if definition else None
            if not field or field.GetOffset() < 0:
                raise RuntimeError(f'No reflected member {klass}::{member}')
            actual = str(field.GetFullTypeName())
            compact = lambda s: s.replace('std::', '').replace(' ', '').replace('Double_t', 'double')
            if compact(actual) != compact(expected):
                raise TypeError(f'{klass}::{member}: {actual}, expected {expected}')
            self.layout[key] = dict(class_name=klass, member=member, type=actual, offset=int(field.GetOffset()))
        return self.root.addressof(obj)+self.layout[key]['offset']

    def object(self, obj, klass, member, expected):
        address = self.address(obj, klass, member, expected)
        value = self.root.BindObject(address, self.layout[(klass, member)]['type'])
        self.root.SetOwnership(value, False)
        return value

    def scalar(self, obj, klass, member, expected, kind):
        return kind.from_address(self.address(obj, klass, member, expected)).value

    def pointer(self, obj, klass, member, expected):
        address = self.scalar(obj, klass, member, expected+'*', ctypes.c_void_p)
        if not address:
            raise ValueError(f'Null {member}')
        value = self.root.BindObject(address, expected.replace('const ', ''))
        self.root.SetOwnership(value, False)
        return value


class CachedChannel:
    def __init__(self, reader, channel):
        self.reader, self.channel = reader, channel
        klass = 'cacheutils::CachingAddNLL'
        self.member = lambda name, typename: reader.object(channel, klass, name, typename)
        self.scalar = lambda name, typename, kind: reader.scalar(channel, klass, name, typename, kind)
        self.data = reader.pointer(channel, klass, 'data_', 'const RooAbsData')
        self.weights = np.array(self.member('weights_', 'vector<double>'), dtype=LD)
        self.widths = np.array(self.member('binWidths_', 'vector<double>'), dtype=LD)
        self.coefficients = self.member('coeffs_', 'vector<RooAbsReal*>')
        self.pdfs = self.member('pdfs_', 'vector<unique_ptr<cacheutils::CachingPdfBase>>')
        self.integrals = self.member('integrals_', 'vector<RooAbsReal*>')
        self.real_sum = self.scalar('isRooRealSum_', 'bool', ctypes.c_bool)
        self.multipdfs = self.member('multiPdfs_', 'vector<pair<const RooMultiPdf*,cacheutils::CachingPdfBase*>>')
        self.correction = sum(p.first.getCorrection() for p in self.multipdfs)
        self.baseline = None

    def snapshot(self):
        native = float(self.channel.getVal())  # Populate exactly the native caches, including profiling.
        mode = self.scalar('basicIntegrals_', 'int', ctypes.c_int)
        if mode == 1:
            raise ValueError('Native integral checks have not settled')
        if hasattr(self, 'integral_mode') and mode != self.integral_mode:
            raise ValueError('Native integral mode changed during scan')
        self.integral_mode = mode
        density = np.zeros(len(self.weights), dtype=LD)
        total = LD(0)
        for i, (coefficient, pdf) in enumerate(zip(self.coefficients, self.pdfs, strict=True)):
            coefficient = LD(coefficient.getVal())
            values = np.asarray(pdf.eval(self.data), dtype=LD)
            if values.shape != density.shape:
                raise ValueError('Native cached rows differ from weights')
            density += coefficient*values
            if mode == 2:
                if len(self.widths) not in (1, len(values)):
                    raise ValueError('Unsupported native integration widths')
                integral = np.sum(values*self.widths, dtype=LD)
            elif self.real_sum:
                integral = LD(self.integrals[i].getVal())
            else:
                integral = LD(1)
            total += coefficient*integral
        if not np.isfinite(total) or total <= 0:
            raise ValueError('Invalid native extended yield')
        if sum(p.first.getCorrection() for p in self.multipdfs) != self.correction:
            raise ValueError('Discrete correction changed during continuous scan')
        # vectorized::nll_reduce overwrites partialSum_ with weight*log(p/sum).
        cached = np.asarray(self.member('partialSum_', 'vector<double>'), dtype=LD)
        counted = self.weights != 0
        if np.any(density[counted] <= 0):
            raise ValueError('Nonpositive density at a counted row')
        reduced = self.weights[counted]*np.log(density[counted]/total)
        if not np.allclose(cached[counted], reduced, rtol=1e-12, atol=1e-9):
            raise ValueError('Reconstructed component reduction differs from native cache')
        return density, total, native

    def reset(self):
        self.baseline = self.snapshot()

    def delta(self):
        density, total, native = self.snapshot()
        base_density, base_total, base_native = self.baseline
        if self.integral_mode == 2:
            # Native basic integrals sum these exact grid rows and widths.
            # Constant bin volumes cancel out of the logarithm in a delta.
            return poisson_delta(density*self.widths, base_density*self.widths, self.weights), native-base_native
        return extended_delta(density, base_density, self.weights, total, base_total), native-base_native


class CachedDeltas:
    def __init__(self, root, parent):
        if np.finfo(LD).nmant < 63:
            raise RuntimeError('Extended precision long double is required')
        if root.runtimedef.get(root.std.string('ADDNLL_ROOREALSUM_NONORM')) != 1:
            raise NotImplementedError('Only native NONORM extended evaluation is supported')
        self.root, self.parent, self.reader = root, parent, Members(root)
        parent.getVal()
        channels, self.channel_layout = channel_nlls(root, parent)
        self.channels = [CachedChannel(self.reader, c) for c in channels]
        klass = 'cacheutils::CachingSimNLL'
        member = lambda name, typename: self.reader.object(parent, klass, name, typename)
        generic = list(member('constrainPdfs_', 'vector<RooAbsPdf*>'))
        self.nuisances = self.reader.pointer(parent, klass, 'nuis_', 'const RooArgSet')
        self.generic_norms = []
        for c in generic:
            if (str(c.ClassName()) != 'RooPoisson' or not c.getX().isConstant()
                    or str(c.getMean().ClassName()) != 'RooRealVar'
                    or [str(v.GetName()) for v in c.getObservables(self.nuisances)] != [str(c.getMean().GetName())]):
                raise NotImplementedError(f'Unsupported normalized constraint {c.ClassName()} {c.GetName()}')
            self.generic_norms.append((c, float(c.getNorm(self.nuisances))))
        if any(m.getVal() != 0 or not m.isConstant() for m in member('channelMasks_', 'vector<RooAbsReal*>')):
            raise NotImplementedError('Floating or active channel masks are unsupported')
        if any(not x for x in member('internalMasks_', 'vector<bool>')):
            raise NotImplementedError('Internally masked channels are unsupported')
        self.gaussians = member('constrainPdfsFast_', 'vector<SimpleGaussianConstraint*>')
        fast_poissons = list(member('constrainPdfsFastPoisson_', 'vector<SimplePoissonConstraint*>'))
        self.poissons = fast_poissons+generic
        for c in self.gaussians:
            if not c.getSigma().isConstant():
                raise NotImplementedError('Floating Gaussian width')
        for c in fast_poissons:
            if not c.getX().isConstant() or not 1e-10 <= c.getX().getVal() < 1000000:
                raise NotImplementedError('Unsupported fast Poisson observation branch')
        self.scales = np.array([self.reader.scalar(c, 'SimpleGaussianConstraint', 'scale_', 'double', ctypes.c_double)
                               for c in self.gaussians], dtype=LD)
        self.observations = np.array([c.getX().getVal() for c in fast_poissons]+
            [c.getX().getVal() if c.getNoRounding() else np.floor(c.getX().getVal()) for c in generic], dtype=LD)
        if np.any(self.observations < 0):
            raise ValueError('Negative Poisson observation')
        self.max_delta_parity_error = 0.
        self.evaluations = 0

    def constraints(self):
        if any(c.getNorm(self.nuisances) != norm for c, norm in self.generic_norms):
            raise ValueError('Normalized Poisson integral changed')
        gaussian = np.array([LD(c.getX().getVal())-LD(c.getMean().getVal()) for c in self.gaussians], dtype=LD)
        poisson = np.array([c.getMean().getVal() for c in self.poissons], dtype=LD)
        return gaussian, poisson

    def reset(self, variable):
        self.variable = variable
        self.parent.getVal()
        self.selected = [c for c in self.channels if c.channel.pdf().dependsOn(variable)]
        for channel in self.selected:
            channel.reset()
        self.base_constraints = self.constraints()
        self.base_native = float(self.parent.getVal())

    def value(self):
        self.root.RooAbsReal.clearEvalErrorLog()
        native = float(self.parent.getVal())
        if self.root.RooAbsReal.numEvalErrors() or not np.isfinite(native) or native >= 9e9:
            raise ValueError('Native evaluation error')
        stable = LD(0)
        for channel in self.selected:
            delta, _ = channel.delta()
            stable += delta
        gaussian, poisson = self.constraints()
        bg, bp = self.base_constraints
        stable += np.sum(-self.scales*(gaussian-bg)*(gaussian+bg), dtype=LD)
        stable += poisson_delta(poisson, bp, self.observations)
        error = abs(float(stable)-(native-self.base_native))
        self.max_delta_parity_error = max(self.max_delta_parity_error, error)
        self.evaluations += 1
        if error >= 1e-6:
            raise ValueError(f'Stable/native NLL delta mismatch {error}')
        return float(stable)
