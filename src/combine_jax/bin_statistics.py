"""Histogram statistical errors and Combine's analytic Barlow–Beeston solve."""

from dataclasses import dataclass
from functools import lru_cache

import jax.numpy as jnp
import numpy as np

from .graph import UnsupportedModelError
from .histograms import hist_bins, immutable_array, lookup_bins
from .root_io import load_root


@lru_cache(maxsize=1)
def error_access():
    root = load_root()
    if not root.gInterpreter.Declare(r'''
      #include "HiggsAnalysis/CombinedLimit/interface/CMSHistErrorPropagator.h"
      namespace combine_jax_io {
      class ErrorAccess : public CMSHistErrorPropagator {
      public:
        ErrorAccess(CMSHistErrorPropagator const& h) : CMSHistErrorPropagator(h, "combine_jax_errors") {}
        RooAbsReal const& observable() const { return x_.arg(); }
        RooArgList const& parameters() const { return binpars_; }
        std::vector<std::vector<unsigned>> const& types() const { return bintypes_; }
      };
      }
    '''):
        raise RuntimeError("Cannot compile CMSHistErrorPropagator extraction helper")
    return root.combine_jax_io.ErrorAccess


def extract_error_propagator(root):
    view = error_access()(root)
    functions = tuple(str(f.GetName()) for f in root.funcList())
    coefficients = tuple(str(f.GetName()) for f in root.coefList())
    if not functions or len(functions) != len(coefficients):
        raise UnsupportedModelError("Invalid CMSHistErrorPropagator components")
    if any(f.ClassName() != "CMSHistFunc" for f in root.funcList()):
        raise UnsupportedModelError("CMSHistErrorPropagator requires CMSHistFunc components")
    types = tuple(tuple(int(t) for t in row) for row in view.types())
    params = iter(str(p.GetName()) for p in view.parameters())
    bin_parameters = []
    for row in types:
        if not row or any(t not in range(5) for t in row):
            raise UnsupportedModelError(f"Unknown CMSHistErrorPropagator bin type {row}")
        if (row[0] in (0, 1) and len(row) != 1) or (row[0] not in (0, 1) and len(row) != len(functions)):
            raise UnsupportedModelError(f"Invalid CMSHistErrorPropagator bin structure {row}")
        try:
            bin_parameters.append(tuple(next(params) if t in (1, 2, 3) else None for t in row))
        except StopIteration as exc:
            raise UnsupportedModelError("Missing bin parameters") from exc
    if next(params, None) is not None:
        raise UnsupportedModelError("Unused bin parameters")
    observable = str(view.observable().GetName())
    inputs = (observable, *functions, *coefficients,
              *(name for row in bin_parameters for name in row if name is not None))
    return inputs, {"observable_name": observable, "functions": functions, "coefs": coefficients,
                    "bin_types": types, "bin_parameters": tuple(bin_parameters)}


def geometry(node, context):
    first = context.nodes[node.attrs["functions"][0]]
    for name in node.attrs["functions"]:
        other = context.nodes[name]
        if (other.attrs["active"] != first.attrs["active"] or
                not np.array_equal(other.attrs["edges"], first.attrs["edges"]) or
                other.attrs["observable_name"] != node.attrs["observable_name"]):
            raise UnsupportedModelError(f"{node.name}: incompatible component histogram geometry")
    if len(node.attrs["bin_types"]) not in (0, first.attrs["active"]):
        raise UnsupportedModelError(f"{node.name}: bin types do not cover the active bins")
    return first.attrs


def error_moments(node, context):
    key = ("error_moments", node.name)
    if key not in context.cache:
        if any(context.depends_on(name, node.attrs["observable_name"]) for name in node.attrs["coefs"]):
            raise UnsupportedModelError(f"{node.name}: observable-dependent histogram coefficients")
        active = geometry(node, context)["active"]
        functions = node.attrs["functions"]
        values = jnp.stack([hist_bins(context.nodes[f], context)[:active] for f in functions])
        errors = jnp.asarray(np.stack([context.nodes[f].attrs["errors"][:active] for f in functions]))
        coefficients = jnp.stack(context.values(node.attrs["coefs"]))
        total = jnp.sum(coefficients[:, None] * values, axis=0)
        variance = jnp.sum((coefficients[:, None] * errors)**2, axis=0)
        # Avoid an infinite derivative from sqrt(0) in an inactive branch.
        sigma = jnp.where(variance > 0, jnp.sqrt(jnp.where(variance > 0, variance, 1.)), 0.)
        context.cache[key] = (total, sigma, values, errors, coefficients)
    return context.cache[key]


def error_bins(node, context):
    key = ("error_bins", node.name)
    if key in context.cache:
        return context.cache[key]
    total, sigma, values, errors, coefficients = error_moments(node, context)
    result = total
    for index, (types, params) in enumerate(zip(node.attrs["bin_types"], node.attrs["bin_parameters"], strict=True)):
        shift = jnp.asarray(0.)
        if types[0] == 1:
            shift = sigma[index] * context.value(params[0])
        elif types[0] != 0:
            for process, (kind, name) in enumerate(zip(types, params, strict=True)):
                if kind == 2:
                    shift += coefficients[process] * values[process, index] * (context.value(name) - 1.)
                elif kind == 3:
                    shift += coefficients[process] * errors[process, index] * context.value(name)
        result = result.at[index].add(shift)
    attrs = geometry(node, context)
    result = jnp.concatenate((jnp.maximum(result, 1e-9), jnp.asarray(attrs["cache"][attrs["active"]:])))
    context.cache[key] = result
    return result


def error_value(node, context):
    return lookup_bins(error_bins(node, context), geometry(node, context)["edges"],
                       context.value(node.attrs["observable_name"]))


def error_integral(node, context, axes):
    if axes != (node.attrs["observable_name"],):
        raise UnsupportedModelError(f"{node.name}: histogram error integral over {axes}")
    attrs = geometry(node, context)
    active = attrs["active"]
    return jnp.sum(error_bins(node, context)[:active] * jnp.asarray(np.diff(attrs["edges"])[:active]))


def barlow_beeston_solution(mean, error, observed, global_observed):
    """Larger root of the quadratic used by CMSHistErrorPropagator v11."""
    safe_error = jnp.where(error > 0, error, 1.)
    mean = jnp.where(error > 0, mean, 0.)
    observed = jnp.where(error > 0, observed, 0.)
    global_observed = jnp.where(error > 0, global_observed, 0.)
    b = safe_error + mean/safe_error - global_observed
    c = mean - observed - (mean/safe_error)*global_observed
    temporary = -.5 * (b + jnp.copysign(1., b) * jnp.sqrt(b*b-4*c))
    second = c / jnp.where(temporary != 0, temporary, 1.)
    return jnp.maximum(temporary, second)


@dataclass(frozen=True)
class BinProfile:
    propagator: str
    indices: tuple[int, ...]
    parameters: tuple[str, ...]
    global_observed: tuple[float, ...]
    observed: np.ndarray
    widths: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray

    def evaluate(self, context):
        mean, error, *_ = error_moments(context.nodes[self.propagator], context)
        indices = jnp.asarray(self.indices)
        mean, error = mean[indices]*self.widths, error[indices]*self.widths
        solution = barlow_beeston_solution(mean, error, self.observed, jnp.asarray(self.global_observed))
        # ROOT 6.40 throws on setVal outside the range; never silently clip it.
        # At zero error ROOT skips assignment and retains the previous profile
        # value. That history-dependent state has no stateless JAX equivalent.
        valid = (error > 0) & (solution >= self.minimum) & (solution <= self.maximum)
        solution = jnp.where(valid, solution, jnp.nan)
        return dict(zip(self.parameters, solution, strict=True))


def build_profiles(nodes, registry, channels, parameters, constraints):
    from .registry import Evaluation

    context = Evaluation(nodes, registry, (), ())
    floating = {p.name for p in parameters}
    profiles, used = [], set()
    for channel in channels:
        shape = nodes[channel.pdf]
        if shape.class_name != "RooRealSumPdf":
            continue
        for function in shape.attrs["functions"]:
            node = nodes[function]
            if node.class_name != "CMSHistErrorPropagator":
                continue
            attrs = geometry(node, context)
            indices, names, globals_ = [], [], []
            for i, (types, params) in enumerate(zip(node.attrs["bin_types"], node.attrs["bin_parameters"], strict=True)):
                if types[0] != 1 or params[0] not in floating:
                    continue
                name = params[0]
                if name in used:
                    raise UnsupportedModelError(f"{name}: shared analytic bin nuisance")
                if nodes[name].class_name != "RooRealVar":
                    raise UnsupportedModelError(f"{name}: analytic bin nuisance must be a RooRealVar")
                related = [nodes[n] for n in constraints if context.depends_on(n, name)]
                matches = [c for c in related if c.class_name in ("RooGaussian", "SimpleGaussianConstraint")
                           and c.inputs[:2] == (name, name+"_In")]
                if len(matches) != 1 or len(related) != 1:
                    raise UnsupportedModelError(f"{name}: analytic profiling requires one matching Gaussian constraint")
                constraint = matches[0]
                gobs, sigma = nodes[constraint.inputs[1]], nodes[constraint.inputs[2]]
                if not gobs.attrs.get("constant") or not sigma.attrs.get("constant") or sigma.attrs["value"] != 1.:
                    raise UnsupportedModelError(f"{name}: analytic profiling requires fixed global observation and unit sigma")
                for dependency in (*node.attrs["functions"], *node.attrs["coefs"]):
                    if context.depends_on(dependency, name):
                        raise UnsupportedModelError(f"{name}: nuisance also affects nominal histogram prediction")
                indices.append(i); names.append(name); globals_.append(gobs.attrs["value"])
                used.add(name)
            if not indices:
                continue
            coords = channel.data.coordinates[node.attrs["observable_name"]]
            bin_indices = np.searchsorted(attrs["edges"], coords, side="left") - 1
            if np.any((bin_indices < 0) | (bin_indices >= len(attrs["edges"])-1)):
                raise UnsupportedModelError(f"{function}: data outside histogram geometry")
            observed = np.zeros(len(attrs["edges"])-1)
            # CMSHistErrorPropagator::setData assigns, rather than sums, each bin.
            for index, weight in zip(bin_indices, channel.data.weights, strict=True):
                observed[index] = weight
            profiles.append(BinProfile(function, tuple(indices), tuple(names), tuple(globals_),
                                       immutable_array(observed[indices]), immutable_array(np.diff(attrs["edges"])[indices]),
                                       immutable_array([nodes[n].attrs["minimum"] for n in names]),
                                       immutable_array([nodes[n].attrs["maximum"] for n in names])))
    return tuple(profiles)
