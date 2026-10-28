"""Likelihood assembly, with constraints factored once from all channels."""

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .data import bind_channels
from .graph import UnsupportedModelError, names, parameters, walk
from .discrete import active_dependencies, channel_correction
from .registry import Evaluation
from .root_io import load_root
from .translators import default_registry


@dataclass(frozen=True)
class Channel:
    name: str
    pdf: str
    observables: tuple[str, ...]
    data: object
    masked: bool = False


def extended_nll(density, expected, weights):
    """CachingAddNLL's canonical extended-density objective on its valid domain."""
    weights = jnp.asarray(weights)
    total = jnp.sum(weights)
    # Guard the inactive branch as well to keep zero-weight derivatives finite.
    safe_density = jnp.where(weights == 0, 1., density)
    shape = -jnp.sum(weights * jnp.log(safe_density))
    extended = expected - total - total * (jnp.log(expected) - jnp.log(jnp.where(total > 0, total, 1.)))
    return shape + extended


class Likelihood:
    def __init__(self, nodes, registry, parameters_, channels, constraints, offset=0., *,
                 profiles=(), constraint_axes=None, mask_configuration=None):
        self.nodes, self.registry = nodes, registry
        self.parameters = parameters_
        self.parameter_names = tuple(p.name for p in parameters_)
        self.initial = np.array([p.value for p in parameters_], dtype=np.float64)
        self.channels, self.constraints = tuple(channels), tuple(constraints)
        self.profiles = tuple(profiles)
        self.constraint_axes = constraint_axes or {}
        self.mask_configuration = dict(mask_configuration or {})
        active = active_dependencies(nodes, [c.pdf for c in channels if not c.masked]+list(constraints))
        self.discrete_choices = {n:dict(nodes[n].attrs) for n in sorted(active) if nodes[n].class_name=="RooMultiPdf"}
        self.offset = float(offset)
        self.value = jax.jit(self._value)
        self.value_and_grad = jax.jit(jax.value_and_grad(self._value))
        self.hessian = jax.jit(jax.hessian(self._value))

    def profile_values(self, theta, *, parameter_names=None, profiles=None):
        parameter_names = self.parameter_names if parameter_names is None else parameter_names
        profiles = self.profiles if profiles is None else profiles
        context = Evaluation(self.nodes, self.registry, parameter_names, theta)
        result = {}
        for profile in profiles:
            result.update(profile.evaluate(context))
        return result

    def channel_value(self, channel, theta, overrides=None):
        overrides = self.profile_values(theta) if overrides is None else overrides
        return self._channel_value(channel, theta, overrides, self.parameter_names)

    def _channel_value(self, channel, theta, overrides, parameter_names):
        if channel.masked:
            return jnp.asarray(0.)
        context = Evaluation(self.nodes, self.registry, parameter_names, theta, channel.data.coordinates, overrides)
        node = self.nodes[channel.pdf]
        expected = (context.integral(channel.pdf, channel.observables) if node.class_name == "RooRealSumPdf"
                    else sum(context.values(node.attrs["coefs"])))
        density = context.density(channel.pdf, channel.observables)
        return extended_nll(density, expected, channel.data.weights)+channel_correction(self.nodes,channel.pdf)

    def constraint_value(self, theta, overrides=None):
        overrides = self.profile_values(theta) if overrides is None else overrides
        return self._constraint_value(theta, overrides, self.parameter_names, self.constraints)

    def _constraint_value(self, theta, overrides, parameter_names, constraints):
        context = Evaluation(self.nodes, self.registry, parameter_names, theta, overrides=overrides)
        result = jnp.asarray(0.)
        for name in constraints:
            axes = self.constraint_axes.get(name, ())
            result -= context.log_value(name)
            if axes:
                result += jnp.log(context.integral(name, axes))
        return result

    def local_evaluator(self, *, constraint_batch_size=1, hessian_chunk_size=None):
        """Compile local parameter slices and assemble derivatives on the host."""
        from .local import LocalEvaluator
        return LocalEvaluator(self, constraint_batch_size=constraint_batch_size,
                              hessian_chunk_size=hessian_chunk_size)

    def _value(self, theta):
        overrides = self.profile_values(theta)
        return (sum(self.channel_value(c, theta, overrides) for c in self.channels)
                + self.constraint_value(theta, overrides) + self.offset)

    def at_zero(self, theta):
        """Return a separate likelihood offset to zero at an explicit reference point."""
        return Likelihood(self.nodes, self.registry, self.parameters, self.channels,
                          self.constraints, self.offset - float(self.value(theta)),
                          profiles=self.profiles, constraint_axes=self.constraint_axes,
                          mask_configuration=self.mask_configuration)


def from_workspace(workspace, *, model="ModelConfig", data="data_obs", registry=None,
                   analytic_barlow_beeston=True):
    from .bin_statistics import build_profiles
    root = load_root()
    config = workspace.obj(model)
    dataset = workspace.data(data) if isinstance(data, str) else data
    if not config or not dataset:
        raise ValueError(f"Missing {model} or {data}")
    pdf = config.GetPdf()
    if pdf.ClassName() not in ("RooSimultaneous", "RooSimultaneousOpt"):
        raise UnsupportedModelError(f"{pdf.GetName()} ({pdf.ClassName()}): simultaneous model required")
    masks = tuple(pdf.channelMasks()) if pdf.ClassName()=="RooSimultaneousOpt" else ()
    observables = config.GetObservables()
    continuous_obs = {v.GetName() for v in observables if v.InheritsFrom("RooAbsReal")}
    parameter_info = parameters(pdf, names(observables), names(config.GetGlobalObservables()),
                                names(config.GetParametersOfInterest()))
    registry = registry or default_registry()
    constraints, channels, graph_roots = {}, [], []
    mask_configuration, mask_controls = {}, set()
    if masks and len(masks)!=pdf.indexCat().numTypes():
        raise UnsupportedModelError("Channel mask count does not match the category")

    def factor(node, shapes):
        if node.getAttribute('ignoreConstraint'):
            if node.dependsOn(observables):
                raise UnsupportedModelError(f'{node.GetName()}: ignoreConstraint on a data-dependent shape')
            # Combine's redefineSignalPOIs marks the corresponding constraint
            # PDF this way before constructing CachingSimNLL.
            return
        if node.ClassName() == "RooProdPdf":
            for child in node.pdfList():
                norm = node.findPdfNSet(child)
                if norm and len(norm):
                    raise UnsupportedModelError(f"{node.GetName()}: conditional product requires explicit lowering")
                factor(child, shapes)
        elif not node.dependsOn(observables):
            if node.ClassName() not in ("SimpleGaussianConstraint", "SimplePoissonConstraint", "RooGaussian", "RooPoisson"):
                raise UnsupportedModelError(f"{node.GetName()} ({node.ClassName()}): constraint not yet supported")
            constraints[str(node.GetName())] = node
        else:
            shapes.append(node)

    category = pdf.indexCat().Clone()
    for index in range(category.numTypes()):
        category.setBin(index)
        label = str(category.getLabel())
        channel_pdf = pdf.getPdf(label)
        if not channel_pdf:
            continue
        masked = False
        if masks:
            mask = masks[index]
            value = float(mask.getVal())
            if not np.isfinite(value):
                raise UnsupportedModelError(f"{mask.GetName()}: nonfinite channel mask")
            masked = value!=0.
            mask_configuration[label] = dict(name=str(mask.GetName()), class_name=str(mask.ClassName()), value=value, masked=masked)
            mask_controls.update(n for n,v in walk(mask).items() if v.ClassName()=="RooRealVar" and not v.isConstant())
        shapes = []
        factor(channel_pdf, shapes)
        if len(shapes) != 1 or shapes[0].ClassName() not in ("RooAddPdf", "RooRealSumPdf"):
            raise UnsupportedModelError(f"Channel {label}: expected one extended sum, got "
                                        + str([(s.GetName(), s.ClassName()) for s in shapes]))
        shape = shapes[0]
        if not shape.canBeExtended():
            raise UnsupportedModelError(f"Channel {label}: nonextended sum")
        obs = tuple(sorted(names(shape.getObservables(observables))))
        channels.append(Channel(label, str(shape.GetName()), obs, None, masked))
        graph_roots.append(shape)
    bindings=bind_channels(dataset,str(pdf.indexCat().GetName()),{c.name:c.observables for c in channels})
    channels=[replace(c,data=bindings[c.name]) for c in channels]
    if pdf.ClassName() == "RooSimultaneousOpt":
        for constraint in pdf.extraConstraints():
            extra_shapes = []
            factor(constraint, extra_shapes)
            if extra_shapes:
                raise UnsupportedModelError(f"Extra constraint {constraint.GetName()} depends on data")
    graph_roots.extend(constraints.values())
    nodes = registry.export(graph_roots, continuous_obs)
    all_dependencies = active_dependencies(nodes, [c.pdf for c in channels]+list(constraints))
    overlap = mask_controls & all_dependencies
    if overlap:
        raise UnsupportedModelError(f"Mask controls also affect likelihood terms: {sorted(overlap)}")
    active = active_dependencies(nodes, [c.pdf for c in channels if not c.masked]+list(constraints))
    parameter_info = tuple(p for p in parameter_info if p.name in active)
    constraint_axes = {
        name: tuple(sorted(names(c.getObservables(config.GetNuisanceParameters()))))
        if c.ClassName() in ("RooGaussian", "RooPoisson") and config.GetNuisanceParameters() else ()
        for name, c in constraints.items()
    }
    profiles = build_profiles(nodes, registry, [c for c in channels if not c.masked], parameter_info, constraints) if analytic_barlow_beeston else ()
    profiled = {n for p in profiles for n in p.parameters}
    parameter_info = tuple(p for p in parameter_info if p.name not in profiled)
    result = Likelihood(nodes, registry, parameter_info, channels, sorted(constraints),
                        profiles=profiles, constraint_axes=constraint_axes,mask_configuration=mask_configuration)
    # Eager construction check reports unsupported integrals before JIT/fitting.
    result._value(jnp.asarray(result.initial))
    return result
