"""Independent fixed-point checks against the installed Combine CachingNLL."""

from importlib.metadata import version
import time

import jax
import numpy as np

from .likelihood import from_workspace
from .root_io import load_caching_nll, REFERENCE_FLAGS


def finite_difference(function, point, steps, *, bounds=None):
    result = []
    for i, step in enumerate(steps):
        if not np.isfinite(step) or step <= 0:
            raise ValueError("Finite-difference steps must be finite and positive")
        if bounds is not None:
            low, high = bounds[0][i], bounds[1][i]
            left, right = point[i]-low, high-point[i]
            if left < 0 or right < 0 or high <= low:
                raise ValueError("Finite-difference point must lie in a nonempty parameter range")
            if min(left, right) < 2*step:
                direction = 1 if right >= left else -1
                step = min(step, max(left, right)/5)
                shift = np.zeros_like(point); shift[i] = direction*step
                if point[i]+shift[i] == point[i]:
                    raise ValueError("Finite-difference step cannot be represented in this range")
                result.append(sum(c*function(point+k*shift) for k,c in enumerate((-25,48,-36,16,-3)))/(12*direction*step))
                continue
        shift = np.zeros_like(point)
        shift[i] = step
        result.append((function(point - 2*shift) - 8*function(point - shift)
                       + 8*function(point + shift) - function(point + 2*shift)) / (12*step))
    return np.asarray(result)


def validate_workspace(workspace, random_points=25, seed=7319, *, analytic_barlow_beeston=True):
    """Validate all channel and global values, gradients and three AD Hessians."""
    root = load_caching_nll()
    model = workspace.obj("ModelConfig")
    data = workspace.data("data_obs")
    likelihood = from_workspace(workspace, analytic_barlow_beeston=analytic_barlow_beeston)
    profiled_names = [name for profile in likelihood.profiles for name in profile.parameters]
    original_profiled = {name: (float(workspace.var(name).getVal()), bool(workspace.var(name).isConstant()))
                         for name in profiled_names}
    reference = root.cacheutils.CachingSimNLL(model.GetPdf(), data, model.GetNuisanceParameters())
    if model.GetPdf().ClassName()=="RooSimultaneousOpt":
        reference.setChannelMasks(model.GetPdf().channelMasks())
    reference.setAnalyticBarlowBeeston(analytic_barlow_beeston)
    reference.clearConstantZeroPoint()
    reference.clearZeroPoint()
    variables = [workspace.var(name) for name in likelihood.parameter_names]
    initial = likelihood.initial.copy()
    channel_references, held_data = [], []
    for channel in likelihood.channels:
        cat = model.GetPdf().indexCat().GetName()
        columns = root.RooArgSet(*(workspace.var(name) for name in channel.observables))
        reduced = data.reduce(root.RooFit.Cut(f"{cat}=={cat}::{channel.name}"), root.RooFit.SelectVars(columns))
        held_data.append(reduced)
        nll = root.cacheutils.CachingAddNLL(f"ref_{channel.name}", "", workspace.pdf(channel.pdf), reduced, True)
        nll.clearConstantZeroPoint()
        nll.clearZeroPoint()
        channel_references.append(nll)

    def set_point(point):
        for variable, value in zip(variables, point, strict=True):
            variable.setVal(float(value))

    def evaluate(point):
        set_point(point)
        return float(reference.getVal())

    rng = np.random.default_rng(seed)
    scales = np.array([p.error if p.error > 0 else 1. for p in likelihood.parameters])
    lower = np.array([p.minimum for p in likelihood.parameters])
    upper = np.array([p.maximum for p in likelihood.parameters])
    margin = np.minimum(1e-2*scales, .01*(upper-lower))
    points = [initial]
    for _ in range(random_points):
        points.append(np.clip(initial + rng.uniform(-2, 2, len(initial))*scales,
                              lower + margin, upper - margin))
    steps = 2e-4 * scales
    start = time.perf_counter()
    likelihood.value_and_grad(initial)[0].block_until_ready()
    compile_seconds = time.perf_counter() - start
    start = time.perf_counter()
    cases = []
    try:
        for i, point in enumerate(points):
            value, gradient = likelihood.value_and_grad(point)
            ref_gradient = finite_difference(evaluate, point, steps)
            ref_value = evaluate(point)
            channel_cases = []
            for channel, ref in zip(likelihood.channels, channel_references, strict=True):
                channel_value = float(likelihood.channel_value(channel, point))
                ref_channel_value = 0. if channel.masked else float(ref.getVal())
                channel_cases.append(dict(name=channel.name, jax=channel_value, combine=ref_channel_value,
                                          absolute_error=abs(channel_value-ref_channel_value)))
            absolute = np.abs(np.asarray(gradient) - ref_gradient)
            relative = absolute / np.maximum(np.abs(ref_gradient), 1e-8)
            case = dict(theta=point.tolist(), jax_nll=float(value), combine_nll=ref_value,
                        nll_absolute_error=abs(float(value)-ref_value), channels=channel_cases,
                        jax_gradient=np.asarray(gradient).tolist(), combine_fd_gradient=ref_gradient.tolist(),
                        gradient_absolute_error=absolute.tolist(), gradient_relative_error=relative.tolist())
            if profiled_names:
                profile_values = likelihood.profile_values(point)
                case["profiled"] = {name: {"jax": float(profile_values[name]), "combine": float(workspace.var(name).getVal())}
                                    for name in profiled_names}
            if i < 3 and len(initial):
                fd_hessian = finite_difference(lambda x: np.asarray(likelihood.value_and_grad(x)[1]), point, steps).T
                hessian = np.asarray(likelihood.hessian(point))
                case.update(hessian=hessian.tolist(), fd_hessian=fd_hessian.tolist(),
                            hessian_max_absolute_error=float(np.max(np.abs(hessian-fd_hessian))),
                            hessian_max_scaled_error=float(np.max(np.abs(hessian-fd_hessian)*np.outer(scales, scales))))
            cases.append(case)
    finally:
        reference.setAnalyticBarlowBeeston(False)
        for name, (value, constant) in original_profiled.items():
            workspace.var(name).setVal(value)
            workspace.var(name).setConstant(constant)
        set_point(initial)
    return dict(combine_version=version("higgs-combine-tool"), root_version=str(root.gROOT.GetVersion()),
                jax_version=jax.__version__, seed=seed, parameters=list(likelihood.parameter_names),
                random_points=random_points, offset="clearConstantZeroPoint and clearZeroPoint",
                constraint_convention="Fast simple constraints; RooGaussian/RooPoisson normalised over nuisance set",
                analytic_barlow_beeston=analytic_barlow_beeston, profiled_parameters=profiled_names,
                runtime_flags=REFERENCE_FLAGS,
                channel_data_convention="Select each channel's observables; retain zero weights in standalone references",
                parameter_scales=scales.tolist(), hessian_scaled_error="abs(H - FD) * outer(parameter_scales, parameter_scales)",
                discrete_choices=likelihood.discrete_choices, channel_masks=likelihood.mask_configuration,
                fd_steps=steps.tolist(), gradient_relative_denominator="max(abs(reference), 1e-8)",
                compile_value_gradient_seconds=compile_seconds,
                comparison_seconds=time.perf_counter()-start, cases=cases,
                max_nll_absolute_error=max(c["nll_absolute_error"] for c in cases),
                max_channel_absolute_error=max(x["absolute_error"] for c in cases for x in c["channels"]),
                max_gradient_relative_error=max(max(c["gradient_relative_error"],default=0.) for c in cases),
                max_hessian_absolute_error=max(c.get("hessian_max_absolute_error", 0.) for c in cases),
                max_hessian_scaled_error=max(c.get("hessian_max_scaled_error", 0.) for c in cases))


def validate_counting(workspace, random_points=25, seed=7319):
    return validate_workspace(workspace, random_points, seed)
