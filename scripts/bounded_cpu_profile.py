"""Reuse validated resident NLLs and bounded gradient loads across profile points."""

from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np

from complete_bounded_cpu_covariance import checked_minimum, check_configuration, core_digest, digest


def fixed_problem(likelihood, plan, fixed_value):
    """Change optimizer bounds only; the original likelihood/evaluator stay intact."""
    assert list(likelihood.parameter_names) == plan["parameter_names"]
    selected = plan["parameter_index"]
    assert likelihood.parameter_names[selected] == plan["parameter"]
    parameter = likelihood.parameters[selected]
    if parameter.minimum == parameter.maximum:
        raise ValueError("The profiled nuisance must be free in the original likelihood")
    if not np.isfinite(fixed_value) or not parameter.minimum <= fixed_value <= parameter.maximum:
        raise ValueError("The requested profile value is outside the nuisance bounds")
    initial = np.asarray(plan["initial"], dtype=np.float64).copy()
    if initial.shape != likelihood.initial.shape or not np.isfinite(initial).all():
        raise ValueError("Use the complete finite common profile starting vector")
    initial[selected] = fixed_value
    parameters = tuple(replace(p, minimum=fixed_value, maximum=fixed_value)
                       if index == selected else p for index, p in enumerate(likelihood.parameters))
    problem = SimpleNamespace(parameter_names=likelihood.parameter_names,
                              initial=likelihood.initial, parameters=parameters)
    return problem, initial


def verify_reference_batch(actual, reference):
    size = len(reference["parameter_names"])
    shapes = dict(validation_values=(26,), validation_vg_values=(26,), validation_gradients=(26, size))
    assert set(actual) == set(shapes)
    for key, shape in shapes.items():
        assert actual[key].shape == shape and actual[key].dtype == np.float64
        assert np.isfinite(actual[key]).all()
        np.testing.assert_array_equal(actual[key], reference[key], err_msg=key)


def prepare(likelihood, plan, minimum_path, output, stack, on_progress):
    """Reproduce all 26 accepted reference arrays exactly before profiling."""
    import jax
    import jaxlib
    from resident_value_spooled_gradient import ResidentValueSpooledGradient

    started = time.perf_counter()
    minimum, _ = checked_minimum(minimum_path)
    assert minimum["model"] == plan["model"] and minimum["source"] == plan["source"]
    assert minimum["jax_version"] == jax.__version__ and minimum["jaxlib_version"] == jaxlib.__version__
    check_configuration(likelihood, minimum)
    checkpoint_path = Path(minimum["gradient_checkpoint"])
    assert digest(checkpoint_path) == minimum["gradient_checkpoint_sha256"]
    checkpoint = json.loads(checkpoint_path.read_bytes())
    assert checkpoint["helper_sha256"] == digest("scripts/resident_value_spooled_gradient.py")
    assert checkpoint["runner_sha256"] == digest("scripts/fit_bounded_cpu.py")
    assert checkpoint["reference_manifest_sha256"] == minimum["manifest_sha256"] == plan["manifest_sha256"]
    assert checkpoint["export_sha256"] == plan["export_sha256"]
    assert digest(checkpoint["arrays"]) == checkpoint["arrays_sha256"]
    with np.load(checkpoint["arrays"], allow_pickle=False) as stored:
        reference = {key: stored[key] for key in (
            "parameter_names", "validation_points", "validation_values",
            "validation_vg_values", "validation_gradients")}
    np.testing.assert_array_equal(reference["parameter_names"], likelihood.parameter_names)
    assert reference["validation_points"].shape == (26, len(likelihood.parameter_names))
    directory = Path("build/profile_results")/(output.stem+"_executables")
    artifact = Path("build/profile_results")/(output.stem+"_gradient_validation.npz")
    if artifact.exists():
        raise FileExistsError("Preserve previous profile gradient evidence")
    paths = set(minimum["frozen_source_files"]) | {
        "scripts/bounded_cpu_profile.py", "scripts/complete_bounded_cpu_covariance.py",
        "scripts/run_profile_points.py", "scripts/profile_plan.py"}
    sources = {path: digest(path) for path in sorted(paths)}
    minimum_checksum = digest(minimum_path)
    evaluator = stack.enter_context(ResidentValueSpooledGradient(likelihood, directory,
        gradient_compiler_options=minimum["gradient_compiler_options"], on_progress=on_progress))
    actual = evaluator.prepare(reference["validation_points"])
    # No enlarged tolerance is composed with the frozen Combine acceptance.
    # A changed result requires a fresh reference comparison before fitting.
    verify_reference_batch(actual, reference)
    temporary = artifact.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **actual, parameter_names=reference["parameter_names"],
                        validation_points=reference["validation_points"])
    temporary.replace(artifact)
    assert core_digest() == minimum["source_sha256"]
    assert all(digest(path) == checksum for path, checksum in sources.items())
    assert digest(minimum_path) == minimum_checksum
    evidence = dict(minimum_report=str(minimum_path), minimum_report_sha256=minimum_checksum,
        gradient_reference=str(checkpoint_path), gradient_reference_sha256=digest(checkpoint_path),
        reference_arrays_sha256=checkpoint["arrays_sha256"],
        fresh_arrays=str(artifact), fresh_arrays_sha256=digest(artifact),
        points=26, all_reference_arrays_bitwise_equal=True,
        gradient_compiler_options=minimum["gradient_compiler_options"],
        frozen_sources=sources, value_preparation=evaluator.value_preparation,
        gradient_preparation=evaluator.gradient_preparation,
        preparation_and_validation_seconds=time.perf_counter()-started,
        scope="Fresh complete 26-point NLL/value-gradient/gradient arrays equal the source-bound, accepted bounded minimum reference exactly. A mismatch stops profiling; no tolerance is relaxed.")
    return evaluator, evidence


def fit_point(likelihood, evaluator, plan, fixed_value, args, *, on_evaluation=None):
    from resident_value_spooled_gradient import fit_separate_calls

    problem, initial = fixed_problem(likelihood, plan, fixed_value)
    result = fit_separate_calls(problem, evaluator, initial=initial, max_calls=args.max_calls,
        strategy=args.strategy, tolerance=args.tolerance, on_evaluation=on_evaluation,
        migrad_iterate=args.migrad_iterate, use_simplex=not args.no_simplex)
    assert result["values"][plan["parameter_index"]] == fixed_value
    record = {key: result[key] for key in (
        "valid_minimum", "edm", "function_calls", "gradient_calls", "fixed",
        "objective_offset", "shifted_nll", "canonical_restoration_absolute_error",
        "maximum_resident_streamed_nll_difference", "nonfinite_value_calls",
        "nonfinite_gradient_calls", "timings")}
    return result["values"], result["nll"], record
