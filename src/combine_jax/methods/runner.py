"""Expected-limit execution on an explicitly prepared background-Asimov artifact."""
import time

from .asymptotic import LimitOptions, expected_limits
from .contracts import validate_expected_model


def run_expected_model(model, poi, *, options=None, r_max=None, max_iterations=10000,
                       compiler_memory_limit_gib=8., discrete_policy='profile',
                       release_compile_cache=False, compile_progress=None, fit_progress=None):
    provenance = validate_expected_model(model, poi, discrete_policy=discrete_policy)
    choices = model.metadata.get('provenance', {}).get('discrete_choices')
    from ..tensor.limit_profile import LimitProfiler
    options = options or LimitOptions()
    profiler = LimitProfiler(model, poi, confidence_level=options.confidence_level,
        r_max=r_max, max_iterations=max_iterations, compiler_memory_limit_gib=compiler_memory_limit_gib,
        release_compile_cache=release_compile_cache, compile_progress=compile_progress,
        fit_progress=fit_progress, defer_compile=True)
    # The runtime passes a temporary model, so these large host arrays can be
    # reclaimed before GPU compilation. Metadata and device inputs stay owned
    # by the profiler; callers that retain their model keep their own ownership.
    del model
    if release_compile_cache:
        from ..local import release_compilation_cache
        release_compilation_cache()
    profiler.compile()
    start = time.perf_counter()
    minimum = profiler.global_minimum()
    result = expected_limits(profiler, minimum, profiler.r_max, options=options)
    from .root_output import audit_crossings
    if any(row['status'] != 'resolved' for row in audit_crossings(result)):
        # Keep the native stepping result unchanged. Two additional conditional
        # fits distinguish a lost nuisance branch from a missing search bracket;
        # neither is silently substituted into a limit or denominator.
        result['unresolved_baseline_diagnostic'] = profiler.diagnose_global_baseline()
    elapsed = time.perf_counter()-start
    result.update(status='computed_pending_native_validation', method='AsymptoticLimits', poi=poi,
        dataset=provenance, timings={**profiler.timings, 'execution_seconds':elapsed},
        compiler_memory=profiler.compiler_memory, profile_fits=profiler.records,
        global_seed_checks=getattr(profiler, 'global_seed_checks', []),
        global_seed_function_evaluations=len(getattr(profiler, 'global_seed_checks', [])),
        evaluation_counts={key: sum(row[key] for row in profiler.records) for key in
            ('function_evaluations', 'gradient_evaluations', 'iteration')},
        gradient_policy='structural_sequential', float64=True,
        discrete_policy=discrete_policy,
        fixed_discrete_choices=provenance.get('fixed_discrete_choices', {}),
        scope='Expected-only CLs profile crossings with device-resident nuisance warm starts. Each fit loop is JIT compiled; scalar crossing control is on the host. Native quantile validation is separate.')
    if choices and discrete_policy=='fixed':
        result.update(status='exploratory_fixed_discrete_pending_native_validation',
            scope='Expected-only limits conditional on the explicitly recorded fixed RooMultiPdf choices. No discrete profiling or agreement with the native discrete envelope is claimed. Fit loops run on device; scalar crossing control is on the host.')
    return result
