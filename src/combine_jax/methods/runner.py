"""Expected-limit execution on an explicitly prepared background-Asimov artifact."""
import time

from .asymptotic import LimitOptions, expected_limits
from .contracts import validate_expected_model


def run_expected_model(model, poi, *, options=None, r_max=None, max_iterations=10000,
                       compiler_memory_limit_gib=8., discrete_policy='profile',
                       release_compile_cache=False, compile_progress=None, fit_progress=None):
    provenance = validate_expected_model(model, poi, discrete_policy=discrete_policy)
    from ..tensor.limit_profile import LimitProfiler
    options = options or LimitOptions()
    profiler = LimitProfiler(model, poi, confidence_level=options.confidence_level,
        r_max=r_max, max_iterations=max_iterations, compiler_memory_limit_gib=compiler_memory_limit_gib,
        release_compile_cache=release_compile_cache, compile_progress=compile_progress,
        fit_progress=fit_progress)
    start = time.perf_counter()
    minimum = profiler.global_minimum()
    result = expected_limits(profiler, minimum, profiler.r_max, options=options)
    elapsed = time.perf_counter()-start
    result.update(status='computed_pending_native_validation', method='AsymptoticLimits', poi=poi,
        dataset=provenance, timings={**profiler.timings, 'execution_seconds':elapsed},
        compiler_memory=profiler.compiler_memory, profile_fits=profiler.records,
        evaluation_counts={key: sum(row[key] for row in profiler.records) for key in
            ('function_evaluations', 'gradient_evaluations', 'iteration')},
        gradient_policy='structural_sequential', float64=True,
        discrete_policy=discrete_policy,
        fixed_discrete_choices=provenance.get('fixed_discrete_choices', {}),
        scope='Expected-only CLs profile crossings with device-resident nuisance warm starts. Each fit loop is JIT compiled; scalar crossing control is on the host. Native quantile validation is separate.')
    if model.metadata.get('provenance', {}).get('discrete_choices') and discrete_policy=='fixed':
        result.update(status='exploratory_fixed_discrete_pending_native_validation',
            scope='Expected-only limits conditional on the explicitly recorded fixed RooMultiPdf choices. No discrete profiling or agreement with the native discrete envelope is claimed. Fit loops run on device; scalar crossing control is on the host.')
    return result
