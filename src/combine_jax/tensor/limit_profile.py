"""Reusable conditional profiles for expected limits, with device-resident starts."""
import time


class LimitProfiler:
    """Compile global/conditional fits once; copy only scalar diagnostics per fit.

    The outer crossing controller is framework-neutral Python. Each minimizer
    loop remains inside JIT, and the nuisance warm-start vector stays on device.
    No host likelihood accumulation or final dense Hessian is used.
    ``compile_kinds`` omits executables a stage will never use. Opting into
    ``release_compile_cache`` clears process-wide JAX tracing caches; explicit
    executables and the persistent disk compilation cache remain usable.
    """

    def __init__(self, model, poi, *, r_max=None, confidence_level=.95,
                 compiler_memory_limit_gib=8., max_iterations=10000,
                 global_up=None, global_start_fraction=.01,
                 compile_kinds=('global', 'conditional', 'unprofiled'),
                 release_compile_cache=False, compile_progress=None, fit_progress=None,
                 defer_compile=False):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from .model import make_model_function
        from .parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS
        from .profiling import make_profile_scan_function
        from ..methods.asymptotic import expected_nll_crossing

        if not jax.config.x64_enabled:
            raise ValueError('Expected limits require JAX float64')
        if not 0 < compiler_memory_limit_gib <= 8:
            raise ValueError('The expected-limit static guard must not exceed 8 GiB')
        if (not compile_kinds or len(set(compile_kinds)) != len(compile_kinds)
                or set(compile_kinds) - {'global', 'conditional', 'unprofiled'}):
            raise ValueError('Select distinct global, conditional or unprofiled executables')
        # Explicit executables survive cache release. This opt-in mode bounds
        # host tracing/compiler memory without changing the numerical program.
        self.release_compile_cache = release_compile_cache
        self.compile_progress = compile_progress
        self.fit_progress = fit_progress
        start = time.perf_counter()
        info = model_parameter_info(model)
        names = info.metadata['names']
        if poi not in names:
            raise ValueError(f'Unknown POI {poi}')
        self.index = names.index(poi)
        self.names = names
        i = self.index
        if info.arrays['lower'][i] == info.arrays['upper'][i]:
            raise ValueError('Expected-limit POI must be floating in the imported model')
        if info.arrays['lower'][i] > 0 or info.arrays['upper'][i] <= 0:
            raise ValueError('POI bounds must include zero and a positive search interval')
        self.r_max = float(info.arrays['upper'][i] if r_max is None else r_max)
        if not np.isfinite(self.r_max) or self.r_max <= 0:
            raise ValueError('A finite positive rMax is required')
        point = info.arrays['initial'].copy()
        point[i] = 0.
        conditional = fit_coordinates(info, initial=point, fixed={poi: 0.})
        global_c = fit_coordinates(info, initial=point)
        k = list(global_c['free_indices']).index(i)
        if not 0 <= global_start_fraction <= 1:
            raise ValueError('Global start fraction must lie in [0, 1]')
        global_c['reference'][i] = global_start_fraction * self.r_max
        global_c['scale'][k] = .1 * self.r_max
        global_c['physical_lower'][k] = 0.
        global_c['physical_upper'][k] = self.r_max
        for c in (global_c, conditional):
            free = c['free_indices']
            c['lower'] = (c['physical_lower'] - c['reference'][free]) / c['scale']
            c['upper'] = (c['physical_upper'] - c['reference'][free]) / c['scale']
        self.coordinates = {
            'global': jax.device_put({n: global_c[n][None, ...] for n in COORDINATE_FIELDS}),
            'conditional': jax.device_put({n: conditional[n][None, ...] for n in COORDINATE_FIELDS})}
        self.arrays = jax.device_put({n: a for n, a in model.arrays.items() if not n.startswith('parameter_info__')})
        self.discrete = 'discrete_profile' in model.metadata
        if self.discrete:
            union=model.arrays['discrete_profile__parameter_union']
            short=fit_coordinates(info,initial=point,fixed={n:float(v) for n,v,use in
                zip(names,point,union,strict=True) if not use})
            self.arrays.update(jax.device_put({'discrete_fit__'+n:short[n] for n in COORDINATE_FIELDS}))
        self.warm = jax.device_put(point)
        function = make_model_function(model, binned_gradient='structural_sequential')
        if self.discrete:
            from .discrete import make_discrete_channels_function
            short_function = make_discrete_channels_function(model)
        self.executables = {}
        self._pending_compile = {}
        self._compiler_memory_limit_gib = compiler_memory_limit_gib
        self.compiler_memory = {}
        self.timings = dict(prepare_seconds=time.perf_counter()-start, compile_seconds=0., lower_seconds=0.,
                            cache_release_seconds=0.)
        self.records = []
        self.poi = poi
        # Native expected-limit global minimization uses Up equal to the median
        # NLL crossing, whereas conditional profiles retain Up=0.5.
        up_global = expected_nll_crossing(.5, confidence_level) if global_up is None else float(global_up)
        if not np.isfinite(up_global) or up_global <= 0:
            raise ValueError('Global error definition must be positive and finite')
        for name, up in [('global', up_global), ('conditional', .5)]:
            if name not in compile_kinds:
                continue
            options=dict(max_iterations=max_iterations,stopping_criterion='edm',
                edm_tolerance=.002*.1*up,edm_validation='estimated')
            if self.discrete:
                from .discrete_fit import make_discrete_fit_function
                discrete_fit=make_discrete_fit_function(model,_full_function=function,
                    _channel_function=short_function,**options)
                def scan(coordinates,arrays,discrete_fit=discrete_fit):
                    result=discrete_fit({n:v[0] for n,v in coordinates.items()},
                        {n:arrays['discrete_fit__'+n] for n in COORDINATE_FIELDS},arrays)
                    return {n:v[None,...] for n,v in result.items()}
            else:scan = make_profile_scan_function(function,**options)
            self._pending_compile[name] = (jax.jit(scan), self.coordinates[name])
        if 'unprofiled' in compile_kinds:
            self._pending_compile['unprofiled'] = (jax.jit(function), self.warm)
        self.initial = jax.device_put(info.arrays['initial'])
        self._jax, self._jnp = jax, jnp
        jax.block_until_ready(self.arrays)
        self.timings['prepare_seconds'] = time.perf_counter()-start
        if not defer_compile:
            self.compile()

    def compile(self):
        """Compile prepared objectives after the caller can release host arrays.

        The default constructor still compiles immediately. Deferred preparation
        lets a GPU-only owner discard its neutral NumPy model before compiler
        workers are created, while device buffers and objective metadata remain.
        """
        for name in tuple(self._pending_compile):
            function, coordinates = self._pending_compile.pop(name)
            self.executables[name] = self._compile(function,
                (coordinates, self.arrays), name, self._compiler_memory_limit_gib)

    def _compile(self, function, arguments, name, limit):
        def progress(stage):
            if self.compile_progress is not None:
                self.compile_progress(name, stage, dict(self.timings), dict(self.compiler_memory))

        def release():
            if self.release_compile_cache:
                from ..local import release_compilation_cache
                start = time.perf_counter()
                release_compilation_cache()
                self.timings['cache_release_seconds'] += time.perf_counter()-start

        progress('lowering')
        start = time.perf_counter()
        lowered = function.lower(*arguments)
        self.timings['lower_seconds'] += time.perf_counter()-start
        progress('lowered')
        release()
        progress('compiling')
        start = time.perf_counter()
        compiled = lowered.compile()
        self.timings['compile_seconds'] += time.perf_counter()-start
        memory = compiled.memory_analysis()
        if memory is None:
            raise RuntimeError('Compiler memory analysis unavailable; cannot apply the static guard')
        fields = {key: int(getattr(memory, key)) for key in ('argument_size_in_bytes',
            'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')}
        total = fields['argument_size_in_bytes']+fields['output_size_in_bytes']+fields['temp_size_in_bytes']-fields['alias_size_in_bytes']
        self.compiler_memory[name] = dict(**fields, total_bytes=total)
        if total > limit*2**30:
            raise MemoryError(f'{name} executable needs {total/2**30:.3f} GiB, above {limit:g} GiB guard')
        del lowered, function
        release()
        progress('compiled')
        return compiled

    def _accept(self, result, kind, mu):
        jax = self._jax
        fields = ('status', 'value', 'iteration', 'function_evaluations',
                  'gradient_evaluations', 'corrected_estimated_edm')
        if self.discrete:
            fields+=('discrete_iteration','discrete_delta','candidate_failures','continuous_fits','short_fits')
        summary = {k: v.item() for k, v in jax.device_get({k: result[k][0] for k in fields}).items()}
        if self.discrete:
            summary.update({k:jax.device_get(result[k][0]).tolist() for k in
                ('discrete_indices','candidate_status','candidate_iterations','candidate_edm','candidate_indices')})
        if kind == 'global':
            # Export once after the global loop finishes. A failed or unresolved
            # later crossing can then be diagnosed from this exact starting
            # point, rather than reconstructing it with another fit.
            summary['parameter_names'] = list(self.names)
            summary['parameter_values'] = jax.device_get(result['values'][0]).tolist()
        summary.update(kind=kind, mu=mu)
        self.records.append(summary)
        if self.fit_progress is not None:
            self.fit_progress(dict(summary))
        if summary['status'] != 0:
            raise RuntimeError(f'{kind} fit failed at {self.poi}={mu}: {summary}')
        self.warm = result['values'][0]
        if self.discrete:
            self.arrays['binned__discrete_indices']=result['discrete_indices'][0]
        return float(summary['value'])

    def global_minimum(self, initial=None):
        c = self.coordinates['global']
        if initial is not None:
            reference = self._jnp.asarray(initial)[None, :]
            if reference.shape != c['reference'].shape:
                raise ValueError('Global initial point has the wrong shape')
            free = c['free_indices'][0]
            c = {**c, 'reference': reference,
                'lower': (c['physical_lower']-reference[:, free])/c['scale'],
                'upper': (c['physical_upper']-reference[:, free])/c['scale']}
        result = self._jax.block_until_ready(self.executables['global'](c, self.arrays))
        return self._accept(result, 'global', None)

    def __call__(self, mu):
        c = self.coordinates['conditional']
        reference = self.warm.at[self.index].set(mu)[None, :]
        free = c['free_indices'][0]
        dynamic = {**c, 'reference': reference,
            'lower': (c['physical_lower']-reference[:, free])/c['scale'],
            'upper': (c['physical_upper']-reference[:, free])/c['scale']}
        result = self._jax.block_until_ready(self.executables['conditional'](dynamic, self.arrays))
        return self._accept(result, 'conditional', float(mu))

    def unprofiled(self, mu):
        return float(self.executables['unprofiled'](self.warm.at[self.index].set(mu), self.arrays))

    def point(self):
        """Explicit export boundary, used only for Asimov construction/reporting."""
        return self._jax.device_get(self.warm)
