"""MultiDimFit profiles with reusable conditional executables and device starts."""
import time

from .profile_compiler import ProfileCompiler


class MultiDimProfiler(ProfileCompiler):
    """Global fit and fixed-parameter profiles, without limit-specific bounds.

    Each conditional fit starts from the saved global solution, as native grid
    and impact fits do. Runtime coordinate indices let different nuisances share
    a single executable. Only the number of fixed parameters affects its shape.
    The full minimizer loop and the saved parameter vectors stay on the device.
    """

    def __init__(self, model, *, conditional_parameters, max_iterations=10000,
                 compiler_memory_limit_gib=8., release_compile_cache=True,
                 compile_progress=None, fit_progress=None, defer_compile=False,
                 edm_validation='estimated', global_edm_validation=None):
        import jax
        import numpy as np
        from scipy.special import chdtri
        from .model import make_model_function
        from .parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS
        from .profiling import make_profile_scan_function

        if not jax.config.x64_enabled:
            raise ValueError('MultiDimFit requires float64')
        if not 0 < compiler_memory_limit_gib <= 8:
            raise ValueError('Static memory guard must be positive and at most 8 GiB')
        started = time.perf_counter()
        self.info = model_parameter_info(model)
        self.names = self.info.metadata['names']
        self.discrete = 'discrete_profile' in model.metadata
        if edm_validation not in ('estimated', 'exact_hessian'):
            raise ValueError('Unknown EDM validation policy')
        global_edm_validation = global_edm_validation or edm_validation
        if global_edm_validation not in ('estimated', 'exact_hessian'):
            raise ValueError('Unknown global EDM validation policy')
        if self.discrete and 'exact_hessian' in (edm_validation, global_edm_validation):
            raise NotImplementedError('Inactive discrete-shape curvature cannot certify an exact full-space EDM')
        self.union = (model.arrays['discrete_profile__parameter_union'].copy()
                      if self.discrete else np.zeros(len(self.names), dtype=bool))
        self.release_compile_cache = release_compile_cache
        self.compile_progress, self.fit_progress = compile_progress, fit_progress
        self.compiler_memory, self.executables, self.records = {}, {}, []
        self.timings = dict(prepare_seconds=0., lower_seconds=0., compile_seconds=0.,
                            cache_release_seconds=0., execution_seconds=0.)
        self._compiler_memory_limit_gib = compiler_memory_limit_gib
        self._pending_compile = {}
        self.arrays = jax.device_put({n: a for n, a in model.arrays.items()
                                     if not n.startswith('parameter_info__')})
        self.initial = jax.device_put(self.info.arrays['initial'])
        self.global_point = self.global_value = self.global_indices = None
        function = make_model_function(model, binned_gradient='structural_sequential')
        self._pending_compile['objective'] = (jax.jit(function), self.initial)
        self._coordinates = {}
        groups = [tuple(group) for group in conditional_parameters]
        for group in groups:
            self._check_parameters(group)
        representatives = {len(group): group for group in groups}
        representatives[0] = ()
        if self.discrete:
            from .discrete import make_discrete_channels_function
            from .discrete_fit import make_discrete_fit_function
            small = make_discrete_channels_function(model)
            short = fit_coordinates(self.info, fixed={n: float(v) for n, v, used in
                zip(self.names, self.info.arrays['initial'], self.union, strict=True) if not used})
            self.arrays.update(jax.device_put({'discrete_fit__'+n: short[n] for n in COORDINATE_FIELDS}))
        for count, group in representatives.items():
            coordinates = self._coordinate_template({n: float(self.info.arrays['initial'][self.names.index(n)])
                                                      for n in group})
            up = .5*float(chdtri(1, .32)) if not count else .5
            options = dict(max_iterations=max_iterations, stopping_criterion='edm',
                           edm_tolerance=.002*.1*up,
                           edm_validation=global_edm_validation if not count else edm_validation,
                           column_batch_size=1, curvature_linearization='recompute')
            if self.discrete:
                fit = make_discrete_fit_function(model, _full_function=function,
                    _channel_function=small, **options)
                def scan(c, a, fit=fit):
                    result = fit({n: v[0] for n, v in c.items()},
                        {n: a['discrete_fit__'+n] for n in COORDINATE_FIELDS}, a)
                    return {n: v[None, ...] for n, v in result.items()}
            else:
                scan = make_profile_scan_function(function, **options)
            key = 'global' if count == 0 else 'conditional_'+str(count)
            self._coordinates[key] = jax.device_put(coordinates)
            self._pending_compile[key] = (jax.jit(scan), self._coordinates[key])
        jax.block_until_ready(self.arrays)
        self.timings['prepare_seconds'] = time.perf_counter()-started
        if not defer_compile:
            self.compile()

    def _check_parameters(self, names):
        if not names or len(set(names)) != len(names):
            raise ValueError('Conditional parameters must be distinct and nonempty')
        for name in names:
            if name not in self.names:
                raise ValueError('Unknown profile parameter '+name)
            i = self.names.index(name)
            if self.info.arrays['lower'][i] == self.info.arrays['upper'][i]:
                raise ValueError('Cannot profile imported fixed parameter '+name)
            if self.union[i]:
                raise NotImplementedError('Profiling a discrete component shape parameter requires a restricted short fit: '+name)

    def _coordinate_template(self, fixed):
        from .parameter_info import fit_coordinates, COORDINATE_FIELDS
        coordinates = fit_coordinates(self.info, fixed=fixed)
        return {n: coordinates[n][None, ...] for n in COORDINATE_FIELDS}

    def compile(self):
        for name in tuple(self._pending_compile):
            function, argument = self._pending_compile.pop(name)
            self.executables[name] = self._compile(function, (argument, self.arrays),
                                                  name, self._compiler_memory_limit_gib)

    def _fit(self, key, coordinates, fixed, arrays):
        import jax
        start = time.perf_counter()
        result = jax.block_until_ready(self.executables[key](coordinates, arrays))
        elapsed = time.perf_counter()-start
        self.timings['execution_seconds'] += elapsed
        fields = ('status', 'value', 'iteration', 'function_evaluations',
                  'gradient_evaluations', 'corrected_estimated_edm',
                  'hessian_edm', 'edm_curvature_validated', 'hessian_evaluations')
        record = {n: v.item() for n, v in jax.device_get({n: result[n][0] for n in fields}).items()}
        record.update(kind=key, fixed=dict(fixed), execution_seconds=elapsed)
        if self.discrete:
            record.update({n: jax.device_get(result[n][0]).tolist() for n in
                ('discrete_indices', 'candidate_failures', 'continuous_fits', 'short_fits')})
        self.records.append(record)
        if self.fit_progress:
            self.fit_progress(dict(record))
        if record['status'] != 0:
            raise RuntimeError('Nonconverged MultiDimFit profile: '+repr(record))
        return result, record

    def global_minimum(self):
        self.initial_nll = float(self.executables['objective'](self.initial, self.arrays))
        result, record = self._fit('global', self._coordinates['global'], {}, self.arrays)
        self.global_point = result['values'][0]
        self.global_value = record['value']
        self.global_indices = result['discrete_indices'][0] if self.discrete else None
        return record

    def profile(self, fixed):
        import jax
        import jax.numpy as jnp
        self._check_parameters(tuple(fixed))
        if self.global_point is None:
            raise RuntimeError('Run a converged global fit before profiles')
        key = 'conditional_'+str(len(fixed))
        if key not in self.executables:
            raise ValueError('This fixed-parameter count was not compiled')
        # Host validation uses the small immutable metadata, never copies the
        # current nuisance vector from the accelerator.
        c = jax.device_put(self._coordinate_template(fixed))
        indices = jnp.asarray([self.names.index(n) for n in fixed], dtype=jnp.int32)
        reference = self.global_point.at[indices].set(jnp.asarray(list(fixed.values())))[None, :]
        free = c['free_indices'][0]
        c = {**c, 'reference': reference,
             'lower': (c['physical_lower']-reference[:, free])/c['scale'],
             'upper': (c['physical_upper']-reference[:, free])/c['scale']}
        arrays = ({**self.arrays, 'binned__discrete_indices': self.global_indices}
                  if self.discrete else self.arrays)
        result, record = self._fit(key, c, fixed, arrays)
        return result['values'][0], record
