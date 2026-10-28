"""Isolated synchronized JAX tensor timings; numerical acceptance is a separate audit."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--points', type=Path, help='NPZ with points and parameter_names, in model order')
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--mode', choices=('jit', 'eager'), default='jit')
    parser.add_argument('--operation', choices=('nll', 'value_gradient', 'hessian', 'hvp', 'covariance_rows', 'fit', 'limit'), required=True)
    parser.add_argument('--poi', help='POI for the expected-limit operation')
    parser.add_argument('--r-max', type=float, help='Initial expected-limit search maximum')
    parser.add_argument('--limit-cl', type=float, default=.95)
    parser.add_argument('--limit-strict-bounds', action='store_true')
    parser.add_argument('--directions', type=Path, help='HVP-only NPZ: directions and parameter_names; one direction per call')
    parser.add_argument('--covariance-parameters', nargs='+', help='Covariance-rows-only requested free parameter names')
    parser.add_argument('--covariance-max-iterations', type=int, default=1000)
    parser.add_argument('--curvature-recompute-point',action='store_true',
                        help='HVP/covariance rows: pair the point and direction in an optimization barrier before exact AD')
    parser.add_argument('--repeat', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--binned-gradient', choices=('ad', 'structural', 'structural_sequential'), default='ad')
    parser.add_argument('--row-chunk-size', type=int, default=1024,
                        help='Rows per streamed yield-score reduction in the structural pullback')
    parser.add_argument('--hessian-linearization',choices=('reuse','recompute'),default='reuse',
                        help='Standalone Hessians: save gradient intermediates or recompute them in each column batch')
    parser.add_argument('--fit-max-iterations', type=int, default=1000)
    parser.add_argument('--single-fit',action='store_true',help='Retain one timed fit without warm fit repetitions')
    parser.add_argument('--fit-stopping-criterion', choices=('projected_gradient', 'edm'), default='projected_gradient')
    parser.add_argument('--fit-edm-tolerance', type=float, default=1e-4,
                        help='Raw NLL EDM; Combine tolerance 0.1 with Up=0.5 gives 1e-4')
    parser.add_argument('--fit-edm-validation', choices=('exact_hessian','estimated'),default='exact_hessian')
    parser.add_argument('--fit-covariance', choices=('full', 'none'), default='full',
                        help='Full exact final curvature or gradient-only fit with uncertainties deferred')
    parser.add_argument('--compile-only', action='store_true',
                        help='Record compilation and static memory only; never execute the objective or fit')
    parser.add_argument('--fit-uncertainty-note',default='')
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--compiler-memory-limit-gib', type=float, default=8.,
                        help='Static compiler allocation guard; CPU maximum 8 GiB, GPU maximum 80%% of measured initial free memory')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.operation == 'limit':
        if not args.poi:
            parser.error('--operation limit requires --poi')
        from benchmark_tensor_limit import benchmark_limit
        return benchmark_limit(args)
    if args.repeat < 1 or args.batch_size < 1 or args.fit_max_iterations < 1 or args.row_chunk_size < 1 or args.covariance_max_iterations < 1:
        parser.error('repeat, batch-size, iteration limits and row-chunk-size must be positive')
    if (args.operation=='hvp') != (args.directions is not None):
        parser.error('--directions is required exactly for HVP operations')
    if args.operation=='hvp' and args.batch_size!=1:
        parser.error('HVP captures require --batch-size 1')
    if (args.operation=='covariance_rows') != (args.covariance_parameters is not None):
        parser.error('--covariance-parameters is required exactly for covariance_rows')
    if args.covariance_parameters and len(set(args.covariance_parameters))!=len(args.covariance_parameters):
        parser.error('Select each covariance parameter once')
    if args.curvature_recompute_point and args.operation not in ('hvp','covariance_rows'):
        parser.error('--curvature-recompute-point applies only to HVP/covariance rows')
    if args.mode == 'eager' and args.cache:
        parser.error('Persistent outer-executable caching applies to jit mode')
    if args.compile_only and args.mode != 'jit':
        parser.error('--compile-only requires jit mode')
    if args.fit_covariance != 'full' and args.operation != 'fit':
        parser.error('--fit-covariance none applies only to fit operations')
    if args.operation == 'fit' and args.fit_covariance == 'none' and args.fit_stopping_criterion == 'edm' and args.fit_edm_validation != 'estimated':
        parser.error('Gradient-only EDM fits require --fit-edm-validation estimated; no exact curvature is computed')
    if args.single_fit and args.operation!='fit':
        parser.error('--single-fit applies only to fit operations')
    if args.hessian_linearization!='reuse' and args.operation!='hessian':
        parser.error('--hessian-linearization recompute applies only to standalone Hessians')
    if not 0 < args.compiler_memory_limit_gib < float('inf') or (args.backend == 'cpu' and args.compiler_memory_limit_gib > 8):
        parser.error('Invalid static allocation guard, or CPU limit above 8 GiB')
    numeric_path = args.output.with_suffix('.npz')
    if args.output.exists() or numeric_path.exists():
        raise FileExistsError('Preserve existing benchmark evidence')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    started = time.perf_counter()
    import jax
    import jaxlib
    import numpy as np
    # Device verification precedes loading the model or doing any numerical work.
    devices = jax.devices()
    assert devices and all(d.platform == args.backend for d in devices), devices
    assert jax.config.x64_enabled
    assert 'ROOT' not in sys.modules
    print('verified devices', devices, 'float64', jax.config.x64_enabled, flush=True)
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import make_model_function
    from combine_jax.tensor.curvature import make_value_gradient_hessian, make_value_gradient_hvps
    from combine_jax.tensor.covariance_rows import make_covariance_rows, STATUS as COVARIANCE_STATUS
    from combine_jax.tensor.fitting import make_fit_function
    from combine_jax.tensor.parameter_info import model_parameter_info, fit_coordinates, COORDINATE_FIELDS
    from combine_jax.tensor.compilation import configure_model_cache

    report = dict(provenance(), status='running', stage='load', python=sys.version,
        script_sha256=sha(__file__), jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        backend=args.backend, mode=args.mode, operation=args.operation, float64=True,
        devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
        model=str(args.model), model_sha256={n: sha(args.model/n) for n in ('model.json', 'arrays.npz')},
        points_source=str(args.points) if args.points else 'model initial',
        points_source_sha256=sha(args.points) if args.points else None,
        timings={}, batch_size=args.batch_size, hessian_linearization=args.hessian_linearization,
        binned_gradient=args.binned_gradient, row_chunk_size=args.row_chunk_size,
        compile_only=args.compile_only,curvature_recompute_point=args.curvature_recompute_point,
        validation='Pending independent numerical audit',
        timing_scope='Isolated operation process; synchronized device execution, excluding host output copies. Resident points cycle from index 1 after the first call at index 0 (nominal only if one point). Fit always restarts from the imported nominal point.',
        capture_policy='Single-point operations reuse the last timed result; multi-point operations capture all points in separate untimed calls.',
        execution=('One outer JIT; model and parameter arrays are runtime arguments.' if args.mode == 'jit' else
                   'jax.disable_jit(True): primitive dispatch and Python control flow; first execution includes primitive compilation.'))
    monitor = Resources(args.backend == 'gpu', args.output)
    limit = args.compiler_memory_limit_gib * 1024**3
    report['compiler_allocation_limit_bytes'] = limit
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started, resources=monitor.snapshot())
        if args.backend == 'gpu':
            report['jax_allocator'] = dict(scope='JAX allocator counters, including peak bytes when exposed; excludes driver/context and non-JAX allocations.', devices=[])
            for device in devices:
                try:
                    counters = device.memory_stats()
                    report['jax_allocator']['devices'].append(dict(id=device.id, counters=counters,
                        available=counters is not None))
                except Exception as exc:
                    report['jax_allocator']['devices'].append(dict(id=device.id, available=False, error=repr(exc)))
        save_json(args.output, report)
    def stage(name):
        report['stage'] = monitor.stage = name
        save()
    save()
    try:
        if args.backend == 'gpu':
            assert limit <= .8*(monitor.device_total-monitor.baseline), 'Static allocation limit exceeds 80% of initial free GPU memory'
        before = time.perf_counter()
        model = TensorBundle.load(args.model)
        from tensor_benchmark_dataset import require_blind_model
        require_blind_model(model)
        info = model_parameter_info(model)
        names = model.metadata['parameter_names']
        points = np.array([info.arrays['initial']])
        if args.points:
            with np.load(args.points, allow_pickle=False) as data:
                np.testing.assert_array_equal(data['parameter_names'], names)
                points = data['points']
        assert points.ndim == 2 and points.shape[1] == len(names) and len(points)
        assert points.dtype == np.float64 and np.isfinite(points).all()
        assert np.all(points >= info.arrays['lower']) and np.all(points <= info.arrays['upper'])
        report.update(structure_key=model.structure_key, parameters=len(names), points=len(points))
        report['timings']['load_seconds'] = time.perf_counter()-before
        if args.cache:
            report['cache'] = configure_model_cache(model, args.cache)
        else:
            jax.config.update('jax_enable_compilation_cache', False)
            report['cache'] = dict(enabled=False)
        before = time.perf_counter()
        arrays = jax.device_put({n: a for n, a in model.arrays.items() if not n.startswith('parameter_info__')})
        resident_points = [jax.device_put(p) for p in points]
        jax.block_until_ready((arrays, resident_points))
        function = make_model_function(model, binned_gradient=args.binned_gradient,
                                       row_chunk_size=args.row_chunk_size)
        call_inputs = [(p, arrays) for p in resident_points]
        if args.operation == 'value_gradient':
            function = jax.value_and_grad(function)
        elif args.operation == 'hessian':
            function = make_value_gradient_hessian(function, column_batch_size=args.batch_size,
                                                   linearization=args.hessian_linearization)
        elif args.operation == 'hvp':
            assert len(points)==1, 'HVP captures use one explicit parameter point'
            with np.load(args.directions,allow_pickle=False) as data:
                np.testing.assert_array_equal(data['parameter_names'],names)
                directions=data['directions']
            assert directions.ndim==2 and directions.shape[1]==len(names) and len(directions)
            assert directions.dtype==np.float64 and np.isfinite(directions).all()
            function=make_value_gradient_hvps(function,recompute_point=args.curvature_recompute_point)
            call_inputs=[(resident_points[0],arrays,jax.device_put(d[None,:])) for d in directions]
            report.update(directions_source=str(args.directions),directions_source_sha256=sha(args.directions),
                directions=len(directions),hvp_method='Direct forward-over-reverse AD; one runtime direction per call, no dense Hessian or saved cross-call tape',
                capture_policy='One direction reuses the last timed result; multiple directions are captured separately in untimed calls.',
                timing_scope='Isolated synchronized value+gradient+HVP calls at one parameter point. Timed directions cycle from index 1 after the first call at index 0. Numerical captures are untimed when multiple directions are supplied.')
        elif args.operation == 'covariance_rows':
            assert len(points)==1, 'Covariance rows use one explicit parameter point'
            coordinates=fit_coordinates(info)
            free_names=[names[i] for i in coordinates['free_indices']]
            if any(name not in free_names for name in args.covariance_parameters):
                raise ValueError('Covariance parameters must be known, floating parameters')
            columns=np.array([free_names.index(name) for name in args.covariance_parameters],dtype=np.int32)
            function=make_covariance_rows(function,max_iterations=args.covariance_max_iterations,
                                          recompute_point=args.curvature_recompute_point)
            call_inputs=[(resident_points[0],arrays,jax.device_put(coordinates['free_indices']),
                          jax.device_put(coordinates['scale']),jax.device_put(columns))]
            report['covariance_options']=dict(parameters=args.covariance_parameters,max_iterations=args.covariance_max_iterations,
                relative_tolerance=1e-8,absolute_tolerance=0.,scale_policy=coordinates['scale_policy'],
                status_codes=COVARIANCE_STATUS,scope='Sequential on-device CG with exact recomputed HVPs, no dense Hessian or repair. Success checks true scaled residuals; it does not certify global definiteness or uncertainties at nonsmooth points or away from a minimum.')
        elif args.operation == 'fit':
            coordinates = fit_coordinates(info)
            coordinates = jax.device_put({n: coordinates[n] for n in COORDINATE_FIELDS})
            function = make_fit_function(function, column_batch_size=args.batch_size, gradient_tolerance=1e-7,
                                         covariance=args.fit_covariance=='full',
                                         max_iterations=args.fit_max_iterations,
                                         stopping_criterion=args.fit_stopping_criterion,
                                         edm_tolerance=args.fit_edm_tolerance,edm_validation=args.fit_edm_validation)
            call_inputs = [(coordinates, arrays)]
            report['fit_options'] = dict(gradient_tolerance=1e-7, max_iterations=args.fit_max_iterations,
                stopping_criterion=args.fit_stopping_criterion, edm_tolerance=args.fit_edm_tolerance,
                edm_validation=args.fit_edm_validation,uncertainty_note=args.fit_uncertainty_note,
                edm_estimator='BFGS EDM times (1+3*inverse_uncertainty), followed by exact-Hessian EDM verification/refinement when curvature is valid; no certificate for invalid curvature',
                covariance='raw inverse exact final AD Hessian' if args.fit_covariance=='full' else 'not requested; uncertainties deferred', initial='imported nominal')
            if args.fit_covariance=='none':
                report['fit_options']['edm_estimator']='BFGS EDM times (1+3*inverse_uncertainty); no exact-Hessian check or covariance certificate'
        jax.block_until_ready(call_inputs)
        # The compiled functions retain schedules only; model buffers are
        # explicit resident arguments. Drop redundant NumPy storage before
        # allocating the (much larger for dense Asimov data) AD working set.
        def current_rss_mib():
            return next(float(line.split()[1])/1024 for line in
                        Path('/proc/self/status').read_text().splitlines() if line.startswith('VmRSS:'))
        report['host_model_release']=dict(buffer_bytes=sum(a.nbytes for a in model.arrays.values()),
            rss_before_mib=current_rss_mib(),scope='Release the importer bundle after completed device transfer and function construction; resident runtime arrays and small parameter metadata remain.')
        del model
        gc.collect()
        report['host_model_release']['rss_after_mib']=current_rss_mib()
        report['timings']['prepare_and_transfer_seconds'] = time.perf_counter()-before
        with jax.disable_jit(args.mode == 'eager'):
            if args.mode == 'jit':
                stage('lower')
                before = time.perf_counter()
                lowered = jax.jit(function).lower(*call_inputs[0])
                report['timings']['lower_seconds'] = time.perf_counter()-before
                stage('compile_or_cache_load')
                before = time.perf_counter()
                function = lowered.compile()
                report['timings']['compile_or_cache_load_seconds'] = time.perf_counter()-before
                memory = function.memory_analysis()
                if memory is not None:
                    report['compiler_memory'] = {n: getattr(memory, n) for n in
                        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                    assert sum(report['compiler_memory'][n] for n in
                        ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes')) < limit
                if args.compile_only:
                    assert 'ROOT' not in sys.modules
                    assert all(sha(p)==h for p,h in report['sources'].items())
                    report.update(status='compiled_not_executed',stage='compile_only_complete',root_imported=False,
                        validation='Compiler memory evidence only; no numerical result or fit was produced')
                    print(report['status'],report['compiler_memory'],flush=True)
                    return
            else:
                report['timings'].update(lower_seconds=None, compile_or_cache_load_seconds=None)
            stage('first_execute')
            before = time.perf_counter()
            result = jax.block_until_ready(function(*call_inputs[0]))
            report['timings']['first_execute_seconds'] = time.perf_counter()-before
            first_fit = None
            repeat = args.repeat
            if args.operation == 'fit':
                # Preserve the first completed result before another potentially
                # long fit. No parameter leaves the device inside its fit loop.
                stage('first_fit_checkpoint')
                before = time.perf_counter()
                first_fit = jax.device_get(result)
                first_path = numeric_path.with_name(numeric_path.stem+'_first_fit.npz')
                assert not first_path.exists()
                np.savez_compressed(first_path, parameter_names=np.array(names), points=points,
                    **{'fit_'+k: np.asarray(v) for k,v in first_fit.items()})
                report['first_fit_checkpoint'] = dict(arrays=str(first_path), arrays_sha256=sha(first_path),
                    optimizer_status=int(first_fit['status']), covariance_status=int(first_fit['covariance_status']),
                    iteration=int(first_fit['iteration']), value=float(first_fit['value']),
                    projected_gradient_max=float(np.max(np.abs(first_fit['projected_gradient']),initial=0.)))
                report['timings']['first_fit_checkpoint_seconds'] = time.perf_counter()-before
                report['fit_warm_policy'] = dict(requested_repetitions=0 if args.single_fit else args.repeat,
                    policy='Checkpoint transfer and file I/O are untimed. Warm fits require a converged first fit and valid covariance.')
                if args.single_fit:
                    repeat=0
                    report['fit_warm_policy']['skipped_reason']='Single-fit measurement requested; retain the first execution.'
                elif not (bool(first_fit['converged']) and bool(first_fit['valid_covariance'])):
                    repeat = 0
                    report['fit_warm_policy']['skipped_reason'] = 'First fit or its covariance is invalid; retain the failed attempt without repeating it.'
                else:
                    first_fit = None
                save()
            report['timings'].update(warm_seconds=[], warm_median_seconds=None)
            stage('warm')
            warm = []
            for i in range(repeat):
                before = time.perf_counter()
                result = jax.block_until_ready(function(*call_inputs[(i+1) % len(call_inputs)]))
                warm.append(time.perf_counter()-before)
                report['timings'].update(warm_seconds=warm, warm_median_seconds=float(np.median(warm)))
                save()
            stage('capture')
            capture = dict(parameter_names=np.array(names), points=points)
            # A single-point operation's last timed result already has the exact
            # requested inputs. Reuse it, avoiding an extra full HHVV Hessian/fit.
            outputs = [first_fit if first_fit is not None else jax.device_get(result)] if len(call_inputs) == 1 else []
            if len(call_inputs) != 1:
                for inputs in call_inputs:
                    outputs.append(jax.device_get(function(*inputs)))
            if args.operation == 'fit':
                capture.update({'fit_'+k: np.asarray(v) for k, v in outputs[0].items()})
                report['fit_result'] = {k: outputs[0][k] for k in
                    ('status','covariance_status','iteration','value','projected_gradient','roundoff_accepts','maximum_roundoff_allowance',
                     'estimated_edm','corrected_estimated_edm','inverse_uncertainty','hessian_edm','edm_curvature_validated',
                     'edm_refinements','edm_refinement_enabled','hessian_evaluations','function_evaluations','gradient_evaluations','line_search_evaluations')}
                opt=outputs[0]
                report['evaluation_counts']=dict(nll_optimizer=int(opt['function_evaluations']),
                    gradient_optimizer=int(opt['gradient_evaluations']),hessian_linearizations=int(opt['hessian_evaluations']),
                    nll_total=int(opt['function_evaluations']+opt['hessian_evaluations']),
                    gradient_total=int(opt['gradient_evaluations']+opt['hessian_evaluations']),
                    scope='Each exact Hessian linearizes one value+gradient, then applies its saved linear map in column batches; those directional derivatives are not additional full value+gradient evaluations.')
                # Invalid convergence/covariance is evidence, never an accepted fit.
            elif args.operation == 'covariance_rows':
                result=outputs[0]
                capture.update({'covariance_'+k:np.asarray(v) for k,v in result.items()})
                capture['covariance_parameter_names']=np.asarray(args.covariance_parameters)
                passed=np.asarray(result['status'])==0
                assert np.isfinite(np.asarray(result['rows'])[passed]).all()
                assert np.isnan(np.asarray(result['rows'])[~passed]).all()
                report['covariance_result']={k:result[k] for k in ('status','iterations','hvp_evaluations',
                    'residual_norm','recurrence_residual_norm','residual_target')}
                report['covariance_result']['all_rows_solved']=bool(passed.all())
            elif args.operation == 'nll':
                capture['values'] = np.asarray(outputs)
            else:
                capture['values'] = np.asarray([o[0] for o in outputs])
                capture['gradients'] = np.asarray([o[1] for o in outputs])
                if args.operation == 'hessian':
                    capture['hessians'] = np.asarray([o[2] for o in outputs])
                elif args.operation == 'hvp':
                    capture['directions']=directions
                    capture['hvps']=np.asarray([o[2][0] for o in outputs])
            if args.operation not in ('fit','covariance_rows'):
                assert all(np.isfinite(v).all() for k, v in capture.items() if k != 'parameter_names')
            np.savez_compressed(numeric_path, **capture)
            report.update(arrays=str(numeric_path), arrays_sha256=sha(numeric_path))
        assert 'ROOT' not in sys.modules
        assert all(sha(p) == h for p, h in report['sources'].items())
        report.update(status='measured_pending_validation', stage='complete', root_imported=False)
    except BaseException:
        report.update(status='failed', exception=traceback.format_exc())
        raise
    finally:
        monitor.close()
        save()
    print(report['status'], report['timings'], flush=True)


if __name__ == '__main__':
    main()
