"""Public measured fits/scans on imported, explicitly selected datasets."""
from pathlib import Path
import gc
import sys
import time

import numpy as np


def workflow_failures(report):
    """Structured stopping/denominator explanations, also usable on saved reports."""
    reasons={1:'iteration cap reached',2:'line search failed',3:'nonfinite objective or derivative',
             4:'invalid starting point or bounds',5:'exact-Hessian EDM validation failed',
             6:'discrete profiling did not converge'}
    options=report.get('fit_options',{});failures=[]
    for fit in report.get('fits',[]):
        status=int(fit.get('status',-1))
        if status==0:continue
        task=fit.get('task','unnamed fit');reason=reasons.get(status,'unknown optimizer status '+str(status))
        message=task+': '+reason
        iteration=fit.get('iteration');limit=options.get('max_iterations')
        if iteration is not None:message+=f' (iterations {iteration}'+(f'/{limit}' if limit is not None else '')+')'
        edm=fit.get('corrected_estimated_edm');target=options.get('edm_tolerance')
        if edm is not None:
            message+=f'; corrected EDM {float(edm):.6g}'
            if target is not None:message+=f', required 0 <= EDM < {float(target):.6g}'
        elif options.get('stopping_criterion')=='edm':message+='; corrected EDM unavailable'
        if status==6:
            message+=f"; discrete iterations {fit.get('discrete_iteration','unknown')}, candidate failures {fit.get('candidate_failures','unknown')}"
        failures.append(dict(task=task,kind='optimizer',status=status,reason=reason,message=message,
            iteration=iteration,max_iterations=limit,corrected_estimated_edm=edm,edm_tolerance=target))
    baseline=report.get('global_baseline_check',{})
    if baseline.get('valid') is False:
        for row in baseline.get('lower_points',[]):
            task=row.get('task') or 'conditional fit '+str(row.get('fit_index'))
            message=f"{task}: invalid global denominator; conditional NLL is {-float(row['deltaNLL']):.6g} below the unrestricted fit (allowance {row['roundoff_allowance']:.6g})"
            failures.append(dict(row,task=task,kind='denominator',reason='conditional below global minimum',message=message))
        for number in baseline.get('nonfinite_points',[]):
            failures.append(dict(task='conditional fit '+str(number),kind='denominator',
                reason='nonfinite conditional NLL',message=f'conditional fit {number}: nonfinite NLL; invalid denominator check'))
        if not baseline.get('lower_points') and not baseline.get('nonfinite_points'):
            reason=baseline.get('reason','invalid global denominator')
            failures.append(dict(task=report.get('selected_global_baseline_task','refit'),kind='denominator',
                reason=reason,message='refit: '+reason))
    if not failures and report.get('status')!='complete_requires_native_validation':
        failures.append(dict(task=report.get('stage','workflow'),kind='workflow',
            reason=report.get('error',report.get('status','unknown failure')),
            message=str(report.get('error',report.get('status','unknown failure')))))
    return failures


def run_model(path, output, *, device='cpu',initial=None,freeze=(),poi=None,points=None,
              cache=None,max_iterations=10000,candidate_batch_size=4,host_memory_limit_gib=11.):
    """Run device-resident EDM fits; save checkpoints, coordinates and provenance.

    Conditional points start from the global fit. If they disprove its minimum,
    retry the unrestricted fit once from the lowest converged conditional point.
    Preserve both attempts and recheck the denominator without shifting the grid.
    A neutral model cannot recover
    a POI folded during import: change POIs/frozen couplings at import time.
    Compilation and execution are timed separately and the 8 GiB guard applies
    before executing each compiled shape. Discrete mode comes from the import:
    independent shapes or explicit native-order shared-shape batching.
    ``host_memory_limit_gib`` separately bounds lifetime process RSS, including
    compiler/Python allocations; it does not change the static allocation guard.
    """
    if type(candidate_batch_size) is not int or candidate_batch_size<1:
        raise ValueError('Candidate batch size must be a positive integer')
    from ..runtime.options import host_memory_limit_gib as validate_host_limit
    host_memory_limit_gib=validate_host_limit(host_memory_limit_gib)
    import jax
    import jaxlib
    from ..datasets import dataset_selection
    from ..runtime.resources import Resources,provenance,save_json,sha,host_memory
    from .bundle import TensorBundle
    from .parameter_info import model_parameter_info,fit_coordinates,COORDINATE_FIELDS
    from .compilation import configure_model_cache
    from .model import make_model_function
    from .fitting import make_fit_function
    if 'ROOT' in sys.modules:raise RuntimeError('Run model fitting in a separate ROOT-free process')
    devices=jax.devices()
    if not jax.config.x64_enabled or not devices or any(d.platform!=device for d in devices):
        raise RuntimeError('Requested device and float64 must be enabled before loading the model')
    path=Path(path);output=Path(output);artifacts=output.with_suffix('.fits')
    if output.exists() or artifacts.exists():raise FileExistsError('Preserve existing fit evidence')
    output.parent.mkdir(parents=True,exist_ok=True);artifacts.mkdir()
    clock=time.perf_counter();started=clock
    host_setup=dict(before_model_load=host_memory())
    model=TensorBundle.load(path);host_setup['after_model_load']=host_memory()
    info=model_parameter_info(model);names=info.metadata['names']
    receipt=model.metadata.get('workflow_import')
    if not receipt:raise ValueError('Model requires explicit public import dataset provenance')
    selected=receipt['dataset']
    dataset_selection(selected['name'],analysis=selected['analysis'],kind=selected['kind'],parameter_names=names)
    point=info.arrays['initial'].copy()
    for name,value in (initial or {}).items():
        if name not in names:raise ValueError('Unknown initial parameter: '+name)
        point[names.index(name)]=value
    if set(freeze)-set(names):raise ValueError('Unknown frozen parameter')
    fixed={n:float(point[names.index(n)]) for n in freeze}
    fit_coordinates(info,initial=point,fixed=fixed)
    if points is not None:
        if poi is None or poi in fixed:raise ValueError('Scan requires an unfrozen POI')
        for value in points:fit_coordinates(info,initial=point,fixed={**fixed,poi:value})
    profiled=bool(model.metadata.get('discrete_profile'))
    discrete_mode=model.metadata['discrete_profile'].get('driver','independent') if profiled else None
    if profiled and discrete_mode not in ('independent','native_batched'):
        raise ValueError('Unknown imported discrete profile driver')
    if profiled and receipt.get('discrete_profile_mode','independent')!=discrete_mode:
        raise ValueError('Import receipt and model discrete mode differ')
    cache_info=configure_model_cache(model,cache) if cache else {'enabled':False}
    if not cache:jax.config.update('jax_enable_compilation_cache',False)
    options=dict(covariance=False,return_matrices=False,inverse_update_block_size=128,
        max_iterations=max_iterations,stopping_criterion='edm',edm_tolerance=1e-4,
        edm_validation='estimated',gradient_tolerance=1e-7)
    if profiled:
        profile_options={k:v for k,v in options.items() if k!='covariance'}
        if discrete_mode=='native_batched':
            from .discrete_batched_fit import make_batched_discrete_fit_function
            function=make_batched_discrete_fit_function(model,candidate_batch_size=candidate_batch_size,
                rescale_small_secants=True,**profile_options)
            options.update(candidate_batch_size=candidate_batch_size,rescale_small_secants=True,
                           binned_gradient='structural_sequential')
        else:
            from .discrete_fit import make_discrete_fit_function
            function=make_discrete_fit_function(model,
                _full_function=make_model_function(model,binned_gradient='structural'),**profile_options)
        union=model.arrays['discrete_profile__parameter_union'].copy()
    else:function=make_fit_function(make_model_function(model,binned_gradient='structural'),**options)
    report=dict(provenance(),status='started',dataset=selected,import_provenance=receipt,
        host_memory_setup=host_setup,
        model_sha256={n:sha(path/n) for n in ('model.json','arrays.npz')},
        versions={'jax':jax.__version__,'jaxlib':jaxlib.__version__},float64=True,
        devices=[dict(platform=d.platform,kind=d.device_kind,id=d.id) for d in devices],
        fit_options=options,discrete='profile' if profiled else 'fixed',
        discrete_profile_mode=discrete_mode,
        discrete_implementation=('native-order maskChannels=2 replay; physical NLL and native rank offset separate'
            if discrete_mode=='native_batched' else 'independent-shape sequential native ordering' if profiled else None),
        cache=cache_info,compiler_static_guard_bytes=8*2**30,
        host_memory_limit_gib=host_memory_limit_gib,compilations=[],fits=[],timings={},
        validation='Method agreement must be established separately against native Combine',
        native_gradient_gate='OPEN at unchanged tolerances' if selected['analysis']=='hig23006' else 'Not evaluated by this command')
    monitor=Resources(device=='gpu',output,rss_limit_mib=host_memory_limit_gib*1024)
    def save(stage):
        report.update(stage=stage,resources=monitor.snapshot(),total_wall_s=time.perf_counter()-started)
        monitor.stage=stage
        if device=='gpu':report['jax_memory_stats']=[d.memory_stats() for d in devices]
        save_json(output,report)
    try:
        save('transfer')
        arrays=jax.device_put({n:a for n,a in model.arrays.items() if not n.startswith('parameter_info__')})
        jax.block_until_ready(arrays);del model;gc.collect()
        report['timings']['setup_s']=time.perf_counter()-clock
        compiled={}
        def fit(label,start,fix):
            coords=fit_coordinates(info,initial=start,fixed=fix)
            c=jax.device_put({n:coords[n] for n in COORDINATE_FIELDS})
            inputs=[c]
            if profiled:
                short=fit_coordinates(info,initial=start,fixed={**{n:float(start[i]) for i,n in enumerate(names) if not union[i]},**fix})
                inputs.append(jax.device_put({n:short[n] for n in COORDINATE_FIELDS}))
            inputs.append(arrays);jax.block_until_ready(inputs)
            key=tuple(len(x['free_indices']) for x in inputs[:-1])
            if key not in compiled:
                save('compile_'+label);now=time.perf_counter()
                lower=jax.jit(function).lower(*inputs);lower_s=time.perf_counter()-now;now=time.perf_counter()
                exe=lower.compile();compile_s=time.perf_counter()-now;memory=exe.memory_analysis()
                if memory is None:raise RuntimeError('Compiler allocation accounting unavailable')
                sizes={n:getattr(memory,n) for n in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
                allocation=sum(sizes[n] for n in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))
                report['compilations'].append(dict(coordinate_sizes=key,lower_s=lower_s,
                    compile_or_cache_load_s=compile_s,memory=sizes,allocation_bytes=allocation))
                save('compiled_'+label)
                if allocation>=8*2**30:raise MemoryError('8 GiB static allocation guard exceeded')
                compiled[key]=exe
            save('execute_'+label);now=time.perf_counter()
            result=jax.block_until_ready(compiled[key](*inputs));elapsed=time.perf_counter()-now
            result=jax.device_get(result)
            artifact=artifacts/(label+'.npz')
            np.savez_compressed(artifact,parameter_names=np.asarray(names),**result)
            row={n:result[n] for n in ('status','value','iteration','function_evaluations','gradient_evaluations','corrected_estimated_edm')}
            row.update(task=label,fixed=fix,execution_s=elapsed,converged=int(result['status'])==0,
                capture=str(artifact),capture_sha256=sha(artifact),deltaNLL=float(result['value'])-float(report['fits'][0]['value']) if report['fits'] else 0.,
                deltaNLL_baseline_task='refit')
            if poi:row[poi]=float(result['values'][names.index(poi)])
            if profiled:
                row.update({n:result[n] for n in ('discrete_indices','discrete_iteration','candidate_failures')})
                row['initial_discrete_indices']=jax.device_get(arrays['binned__discrete_indices'])
                if discrete_mode=='native_batched':
                    row.update({n:result[n] for n in ('discrete_delta','candidate_waves','candidate_status',
                        'masking_offset','native_rank_value','raw_value','discrete_index_history',
                        'discrete_raw_values','discrete_native_values','discrete_masking_offsets')})
            report['fits'].append(row);save('completed_'+label)
            return result
        global_fit=fit('refit',point,fixed)
        baseline_fit=global_fit;baseline_task='refit';best=None;best_task=None
        if int(global_fit['status'])==0:
            if profiled:arrays['binned__discrete_indices']=jax.device_put(global_fit['discrete_indices'])
            for i,value in enumerate(points or ()):
                task='point_%03d'%i
                candidate=fit(task,global_fit['values'],{**fixed,poi:value})
                if int(candidate['status'])==0 and np.isfinite(candidate['value']) and (best is None or candidate['value']<best['value']):
                    best=candidate;best_task=task
        from ..scan_comparison import check_global_baseline
        conditionals=[r for r in report['fits'] if r['task'].startswith('point_')]
        report['initial_global_baseline_check']=check_global_baseline([global_fit,*conditionals])
        if best is not None and not check_global_baseline([global_fit,best])['valid']:
            recovery_task='profile_recovery_refit' if profiled else 'recovery_refit'
            if profiled:arrays['binned__discrete_indices']=jax.device_put(best['discrete_indices'])
            report['global_recovery']=dict(trigger='converged conditional below unrestricted fit',
                seed_task=best_task,seed_nll=float(best['value']),task=recovery_task,
                fixed=dict(fixed),initial_global_nll=float(global_fit['value']),
                status='started',attempt_limit=1,agreement_claimed=False)
            save('start_'+recovery_task)
            recovered=fit(recovery_task,best['values'],fixed)
            if int(recovered['status'])==0 and recovered['value']<global_fit['value']:
                baseline_fit=recovered;baseline_task=recovery_task
            report['global_recovery'].update(status='completed',selected_baseline_task=baseline_task,
                recovered_global_nll=float(recovered['value']),recovered_status=int(recovered['status']),
                recovered_corrected_estimated_edm=float(recovered['corrected_estimated_edm']),
                execution_s=report['fits'][-1]['execution_s'],
                nll_improvement=float(global_fit['value']-baseline_fit['value']))
            if profiled:report['profile_global_recovery']=dict(report['global_recovery'])
        report['all_fits_converged']=len(conditionals)==len(points or ()) and all(r['converged'] for r in report['fits'])
        report['global_baseline_check']=check_global_baseline([baseline_fit,*conditionals])
        report['selected_global_baseline_task']=baseline_task
        report['scan_points']=[dict(task=r['task'],fixed=r['fixed'],status=r['status'],raw_nll=float(r['value']),
            deltaNLL=float(r['value']-baseline_fit['value']),baseline_task=baseline_task) for r in conditionals]
        report['status']='complete_requires_native_validation' if report['all_fits_converged'] else 'fit_failed'
        if not report['global_baseline_check']['valid']:
            report['status']='invalid_global_baseline'
        report['timings'].update(execution_s=sum(r['execution_s'] for r in report['fits']),
            lower_s=sum(r['lower_s'] for r in report['compilations']),
            compile_or_cache_load_s=sum(r['compile_or_cache_load_s'] for r in report['compilations']))
    except BaseException as exc:
        report.update(status='failed',error=repr(exc),all_fits_converged=False);raise
    finally:
        report['failures']=workflow_failures(report)
        monitor.close();save('complete' if report['status'].startswith('complete') else 'failed')
    return report
