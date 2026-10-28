"""Capture one resident device profile scan; numerical acceptance is separate."""
import argparse
import gc
import os
from pathlib import Path
import sys
import time
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--parameter')
    parser.add_argument('--values',type=float,nargs='+')
    parser.add_argument('--plan',type=Path,help='Revalidated shared baseline-bound physical grid')
    parser.add_argument('--backend',choices=('cpu','gpu'),required=True)
    parser.add_argument('--edm-validation',choices=('exact_hessian','estimated'),default='exact_hessian')
    parser.add_argument('--uncertainty-note',default='')
    parser.add_argument('--max-iterations',type=int,default=2000)
    parser.add_argument('--compiler-memory-limit-gib',type=float,default=8.)
    parser.add_argument('--cache',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.plan is None and (args.parameter is None or args.values is None):
        parser.error('Supply --plan or both --parameter and --values')
    if args.plan is not None and (args.parameter is not None or args.values is not None):
        parser.error('A shared plan supplies its own parameter and physical grid')
    if args.max_iterations<1:parser.error('max-iterations must be positive')
    limit=args.compiler_memory_limit_gib
    if not 0<limit<float('inf') or (args.backend=='cpu' and limit>8):
        parser.error('Invalid compiler allocation guard, or CPU limit above 8 GiB')
    if args.edm_validation=='estimated' and not args.uncertainty_note:
        parser.error('Estimated EDM requires an explicit curvature limitation')
    numeric=args.output.with_suffix('.npz')
    if args.output.exists() or numeric.exists():raise FileExistsError('Preserve existing profile evidence')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    os.environ.update(JAX_PLATFORMS='cuda' if args.backend=='gpu' else 'cpu',JAX_ENABLE_X64='true',
        XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jaxlib
    import numpy as np
    devices=jax.devices()
    assert devices and all(d.platform==args.backend for d in devices) and jax.config.x64_enabled
    assert 'ROOT' not in sys.modules
    print('verified devices',devices,'float64',jax.config.x64_enabled,flush=True)
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.compilation import configure_model_cache
    from combine_jax.tensor.parameter_info import model_parameter_info
    from combine_jax.tensor.profiling import prepare_profile_scan
    from tensor_benchmark_dataset import require_blind_model
    started=time.perf_counter()
    report=dict(provenance(),status='running',stage='load',script_sha256=sha(__file__),python=sys.version,
        backend=args.backend,jax_version=jax.__version__,jaxlib_version=jaxlib.__version__,float64=True,
        devices=[dict(platform=d.platform,kind=d.device_kind,id=d.id) for d in devices],
        model=str(args.model),model_sha256={n:sha(args.model/n) for n in ('model.json','arrays.npz')},
        parameter=args.parameter,grid=args.values,timings={},compiler_allocation_limit_bytes=limit*2**30,
        fit_options=dict(initial='imported nominal',stopping_criterion='edm',edm_tolerance=1e-4,
            edm_validation=args.edm_validation,max_iterations=args.max_iterations,uncertainty_note=args.uncertainty_note),
        scope='One outer JIT over a sequential device map. Every profile point starts at the same imported nominal vector with the requested coordinate fixed. No intermediate fit state leaves the device. Compile and synchronized execution are separate; final covariance outputs are omitted. Numerical profile agreement and timing eligibility require separate audits.')
    monitor=Resources(args.backend=='gpu',args.output)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,resources=monitor.snapshot())
        if args.backend=='gpu':
            rows=[]
            for device in devices:
                try:
                    counters=device.memory_stats()
                    rows.append(dict(id=device.id,counters=counters,available=counters is not None))
                except Exception as exc:rows.append(dict(id=device.id,available=False,error=repr(exc)))
            report['jax_allocator']=dict(devices=rows,scope='JAX allocator counters exclude driver/context and non-JAX allocations.')
        save_json(args.output,report)
    def stage(name):
        report['stage']=monitor.stage=name;save()
    try:
        if args.plan is not None:
            stage('validate_profile_plan');before=time.perf_counter()
            from tensor_profile_plan import load_plan
            plan=load_plan(args.plan)
            assert plan['model_sha256']==report['model_sha256'] and plan['host']==report['host']
            assert plan['initial']=='imported nominal'
            args.parameter=plan['parameter'];args.values=plan['grid']
            report.update(parameter=args.parameter,grid=args.values,
                profile_plan=dict(path=str(args.plan),sha256=sha(args.plan)))
            report['timings']['plan_validation_seconds']=time.perf_counter()-before
        stage('load')
        if args.backend=='gpu':assert limit*2**30<=.8*(monitor.device_total-monitor.baseline)
        before=time.perf_counter();model=TensorBundle.load(args.model)
        require_blind_model(model)
        hhvv='r_hhvv' in model.metadata['parameter_names']
        if hhvv and args.edm_validation!='estimated':
            raise ValueError('HHVV profile benchmark uses accepted estimated EDM with uncertified curvature')
        report['dataset']=model.metadata.get('provenance',{}).get('dataset',{}).get('name','data_obs')
        info=model_parameter_info(model);names=np.array(info.metadata['names'])
        report['fit_state']=info.metadata.get('fit_policy')
        report['timings']['load_seconds']=time.perf_counter()-before
        if args.cache:report['compilation_cache']=configure_model_cache(model,args.cache)
        stage('prepare_and_transfer');before=time.perf_counter()
        function,coordinates,arrays=prepare_profile_scan(model,args.parameter,args.values,
            stopping_criterion='edm',edm_tolerance=1e-4,edm_validation=args.edm_validation,
            max_iterations=args.max_iterations)
        del model,info;gc.collect()
        jax.block_until_ready((coordinates,arrays))
        report['timings']['prepare_and_transfer_seconds']=time.perf_counter()-before
        stage('lower');before=time.perf_counter();lowered=function.lower(coordinates,arrays)
        report['timings']['lower_seconds']=time.perf_counter()-before
        stage('compile');before=time.perf_counter();compiled=lowered.compile()
        report['timings']['compile_or_cache_load_seconds']=time.perf_counter()-before
        memory=compiled.memory_analysis()
        if memory is not None:
            report['compiler_memory']={key:getattr(memory,key) for key in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
            assert sum(report['compiler_memory'][key] for key in
                ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))<limit*2**30
        stage('execute');before=time.perf_counter()
        result=compiled(coordinates,arrays);jax.block_until_ready(result)
        report['timings']['execute_seconds']=time.perf_counter()-before
        stage('capture');before=time.perf_counter();result=jax.device_get(result)
        np.savez_compressed(numeric,parameter_names=names,grid=np.asarray(args.values),**result)
        report['timings']['host_capture_seconds']=time.perf_counter()-before
        report.update(arrays=str(numeric),arrays_sha256=sha(numeric),point_statuses=result['status'],
            all_converged=bool(np.all(result['status']==0)),
            evaluation_counts={key:int(np.sum(result[key])) for key in
                ('function_evaluations','gradient_evaluations','line_search_evaluations','hessian_evaluations')},
            status='measured_pending_validation',stage='complete',root_imported='ROOT' in sys.modules)
        counts=report['evaluation_counts']
        counts['nll_total']=counts['function_evaluations']+counts['hessian_evaluations']
        counts['gradient_total']=counts['gradient_evaluations']+counts['hessian_evaluations']
        counts['scope']='Optimizer counters plus one value/gradient linearization per exact-Hessian evaluation.'
        assert not report['root_imported']
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc());raise
    finally:
        monitor.close();save()
    print(report['status'],report['point_statuses'],flush=True)


if __name__=='__main__':main()
