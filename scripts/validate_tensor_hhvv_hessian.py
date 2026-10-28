"""One complete device-loop HHVV Hessian against the saved full CPU AD oracle."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import threading
import time
import traceback

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function
from combine_jax.tensor.curvature import make_value_gradient_hessian
from compare_local_cpu_hessian import load_arrays,matrix_errors,reassemble_local


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-size',type=int,default=4)
    parser.add_argument('--output',default='reports/tensor_hhvv_hessian.json')
    parser.add_argument('--directory',default='build/tensor_hhvv_hessian')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('--batch-size must be positive')
    path=Path(args.output);directory=Path(args.directory)
    assert not path.exists() and not directory.exists();directory.mkdir();started=time.perf_counter()
    gate_path=Path('reports/tensor_hhvv_unique_gradients.json');gate=json.loads(gate_path.read_bytes());assert gate['status']=='passed'
    capture_path=Path(gate['capture']);assert sha(capture_path)==gate['capture_sha256'];capture=json.loads(capture_path.read_bytes())
    assert sha(capture['arrays'])==capture['arrays_sha256']
    imp_path=Path(capture['import_report']);assert sha(imp_path)==capture['import_report_sha256'];imp=json.loads(imp_path.read_bytes())
    for p,digest in {**capture['tensor_sources'],**imp['artifacts']}.items():assert sha(p)==digest
    sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))}
    assert set(sources)-set(capture['tensor_sources'])=={'src/combine_jax/tensor/curvature.py'}
    prior_audit=Path('reports/gpu_hhvv_recovered_hessian_reference_checked_full_audit.json');accepted=json.loads(prior_audit.read_bytes())
    assert accepted['status']=='passed';reference_path=Path(accepted['local_reference']);assert sha(reference_path)==accepted['local_reference_sha256']
    reference=json.loads(reference_path.read_bytes());assert reference['arrays_sha256']==accepted['local_arrays_sha256']
    assert reference['source']==capture['source'];oracle=load_arrays(reference)
    model=TensorBundle.load(imp['model']);names=model.metadata['parameter_names'];point=model.arrays['scalars__parameter_initial']
    np.testing.assert_array_equal(names,oracle['parameter_names']);np.testing.assert_array_equal(point,oracle['point'])
    with np.load(capture['arrays'],allow_pickle=False) as a:
        np.testing.assert_array_equal(point,a['point']);gradient_reference=a['gradient'];nll_reference=float(a['vg_value'])
    fixed=model.arrays['scalars__parameter_fixed'];assert fixed.sum()==4;free=~fixed
    report=dict(status='running',stage='prepare',script_sha256=sha(__file__),tensor_sources=sources,
        import_report=str(imp_path),import_report_sha256=sha(imp_path),gradient_gate=str(gate_path),gradient_gate_sha256=sha(gate_path),
        reference_report=str(reference_path),reference_report_sha256=sha(reference_path),reference_arrays_sha256=reference['arrays_sha256'],
        prior_reference_audit=str(prior_audit),prior_reference_audit_sha256=sha(prior_audit),source=capture['source'],
        tests_log='reports/tensor_curvature_tests.log',tests_log_sha256=sha('reports/tensor_curvature_tests.log'),
        combine_version=imp['combine_version'],root_version=imp['root_version'],runtime_flags=imp['runtime_flags'],
        parameters=len(point),column_batch_size=args.batch_size,columns=len(point),column_batches=(len(point)+args.batch_size-1)//args.batch_size,
        hostname=platform.node(),jax_version=jax.__version__,backend=jax.default_backend(),float64=bool(jax.config.x64_enabled),
        cpu_affinity=sorted(os.sched_getaffinity(0)),xla_flags=os.environ.get('XLA_FLAGS'),timings={},
        execution='One JIT, one gradient linearization, complete Hessian assembled by bounded column batches in lax.while_loop; parameters and blocks stay on device.',
        scope='Complete nominal tensor Hessian versus the saved full old CPU AD oracle on free coordinates. Native full-Hessian, fit/covariance, repeated-call and GPU acceptance are not established.',
        excluded_fixed_parameters=np.array(names)[fixed].tolist(),rss_limit_mib=11264,compiler_allocation_limit_bytes=8*1024**3)
    assert report['float64'] and report['backend']=='cpu' and len(report['cpu_affinity'])<=8
    assert '12 passed, 2 warnings' in Path(report['tests_log']).read_text()
    jax.config.update('jax_enable_compilation_cache',False)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary=path.with_suffix('.json.tmp');temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    save();stop=threading.Event()
    def monitor():
        ticks=0
        while not stop.wait(1.):
            ticks+=1;rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
            if rss>report['rss_limit_mib']:
                guard=dict(status='terminated_resource_limit',stage=report['stage'],rss_mib=rss,limit_mib=report['rss_limit_mib'],elapsed_seconds=time.perf_counter()-started)
                path.with_name(path.stem+'_guard.json').write_text(json.dumps(guard,indent=2)+'\n')
                print(guard,flush=True);os._exit(75)
            if ticks%20==0:print('heartbeat',report['stage'],round(time.perf_counter()-started,1),round(rss,1),flush=True)
    worker=threading.Thread(target=monitor,daemon=True);worker.start()
    try:
        arrays=jax.device_put(dict(model.arrays));theta=jax.device_put(point)
        function=make_value_gradient_hessian(make_model_function(model),column_batch_size=args.batch_size)
        report['stage']='lower';save();before=time.perf_counter();lowered=jax.jit(function).lower(theta,arrays)
        report['timings']['lowering_seconds']=time.perf_counter()-before;report['stage']='compile';save()
        before=time.perf_counter();compiled=lowered.compile();report['timings']['compile_seconds']=time.perf_counter()-before
        memory=compiled.memory_analysis();report['compiler_memory']={n:getattr(memory,n) for n in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        assert sum(report['compiler_memory'][k] for k in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))<report['compiler_allocation_limit_bytes'], 'Compiler allocation exceeds resource budget'
        report['stage']='execute';save();before=time.perf_counter();result=jax.block_until_ready(compiled(theta,arrays))
        report['timings']['first_execution_seconds']=time.perf_counter()-before
        before=time.perf_counter();value,gradient,hessian=map(np.asarray,result);report['timings']['host_copy_seconds']=time.perf_counter()-before
        artifact=directory/'hessian.npz';np.savez_compressed(artifact,point=point,parameter_names=np.array(names),parameter_scales=oracle['parameter_scales'],value=value,gradient=gradient,hessian=hessian)
        report.update(stage='compare',arrays=str(artifact),arrays_sha256=sha(artifact));save()
        assert np.isfinite(hessian).all() and np.isfinite(gradient).all() and np.isfinite(value)
        np.testing.assert_array_equal(hessian[fixed,:],0.);np.testing.assert_array_equal(hessian[:,fixed],0.)
        absolute=np.abs(gradient-gradient_reference);relative=absolute/np.maximum(np.abs(gradient_reference),1e-8)
        report['gradient_comparison']=dict(max_absolute_error=float(np.max(absolute)),max_relative_error=float(np.max(relative)))
        report['nll_absolute_error']=abs(float(value)-nll_reference);assert report['nll_absolute_error']<1e-6 and np.max(relative)<1e-4
        report['reference_reassembly']=reassemble_local(reference,oracle)
        common=dict(point=point[free],parameter_names=np.array(names)[free],parameter_scales=oracle['parameter_scales'][free])
        comparison=matrix_errors(dict(common,hessian=oracle['hessian'][np.ix_(free,free)]),dict(common,hessian=hessian[np.ix_(free,free)]))
        report.update(comparison=comparison,full_matrix_shape=list(hessian.shape),scaled_absolute_tolerance=1e-6)
        assert comparison['passed'];assert all(sha(p)==h for p,h in sources.items()) and 'ROOT' not in sys.modules
        report.update(status='validated_old_ad',stage='complete')
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:stop.set();worker.join();save()
    print(report['status'],report['timings'],report['comparison'],flush=True)


if __name__=='__main__':main()
