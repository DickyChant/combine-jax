"""Capture one full tensor HHVV AD gradient; apply native gates separately."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import sys
import time
import traceback

import jax
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--values',default='reports/tensor_hhvv_compact_values.json')
    parser.add_argument('--output',default='reports/tensor_hhvv_gradients_capture.json')
    parser.add_argument('--directory',default='build/tensor_hhvv_gradients')
    args=parser.parse_args();output=Path(args.output);directory=Path(args.directory)
    assert not output.exists() and not directory.exists();directory.mkdir()
    started=time.perf_counter();value_path=Path(args.values);values=json.loads(value_path.read_bytes())
    assert values['status']=='passed' and values['float64']
    import_path=Path(values['import_report']);assert sha(import_path)==values['import_report_sha256']
    imp=json.loads(import_path.read_bytes());assert imp['status']=='passed' and imp['channels']==459
    for p,h in {**values['tensor_sources'],**imp['artifacts']}.items():assert sha(p)==h
    native_path=Path(values['native_report']);assert sha(native_path)==values['native_report_sha256']
    native=json.loads(native_path.read_bytes())
    assert sha(values['validation_arrays'])==values['validation_arrays_sha256']
    with np.load(values['validation_arrays'],allow_pickle=False) as a:
        points=a['points'];nll_values=a['jax_nll'];native_nll=a['native_nll']
    np.testing.assert_array_equal(points,[c['theta'] for c in native['cases']])
    old_path=Path('reports/gpu_hhvv_cpu_scoped_gradient_all_points_checkpoint.json')
    old=json.loads(old_path.read_bytes());assert sha(old['arrays'])==old['arrays_sha256']
    accepted_path=Path('reports/gpu_hhvv_scoped_gradient_comparison.json')
    accepted=json.loads(accepted_path.read_bytes());assert accepted['status']=='passed'
    assert accepted['provenance']['cpu']['benchmark_sha256']==sha(old_path)
    with np.load(old['arrays'],allow_pickle=False) as a:
        names=a['parameter_names'];labels=a['validation_labels'];old_gradients=a['validation_gradients']
        np.testing.assert_array_equal(a['validation_points'],points)
    model=TensorBundle.load(imp['model']);assert model.metadata['parameter_names']==names.tolist()==native['parameter_names']
    assert values['source']==old['source']==native['source']
    fixed=model.arrays['scalars__parameter_fixed'];assert np.count_nonzero(fixed)==4
    assert points.shape==old_gradients.shape==(26,2656)
    report=dict(status='running',stage='prepare',scope='Full tensor NLL and AD gradient capture at all 26 points. Old AD comparison is supplemental; native nominal/random FD gates must run separately. No Hessian or fit acceptance.',
        script_sha256=sha(__file__),values_report=str(value_path),values_report_sha256=sha(value_path),
        import_report=str(import_path),import_report_sha256=sha(import_path),tensor_sources=values['tensor_sources'],
        old_checkpoint=str(old_path),old_checkpoint_sha256=sha(old_path),old_arrays_sha256=old['arrays_sha256'],
        old_comparison=str(accepted_path),old_comparison_sha256=sha(accepted_path),
        source=values['source'],combine_version=imp['combine_version'],root_version=imp['root_version'],runtime_flags=imp['runtime_flags'],
        parameters=2656,channels=459,backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],
        jax_version=jax.__version__,python=platform.python_version(),float64=bool(jax.config.x64_enabled),hostname=platform.node(),
        cpu_affinity=sorted(os.sched_getaffinity(0)),cpu_model=values['cpu_model'],
        environment={k:os.environ.get(k) for k in ('XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        structure_key=model.structure_key,fixed_parameters=names[fixed].tolist(),cases=[],timings={},
        execution='One complete JIT value_and_grad objective, runtime model arrays, device reductions and profile solve; no per-term host accumulation.',
        cache_policy='Persistent compilation cache disabled for this measurement.',
        timing_scope='Local CPU only; synchronized warm calls exclude host copies and model import; no cross-host performance comparison.')
    assert report['float64'] and report['backend']=='cpu' and report['hostname']==values['hostname']
    jax.config.update('jax_enable_compilation_cache',False)
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        tmp=output.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    save();gradients=[];vg_values=[];array_path=directory/'gradients.npz'
    def save_arrays():
        np.savez_compressed(array_path,parameter_names=names,point=points[0],value=nll_values[0],
            gradient=gradients[0],vg_value=vg_values[0],validation_points=points[:len(gradients)],validation_labels=labels[:len(gradients)],
            validation_values=nll_values[:len(gradients)],validation_vg_values=np.array(vg_values),validation_gradients=np.array(gradients))
        report.update(arrays=str(array_path),arrays_sha256=sha(array_path),captured_points=len(gradients))
    try:
        arrays=jax.device_put(dict(model.arrays));initial=jax.device_put(points[0])
        before=time.perf_counter();function=jax.value_and_grad(make_model_function(model))
        report['timings']['function_preparation_seconds']=time.perf_counter()-before
        before=time.perf_counter();assert not jax.make_jaxpr(function)(initial,arrays).consts
        report['timings']['trace_check_seconds']=time.perf_counter()-before
        report.update(stage='lower',closed_over_arrays=0);save()
        before=time.perf_counter();lowered=jax.jit(function).lower(initial,arrays)
        report['timings']['lowering_seconds']=time.perf_counter()-before;report['stage']='compile';save()
        before=time.perf_counter();compiled=lowered.compile();report['timings']['compile_seconds']=time.perf_counter()-before
        memory=compiled.memory_analysis()
        report['compiler_memory']={n:getattr(memory,n) for n in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes','alias_size_in_bytes')}
        report['stage']='validate';save()
        for i,point in enumerate(points):
            before=time.perf_counter();value,gradient=compiled(jax.device_put(point),arrays);value=float(value);gradient=np.asarray(gradient)
            elapsed=time.perf_counter()-before;gradients.append(gradient);vg_values.append(value);save_arrays()
            row=dict(index=i,label=str(labels[i]),nonfinite_indices=np.flatnonzero(~np.isfinite(gradient)).tolist(),execution_and_copy_seconds=elapsed)
            report['cases'].append(row);save()
            assert np.isfinite(value) and not row['nonfinite_indices']
            np.testing.assert_array_equal(gradient[fixed],np.zeros(np.count_nonzero(fixed)))
            absolute=np.abs(gradient[~fixed]-old_gradients[i,~fixed]);relative=absolute/np.maximum(np.abs(old_gradients[i,~fixed]),1e-8)
            row.update(nll_absolute_error=abs(value-native_nll[i]),old_ad_max_absolute_error=float(np.max(absolute)),
                old_ad_max_relative_error=float(np.max(relative)),old_ad_worst_relative_parameter=str(names[~fixed][np.argmax(relative)]))
            save();print(i,row,flush=True)
            assert row['nll_absolute_error']<1e-6 and row['old_ad_max_relative_error']<1e-4
        report['stage']='warm';save();warm=[]
        for i in range(11):
            before=time.perf_counter();jax.block_until_ready(compiled(initial,arrays));elapsed=time.perf_counter()-before
            if i:warm.append(elapsed)
        report['timings'].update(warm_seconds=warm,median_warm_seconds=float(np.median(warm)))
        assert all(sha(p)==h for p,h in report['tensor_sources'].items()) and 'ROOT' not in sys.modules
        report.update(status='captured',stage='complete')
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['timings'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
