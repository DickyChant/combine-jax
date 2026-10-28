"""Cold compile scaling for repeated genuine HHVV 2D/3D channel structures."""
import argparse
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time

from tensor_benchmark_common import Resources, provenance, save_json, sha
from tensor_benchmark_provenance import verified_source_record


FAMILIES = {'ww_2d':'WW_lvqq_spin0_e_HP_bb_LDy_Run2', 'vv_3d':'VV_qqqq_spin0_ch1'}


def worker(args):
    import jax
    import numpy as np
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import unpack_model, compose_binned, make_model_function
    from combine_jax.tensor.binned_channels import pack_binned_channels
    from combine_jax.tensor.constraints import pack_constraints
    from combine_jax.tensor.counting import pack_counting_channels
    assert jax.config.x64_enabled and jax.default_backend()=='cpu'
    jax.config.update('jax_enable_compilation_cache',False)
    report=dict(provenance(),status='running',script_sha256=sha(__file__),family=args.family,
        channels=args.channels,jax_version=jax.__version__,float64=True,backend='cpu',operations={})
    monitor=Resources(False,args.output)
    def save():
        report['resources']=monitor.snapshot();save_json(args.output,report)
    save()
    try:
        exported=json.loads(Path('reports/tensor_hhvv_scalars.json').read_text())
        assert sha(exported['export'])==exported['export_sha256']
        with Path(exported['export']).open('rb') as stream:
            payload=pickle.load(stream)
        full_path=Path('build/tensor_hhvv_fit_import/model')
        full=TensorBundle.load(full_path)
        scalars=unpack_model(full)['scalars']
        source=next(row for row in payload['channels'] if row[0]==FAMILIES[args.family])
        name,pdf,axes,coordinates,weights,storage,masked=source
        assert not masked
        rows=[dict(name=f'{name}_repeat_{i}',pdf=pdf,observables=axes,coordinates=coordinates,
                   weights=weights,masked=False) for i in range(args.channels)]
        before=time.perf_counter()
        binned=pack_binned_channels(payload['nodes'],rows,scalars,compact_data=True)
        constraints=pack_constraints(payload['nodes'],[],{},scalars)
        counting=pack_counting_channels(payload['nodes'],[],scalars)
        model=compose_binned(scalars,constraints,counting,binned)
        report.update(import_seconds=time.perf_counter()-before,source_channel=name,
            source_dimensions=len(axes),source_data_rows=len(weights),source_nonzero_rows=int(np.count_nonzero(weights)),
            density_batches=len(binned.metadata['program']),nll_batches=len(binned.metadata['nll_program']),
            structure_key=model.structure_key,buffer_bytes=sum(a.nbytes for a in model.arrays.values()),
            scalar_batches=len(scalars.metadata['schedule']),parameters=len(model.metadata['parameter_names']),
            input_export=exported['export'],input_export_sha256=exported['export_sha256'],
            full_model_sha256={n:sha(full_path/n) for n in ('model.json','arrays.npz')})
        native_path=Path('reports/hhvv_random_values.json')
        native=json.loads(native_path.read_text())
        assert native['source']==payload['source']
        expected=next(row['combine'] for row in native['cases'][0]['channels'] if row['name']==name)
        report.update(native_reference=str(native_path),native_reference_sha256=sha(native_path),
                      native_single_channel_nll=expected)
        point=jax.device_put(np.array(native['cases'][0]['theta']))
        arrays=jax.device_put(dict(model.arrays));jax.block_until_ready((point,arrays))
        function=make_model_function(model)
        numeric={}
        for operation,fn in [('nll',function),('value_gradient',jax.value_and_grad(function))]:
            monitor.stage=operation;save()
            before=time.perf_counter();trace=jax.make_jaxpr(fn)(point,arrays)
            trace_seconds=time.perf_counter()-before
            assert not trace.consts
            before=time.perf_counter();lowered=jax.jit(fn).lower(point,arrays)
            lower_seconds=time.perf_counter()-before
            before=time.perf_counter();compiled=lowered.compile()
            compile_seconds=time.perf_counter()-before
            memory=compiled.memory_analysis()
            static_bytes=sum(getattr(memory,n) for n in ('argument_size_in_bytes','output_size_in_bytes','temp_size_in_bytes'))
            assert static_bytes<8*1024**3
            before=time.perf_counter();result=jax.device_get(compiled(point,arrays))
            execute_seconds=time.perf_counter()-before
            value=float(result if operation=='nll' else result[0])
            per_channel_error=abs(value/args.channels-expected)
            assert per_channel_error<1e-6
            report['operations'][operation]=dict(top_level_equations=len(trace.jaxpr.eqns),closed_over_arrays=len(trace.consts),
                trace_seconds=trace_seconds,lower_seconds=lower_seconds,compile_seconds=compile_seconds,
                trace_lower_compile_seconds=trace_seconds+lower_seconds+compile_seconds,
                first_execute_including_host_copy_seconds=execute_seconds,static_allocation_bytes=static_bytes,
                per_channel_native_nll_absolute_error=per_channel_error)
            if operation=='value_gradient':numeric['gradient_per_channel']=np.array(result[1])/args.channels
            save();print('measured',args.family,args.channels,operation,compile_seconds,flush=True)
        assert 'ROOT' not in sys.modules and all(sha(p)==digest for p,digest in report['sources'].items())
        np.savez_compressed(args.output.with_suffix('.npz'),**numeric)
        report.update(status='measured_native_value_checked',arrays=str(args.output.with_suffix('.npz')),
                      arrays_sha256=sha(args.output.with_suffix('.npz')),root_imported=False)
        save()
    except BaseException as exc:
        report.update(status='failed',error=repr(exc));save();raise
    finally:
        monitor.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family',choices=tuple(FAMILIES))
    parser.add_argument('--channels',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--directory',type=Path)
    parser.add_argument('--summarize-existing',action='store_true')
    parser.add_argument('--worker-revision',help='Required committed producer revision when summarizing existing measurements')
    args=parser.parse_args();assert not args.output.exists()
    if args.family:
        assert args.channels>0
        worker(args);return
    import numpy as np
    assert args.directory
    if args.summarize_existing:
        assert args.directory.is_dir() and args.worker_revision
    else:
        assert not args.directory.exists() and args.worker_revision is None
        args.directory.mkdir(parents=True)
    report=dict(status='running',script_sha256=sha(__file__),cases=[],
        worker_source_revision=args.worker_revision,
        scope='Controlled repetition of one genuine WW lvqq 2D and one VV qqqq 3D channel, with their exact imported native evaluation modes. All copies share the original floating parameters and scalar program. Constraints are excluded. Not scaling of independent nuisance dimensions or all 200 census structures.',
        policy='Channel counts 1,4,16; three sequential fresh processes each, persistent cache disabled, CPU float64. Runtime buffers are not closed-over constants. Native nominal channel NLL checked per copy; gradient-per-copy consistency is an AD replication check, not a new native derivative reference.')
    for family in FAMILIES:
        for channels in (1,4,16):
            for repetition in range(3):
                output=args.directory/f'{family}_{channels}_{repetition}.json'
                command=[sys.executable,__file__,'--family',family,'--channels',str(channels),'--output',str(output)]
                if not args.summarize_existing:
                    with output.with_suffix('.log').open('x') as stream:
                        subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,check=True)
                result=json.loads(output.read_text())
                assert result['status']=='measured_native_value_checked' and result['float64']
                assert result['family']==family and result['channels']==channels
                verified=verified_source_record(result,'scripts/benchmark_tensor_hhvv_scaling.py',args.worker_revision)
                if 'worker_source_evidence' not in report:
                    report.update(worker_source_evidence=verified,host=result['host'],
                        cpu_model=result['cpu_model'],cpu_affinity=result['cpu_affinity'],
                        environment=result['environment'],jax_version=result['jax_version'],float64=True)
                assert report['worker_source_evidence']==verified
                archived=args.output.with_name(args.output.stem+'_'+family+f'_{channels}_{repetition}.json')
                assert not archived.exists();archived.write_bytes(output.read_bytes())
                report['cases'].append(dict(family=family,channels=channels,repetition=repetition,
                    report=str(archived),report_sha256=sha(archived),operations=result['operations'],
                    density_batches=result['density_batches'],nll_batches=result['nll_batches'],
                    buffer_bytes=result['buffer_bytes'],peak_rss_mib=result['resources']['peak_rss_mib']))
                save_json(args.output,report);print('completed',family,channels,repetition,flush=True)
    summaries=[]
    for family in FAMILIES:
        cases=[c for c in report['cases'] if c['family']==family]
        reference=None
        for case in cases:
            measured=json.loads(Path(case['report']).read_text())
            assert sha(case['report'])==case['report_sha256']
            assert sha(measured['arrays'])==measured['arrays_sha256']
            with np.load(measured['arrays'],allow_pickle=False) as arrays:gradient=arrays['gradient_per_channel']
            if reference is None:reference=gradient
            np.testing.assert_allclose(gradient,reference,atol=1e-9,rtol=1e-9)
        for operation in ('nll','value_gradient'):
            # Shape-one broadcasts can add/remove a few trace operations. Record
            # that specialization rather than asserting identical equation counts.
            equations={n:sorted({c['operations'][operation]['top_level_equations'] for c in cases if c['channels']==n}) for n in (1,4,16)}
            summaries.append(dict(family=family,operation=operation,top_level_equations_by_channels=equations,
                constant_trace_for_batched_channels=equations[4]==equations[16],
                sizes=[dict(channels=n,median_compile_seconds=float(np.median([c['operations'][operation]['compile_seconds'] for c in cases if c['channels']==n])),
                    median_trace_lower_compile_seconds=float(np.median([c['operations'][operation]['trace_lower_compile_seconds'] for c in cases if c['channels']==n]))) for n in (1,4,16)]))
    report.update(status='measured_native_values_and_replication_validated',summaries=summaries)
    save_json(args.output,report)


if __name__=='__main__':
    main()
