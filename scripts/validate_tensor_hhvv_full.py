"""Full neutral HHVV model against native nominal/random and saved fit-point values."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import time
import traceback

import jax
import jax.numpy as jnp
import numpy as np

from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import make_model_function,evaluate_model_numpy,unpack_model
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.counting import make_counting_function
from combine_jax.tensor.templates import make_template_function
from combine_jax.tensor.binned_channels import make_binned_function
from combine_jax.tensor.constraints import make_constraint_function


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def error(actual,reference,tolerance=1e-6):
    delta=np.abs(actual-reference)
    return dict(max_absolute_error=float(np.max(delta,initial=0.)),passed=bool(np.all(np.isfinite(actual)) and np.all(np.isfinite(reference)) and np.all(delta<tolerance)))


def diagnostics(model):
    components=unpack_model(model);keys={k:tuple(c.arrays) for k,c in components.items()}
    sf=make_function(components['scalars']);cf=make_counting_function(components['channels']);tf=make_template_function(components['templates'])
    bf=make_binned_function(components['binned']);pf=make_constraint_function(components['constraints'])
    def evaluate(theta,arrays):
        b={k:{n:arrays[k+'__'+n] for n in names} for k,names in keys.items()};values=sf(theta,b['scalars'])
        template,profiles,_=tf(values,b['templates'])
        channels=jnp.concatenate((cf(values,b['channels']),template,bf(values,b['binned'])))
        return channels,pf(values,profiles,b['constraints']),profiles
    return evaluate


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='reports/tensor_hhvv_full_values.json')
    parser.add_argument('--directory',default='build/tensor_hhvv_full_validation');parser.add_argument('--import-report',default='reports/tensor_hhvv_full_import.json')
    parser.add_argument('--runtime-source',action='append',default=[],help='Explicit changed runtime source to validate with the unchanged imported artifact; repeat for each changed file')
    args=parser.parse_args()
    path=Path(args.output);directory=Path(args.directory);assert not path.exists() and not directory.exists();directory.mkdir()
    started=time.perf_counter();import_path=Path(args.import_report);imp=json.loads(import_path.read_bytes())
    assert imp['status']=='passed' and imp['channels']==459 and imp['root_evaluation_errors']==0
    for p,h in imp['artifacts'].items():assert sha(p)==h
    sources={str(p):sha(p) for p in sorted(Path('src/combine_jax/tensor').glob('*.py'))}
    assert sources.keys()==imp['tensor_sources'].keys()
    updates={p:dict(import_sha256=h,runtime_sha256=sources[p]) for p,h in imp['tensor_sources'].items() if h!=sources[p]}
    assert set(updates)==set(args.runtime_source), 'Every runtime source change must be declared explicitly'
    native_path=Path('reports/hhvv_random_values.json');native=json.loads(native_path.read_bytes());terms_path=Path('reports/tensor_hhvv_likelihood_terms.json');terms=json.loads(terms_path.read_bytes())
    point_path=Path('reports/gpu_local_hhvv_native_hesse1_point_check.json');point_ref=json.loads(point_path.read_bytes())
    source={k:imp['source'][k] for k in ('sha256','configuration')}
    assert native['source']==source==point_ref['source'] and native['runtime_flags']==imp['runtime_flags']==point_ref['runtime_flags']
    assert native['combine_version']==imp['combine_version']==point_ref['combine_version'] and native['root_version']==imp['root_version']==point_ref['root_version']
    assert point_ref['status']=='point_checked' and all(q['roofit_evaluation_errors']==0 for q in point_ref['probes'])
    assert sha(terms['validation_arrays'])==terms['validation_arrays_sha256'];refs=np.load(terms['validation_arrays'],allow_pickle=False)
    model=TensorBundle.load(imp['model']);assert model.metadata['provenance']['all_card_channels_imported']
    assert model.metadata['parameter_names']==native['parameter_names']==point_ref['parameter_names']
    np.testing.assert_array_equal(model.arrays['scalars__parameter_initial'],native['initial'])
    assert model.metadata['provenance']['discrete_choices']==point_ref['discrete_choices']
    assert model.metadata['provenance']['channel_masks']==point_ref['channel_masks']
    assert model.metadata['components']['templates']['profile_names']==refs['profile_names'].tolist()
    constraint_order=[refs['constraint_names'].tolist().index(n) for n in model.metadata['constraint_names']]
    report=dict(status='running',stage='compile',scope='All 459 HHVV channel values and full NLL at nominal + 25 random native points, plus the checked native fit point. No HHVV gradient/Hessian/fit/covariance acceptance.',
        import_report=str(import_path),import_report_sha256=sha(import_path),native_report=str(native_path),native_report_sha256=sha(native_path),
        terms_report=str(terms_path),terms_report_sha256=sha(terms_path),point_report=str(point_path),point_report_sha256=sha(point_path),
        script_sha256=sha(__file__),source=source,combine_version=imp['combine_version'],root_version=imp['root_version'],runtime_flags=imp['runtime_flags'],
        tensor_sources=sources,import_tensor_sources=imp['tensor_sources'],runtime_source_updates=updates,
        jax_version=jax.__version__,python=platform.python_version(),float64=bool(jax.config.x64_enabled),
        backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],hostname=platform.node(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown'),
        environment={k:os.environ.get(k) for k in ('XLA_FLAGS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        parameter_names=model.metadata['parameter_names'],channel_names=model.metadata['channel_names'],constraint_names=model.metadata['constraint_names'],
        profile_names=model.metadata['components']['templates']['profile_names'],structure_key=model.structure_key,cases=[],timings={})
    assert report['float64'] and report['backend']=='cpu'
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(path)
    save()
    try:
        arrays=jax.device_put(dict(model.arrays));initial=np.array(native['initial']);functions={}
        for name,function in [('nll',make_model_function(model)),('diagnostics',diagnostics(model))]:
            before=time.perf_counter();lowered=jax.jit(function).lower(initial,arrays);report['timings'][name+'_lowering_seconds']=time.perf_counter()-before
            before=time.perf_counter();functions[name]=lowered.compile();report['timings'][name+'_compile_seconds']=time.perf_counter()-before
            assert not jax.make_jaxpr(function)(initial,arrays).consts
            save()
        report.update(stage='validate',closed_over_arrays=0);save()
        saved={k:[] for k in ('native_channels','jax_channels','native_constraints','jax_constraints','native_profiles','jax_profiles','native_nll','jax_nll','numpy_nll')}
        for i,case in enumerate(native['cases']):
            point=np.array(case['theta']);np.testing.assert_array_equal(point,refs['points'][i])
            value=float(functions['nll'](point,arrays));channels,constraints,profiles=map(np.asarray,functions['diagnostics'](point,arrays))
            lookup={q['name']:q['combine'] for q in case['channels']};expected=np.array([lookup[n] for n in report['channel_names']])
            expected_constraints=refs['native_constraints'][i,constraint_order];expected_profiles=refs['profile_values'][i]
            numpy_value=float(evaluate_model_numpy(model,point))
            row=dict(index=i,nll=error(np.array(value),np.array(case['combine_nll'])),numpy_nll=error(np.array(numpy_value),np.array(case['combine_nll'])),
                channels=error(channels,expected),constraints=error(constraints,expected_constraints),profiles=error(profiles,expected_profiles,1e-10))
            report['cases'].append(row);save();print('HHVV',i,row,flush=True)
            for k,v in [('native_channels',expected),('jax_channels',channels),('native_constraints',expected_constraints),('jax_constraints',constraints),
                ('native_profiles',expected_profiles),('jax_profiles',profiles),('native_nll',case['combine_nll']),('jax_nll',value),('numpy_nll',numpy_value)]:saved[k].append(v)
        point=np.array(point_ref['point']);value=float(functions['nll'](point,arrays));numpy_value=float(evaluate_model_numpy(model,point))
        report['native_point']=dict(theta=point.tolist(),native_nll=point_ref['canonical_candidate_nll'],jax_nll=value,numpy_nll=numpy_value,
            jax=error(np.array(value),np.array(point_ref['canonical_candidate_nll'])),numpy=error(np.array(numpy_value),np.array(point_ref['canonical_candidate_nll'])),
            scope='Value agreement at the saved native point only; native covariance was invalid, and no fit/uncertainty agreement is claimed.')
        output=directory/'validation.npz';np.savez_compressed(output,**{k:np.array(v) for k,v in saved.items()},points=refs['points'],
            channel_names=np.array(report['channel_names']),constraint_names=np.array(report['constraint_names']),profile_names=np.array(report['profile_names']))
        report.update(validation_arrays=str(output),validation_arrays_sha256=sha(output))
        assert len(report['cases'])==26 and all(q[k]['passed'] for q in report['cases'] for k in ('nll','numpy_nll','channels','constraints','profiles'))
        assert report['native_point']['jax']['passed'] and report['native_point']['numpy']['passed']
        assert all(sha(p)==h for p,h in report['tensor_sources'].items())
        report.update(status='passed',stage='complete')
    except BaseException:report.update(status='failed',exception=traceback.format_exc());raise
    finally:save()
    print(report['status'],report['timings'],report['peak_rss_mib'],flush=True)


if __name__=='__main__':main()
