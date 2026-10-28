"""Validate all HHVV scalar tensor outputs against native Combine at 26 saved points.

Reuses the hash-bound existing export as a migration input; the saved output is
framework-neutral. This does not claim a complete tensor NLL or fit validation.
"""
from collections import Counter
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import pickle
import resource
import time
import traceback

import jax
import numpy as np

from combine_jax.graph import walk
from combine_jax.root_io import open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.jax_scalar import make_function
from combine_jax.tensor.scalar import SCALAR_CLASSES, dependency_flags, import_scalars, evaluate_numpy


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def errors(actual, reference, names):
    difference = np.abs(actual-reference)
    finite = np.all(np.isfinite(actual)) and np.all(np.isfinite(reference))
    threshold = 1e-10+1e-12*np.abs(reference)
    worst = int(np.argmax(difference/threshold))
    return dict(passed=bool(finite and np.all(difference <= threshold)),
        maximum_absolute_error=float(np.max(difference)),
        maximum_scaled_error=float(np.max(difference/np.maximum(1.,np.abs(reference)))),
        tolerance_ratio=float(difference[worst]/threshold[worst]),
        worst_node=names[worst], actual_at_worst=float(actual[worst]),
        reference_at_worst=float(reference[worst]), failures=int(np.count_nonzero(difference > threshold)))


def main():
    output = Path('reports/tensor_hhvv_scalars.json')
    directory = Path('build/tensor_hhvv_scalars')
    assert not output.exists() and not directory.exists(), 'Preserve existing evidence'
    started = time.perf_counter()
    report = dict(status='running', scope='All observable/profile-independent scalar nodes only; not a full NLL, gradient, Hessian or fit gate.',
        script_sha256=digest(__file__), cases=[],
        tolerances=dict(absolute=1e-10, relative=1e-12),
        combine_version=version('higgs-combine-tool'), jax_version=jax.__version__,
        float64=bool(jax.config.x64_enabled), backend=jax.default_backend())
    def save():
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        temporary.replace(output)
    save()
    try:
        census_path = Path('reports/structure_census.json')
        census = json.loads(census_path.read_bytes())
        cache = Path(census['export']['path'])
        assert digest(cache) == census['export']['sha256']
        assert digest(census['workspace']['path']) == census['workspace']['sha256']
        manifest_path = Path(census['export']['manifest'])
        assert digest(manifest_path) == census['export']['manifest_sha256']
        manifest = json.loads(manifest_path.read_bytes())
        with cache.open('rb') as stream:
            payload = pickle.load(stream)
        model = manifest['models']['hhvv']
        assert payload['source'] == model['source'] == {k:census['workspace'][k] for k in ('sha256','configuration')}
        nodes, parameters = payload['nodes'], payload['parameters']
        observables = {n for c in payload['channels'] for n in c[2]}
        profiles = {n for p in payload['profiles'] for n in p.parameters}
        flags = dependency_flags(nodes,parameters,observables=observables,profiles=profiles)
        roots = sorted(n for n in nodes if nodes[n][0] in SCALAR_CLASSES and not any(flags[n][1:]))
        report.update(census=str(census_path), census_sha256=digest(census_path),
            source=model['source'], export=str(cache), export_sha256=digest(cache),
            scalar_roots=len(roots), scalar_classes=dict(sorted(Counter(nodes[n][0] for n in roots).items())),
            excluded_observable_or_profile_scalar_nodes=sum(nodes[n][0] in SCALAR_CLASSES and any(flags[n][1:]) for n in nodes))
        source_files = sorted(Path('src/combine_jax/tensor').glob('*.py'))
        report['tensor_sources'] = {str(p):digest(p) for p in source_files}
        reference_values, jax_values, numpy_values = [], [], []
        with open_workspace(census['workspace']['path']) as workspace:
            report['root_version'] = str(__import__('ROOT').gROOT.GetVersion())
            assert report['root_version'] == manifest['root_version']
            assert report['combine_version'] == manifest['combine_version']
            for n,settings in model['source']['configuration'].items():
                workspace.var(n).setVal(settings['value'])
                workspace.var(n).setConstant(settings['constant'])
            native = walk(workspace.obj('ModelConfig').GetPdf())
            before = time.perf_counter()
            bundle = import_scalars(nodes,parameters,roots,fixed_value=lambda n:float(native[n].getVal()),
                observables=observables,profiles=profiles,
                provenance=dict(source=model['source'], export_sha256=report['export_sha256'],
                    combine_version=report['combine_version'],root_version=report['root_version'],
                    census_sha256=report['census_sha256']))
            report['import_seconds'] = time.perf_counter()-before
            bundle.save(directory)
            bundle = TensorBundle.load(directory)
            report.update(bundle=str(directory), bundle_metadata_sha256=digest(directory/'model.json'),
                bundle_arrays_sha256=digest(directory/'arrays.npz'), structure_key=bundle.structure_key,
                scalar_slots=len(bundle.arrays['initial_values']), folded_nodes=len(bundle.metadata['folded_nodes']),
                folded_classes=bundle.metadata['folded_classes'],
                batches=bundle.metadata['batches'], maximum_depth=bundle.metadata['maximum_depth'],
                dynamic_instructions=bundle.metadata['dynamic_instructions'],
                buffer_bytes=sum(a.nbytes for a in bundle.arrays.values()),
                operation_batches=dict(sorted(Counter(s['operation'] for s in bundle.metadata['schedule']).items())),
                fixed_parameters=[p.name for p in parameters if p.minimum == p.maximum])
            arrays = jax.device_put(dict(bundle.arrays))
            function = make_function(bundle)
            nominal = np.array(model['reference']['cases'][0]['theta'])
            before = time.perf_counter()
            lowered = jax.jit(function).lower(nominal,arrays)
            report['lowering_seconds'] = time.perf_counter()-before
            before = time.perf_counter()
            compiled = lowered.compile()
            report['compile_seconds'] = time.perf_counter()-before
            report['closed_over_array_constants'] = len(jax.make_jaxpr(function)(nominal,arrays).consts)
            assert report['closed_over_array_constants'] == 0
            save()
            for i,case in enumerate(model['reference']['cases']):
                theta = np.array(case['theta'],dtype=np.float64)
                for p,v in zip(parameters,theta,strict=True):
                    workspace.var(p.name).setVal(float(v))
                ref = np.array([native[n].getVal() for n in roots],dtype=np.float64)
                before = time.perf_counter()
                value = np.array(compiled(theta,arrays).block_until_ready())
                seconds = time.perf_counter()-before
                eager = evaluate_numpy(bundle,theta)
                row = dict(index=i,label=case.get('label',str(i)),
                    jax_vs_native=errors(value,ref,roots),numpy_vs_native=errors(eager,ref,roots),
                    execution_seconds=seconds)
                report['cases'].append(row)
                reference_values.append(ref); jax_values.append(value); numpy_values.append(eager)
                save()
                print('case',i,'jax error',row['jax_vs_native']['maximum_absolute_error'],
                      'scaled',row['jax_vs_native']['maximum_scaled_error'],
                      'passed',row['jax_vs_native']['passed'] and row['numpy_vs_native']['passed'],flush=True)
            arrays_path = directory/'validation.npz'
            np.savez_compressed(arrays_path, names=np.array(roots),
                parameter_names=np.array([p.name for p in parameters]),
                points=np.array([c['theta'] for c in model['reference']['cases']]),
                native=np.array(reference_values), jax=np.array(jax_values), numpy=np.array(numpy_values))
            report.update(validation_arrays=str(arrays_path), validation_arrays_sha256=digest(arrays_path))
        assert len(report['cases']) == 26
        assert all(c['jax_vs_native']['passed'] and c['numpy_vs_native']['passed'] for c in report['cases'])
        assert all(digest(p) == h for p,h in report['tensor_sources'].items())
        report['status'] = 'scalar_values_checked'
    except BaseException:
        report.update(status='failed',exception=traceback.format_exc())
        raise
    finally:
        save()
    print(json.dumps({k:v for k,v in report.items() if k not in ('cases','tensor_sources','scalar_classes')},indent=2))


if __name__ == '__main__':
    main()
