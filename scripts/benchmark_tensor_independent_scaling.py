"""Independent-parameter scaling of genuine HHVV channel structures.

Prepare neutral models and graph-oracle references locally; cold workers need
only JSON/NPZ inputs and JAX. Channel copies share no floating parameters,
including their POI. Constraints are excluded, as in the shared-parameter test.
"""
import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha


FAMILIES = {'ww_2d': 'WW_lvqq_spin0_e_HP_bb_LDy_Run2', 'vv_3d': 'VV_qqqq_spin0_ch1'}
SIZES = (1, 4, 16)


def rename(value, names):
    if isinstance(value, str):
        return names.get(value, value)
    if isinstance(value, dict):
        return {rename(k, names): rename(v, names) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(rename(v, names) for v in value)
    if isinstance(value, list):
        return [rename(v, names) for v in value]
    return value  # Numeric template buffers are immutable import inputs.


def clone_channel(exported, channel, fixed_values, count):
    from combine_jax.discrete import active_dependencies
    from combine_jax.tensor.binned_channels import pack_binned_channels
    from combine_jax.tensor.constraints import pack_constraints
    from combine_jax.tensor.counting import pack_counting_channels
    from combine_jax.tensor.model import compose_binned
    from combine_jax.tensor.scalar import SCALAR_CLASSES, dependency_flags, import_scalars
    deps = sorted(active_dependencies(exported.nodes, [channel.pdf]))
    selected = [p for p in exported.parameters if p.name in deps]
    assert selected and not channel.masked
    nodes, parameters, rows, originals, observables = {}, [], [], {}, []
    for copy in range(count):
        names = {n: f'copy_{copy}__{n}' for n in deps}
        assert not (set(names.values()) & set(deps))
        for n in deps:
            node = exported.nodes[n]
            nodes[names[n]] = (node.class_name, rename(node.inputs, names), rename(dict(node.attrs), names))
            originals[names[n]] = n
        parameters.extend(replace(p, name=names[p.name]) for p in selected)
        axes = rename(channel.observables, names)
        observables.extend(axes)
        rows.append(dict(name=f'copy_{copy}', pdf=names[channel.pdf], observables=axes,
                         coordinates=rename(dict(channel.data.coordinates), names),
                         weights=channel.data.weights, masked=False))
    flags = dependency_flags(nodes, parameters, observables=observables)
    roots = sorted(n for n in nodes if nodes[n][0] in SCALAR_CLASSES and not any(flags[n][1:]))
    scalars = import_scalars(nodes, parameters, roots,
                            fixed_value=lambda n: fixed_values[originals[n]], observables=observables)
    binned = pack_binned_channels(nodes, rows, scalars, compact_data=True)
    model = compose_binned(scalars, pack_constraints(nodes, [], {}, scalars),
                           pack_counting_channels(nodes, [], scalars), binned)
    assert len(model.metadata['parameter_names']) == count*len(selected)
    assert len(set(model.metadata['parameter_names'])) == count*len(selected)
    return model, selected, dict(source_nodes=len(deps), density_batches=len(binned.metadata['program']),
        nll_batches=len(binned.metadata['nll_program']), scalar_batches=len(scalars.metadata['schedule']),
        dynamic_scalar_instructions=scalars.metadata['dynamic_instructions'],
        folded_nodes=len(scalars.metadata['folded_nodes']))


def prepare(args):
    import jax
    from hhvv_export_cache import load_export
    from combine_jax.discrete import active_dependencies
    from combine_jax.likelihood import Likelihood
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import unpack_model, evaluate_model_numpy
    from combine_jax.tensor.scalar import evaluate_numpy
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    assert not args.directory.exists() and not args.output.exists()
    args.directory.mkdir(parents=True)
    native_path = Path('reports/hhvv_random_values.json')
    native = json.loads(native_path.read_text())
    export_evidence = Path('reports/tensor_hhvv_compile_scaling_audited_ww_2d_1_0.json')
    evidence = json.loads(export_evidence.read_text())
    export_path = Path(evidence['input_export'])
    assert sha(export_path) == evidence['input_export_sha256']
    exported = load_export(export_path, native['source'])
    full_path = Path('build/tensor_hhvv_fit_import/model')
    full = TensorBundle.load(full_path)
    assert {n: sha(full_path/n) for n in ('model.json', 'arrays.npz')} == evidence['full_model_sha256']
    scalars = unpack_model(full)['scalars']
    folded = dict(zip(scalars.metadata['output_names'], evaluate_numpy(scalars, exported.initial), strict=True))
    indices = {name: i for i, name in enumerate(native['parameter_names'])}
    report = dict(provenance(), status='preparing', script_sha256=sha(__file__),
        export=str(export_path), export_sha256=sha(export_path),
        export_evidence=str(export_evidence), export_evidence_sha256=sha(export_evidence),
        native_reference=str(native_path), native_reference_sha256=sha(native_path),
        full_model_sha256={n: sha(full_path/n) for n in ('model.json', 'arrays.npz')},
        families={}, models=[],
        scope='Two genuine channel structures, cloned with independent parameter vectors (POI included). '
              'Fixed quantities retain certified native folded values. Constraints are excluded. '
              'Native 26-point channel values and independent old graph AD gradients are references. '
              'Preparation overlaps the local fit and is not a performance measurement.')
    save_json(args.output, report)
    for family, name in FAMILIES.items():
        channel = next(c for c in exported.channels if c.name == name)
        deps = active_dependencies(exported.nodes, [channel.pdf])
        parameters = [p for p in exported.parameters if p.name in deps]
        free = np.array([p.minimum != p.maximum for p in parameters])
        points = np.array([[case['theta'][indices[p.name]] for p in parameters] for case in native['cases']])
        expected = np.array([next(row['combine'] for row in case['channels'] if row['name'] == name)
                             for case in native['cases']])
        oracle = Likelihood(exported.nodes, exported.registry, [p for p in parameters if p.minimum != p.maximum],
                            [channel], [], 0.)
        values, gradients = [], []
        for point in points:
            value, grad = jax.device_get(oracle.value_and_grad(point[free]))
            values.append(float(value))
            complete = np.zeros(len(parameters)); complete[free] = grad; gradients.append(complete)
        values, gradients = np.array(values), np.array(gradients)
        assert np.isfinite(values).all() and np.isfinite(gradients).all()
        assert max(abs(values-expected)) < 1e-6
        ref = args.directory/(family+'_reference.npz')
        np.savez_compressed(ref, parameter_names=np.array([p.name for p in parameters]),
                            points=points, native_values=expected, oracle_values=values, gradients=gradients, free=free)
        report['families'][family] = dict(source_channel=name, dimensions=len(channel.observables),
            source_data_rows=len(channel.data.weights), nonzero_rows=int(np.count_nonzero(channel.data.weights)),
            parameters_per_channel=len(parameters), floating_per_channel=int(free.sum()),
            reference=str(ref), reference_sha256=sha(ref),
            oracle_native_nll_max_absolute_error=float(max(abs(values-expected))))
        for count in SIZES:
            model, selected, batches = clone_channel(exported, channel, folded, count)
            assert [p.name for p in selected] == [p.name for p in parameters]
            path = args.directory/f'{family}_{count}'
            model.save(path)
            # Different native points in different copies catch accidental sharing
            # or parameter-index leakage before any compilation is benchmarked.
            errors = []
            for base in range(26):
                cases = (np.arange(count)+base) % 26
                value = evaluate_model_numpy(model, points[cases].reshape(-1))
                errors.append(float(abs(value-expected[cases].sum())))
            assert max(errors) < 1e-6
            report['models'].append(dict(family=family, channels=count, model=str(path),
                model_sha256={n: sha(path/n) for n in ('model.json', 'arrays.npz')},
                structure_key=model.structure_key, parameters=len(model.metadata['parameter_names']),
                buffer_bytes=sum(a.nbytes for a in model.arrays.values()), batches=batches,
                mixed_points=26, numpy_native_nll_max_absolute_error=max(errors)))
            save_json(args.output, report)
            print('prepared', family, count, 'parameters', len(model.metadata['parameter_names']), flush=True)
    assert all(sha(p) == digest for p, digest in report['sources'].items())
    report.update(status='prepared_native_values_and_oracle_gradients', oracle_sources={
        str(p): sha(p) for p in sorted(Path('src/combine_jax').glob('*.py'))})
    save_json(args.output, report)


def equation_count(jaxpr):
    def children(value):
        if hasattr(value, 'jaxpr'): return equation_count(value.jaxpr)
        if hasattr(value, 'eqns'): return equation_count(value)
        if isinstance(value, dict): return sum(children(v) for v in value.values())
        if isinstance(value, (list, tuple)): return sum(children(v) for v in value)
        return 0
    return len(jaxpr.eqns)+sum(children(eqn.params) for eqn in jaxpr.eqns)


def worker(args):
    import jax
    import jaxlib
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.model import make_model_function
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    jax.config.update('jax_enable_compilation_cache', False)
    assert not args.output.exists()
    manifest = json.loads(args.manifest.read_text())
    assert manifest['status'] == 'prepared_native_values_and_oracle_gradients'
    case = next(r for r in manifest['models'] if r['family'] == args.family and r['channels'] == args.channels)
    ref = manifest['families'][args.family]
    assert sha(ref['reference']) == ref['reference_sha256']
    assert {n: sha(Path(case['model'])/n) for n in ('model.json', 'arrays.npz')} == case['model_sha256']
    model = TensorBundle.load(case['model'])
    with np.load(ref['reference'], allow_pickle=False) as a:
        base_points, values, gradients = a['points'], a['native_values'], a['gradients']
    selector = (np.arange(26)[:, None]+np.arange(args.channels)[None, :]) % 26
    points = base_points[selector].reshape(26, -1)
    expected_values = values[selector].sum(axis=1)
    expected_gradients = gradients[selector].reshape(26, -1)
    report = dict(provenance(), status='running', script_sha256=sha(__file__),
        manifest=str(args.manifest), manifest_sha256=sha(args.manifest), case=case,
        family=args.family, channels=args.channels, jax_version=jax.__version__, jaxlib_version=jaxlib.__version__,
        float64=True, backend='cpu', cache_enabled=False, operations={},
        scope='Fresh-process cold compile scaling with independent full channel parameter vectors. '
              '26 different mixtures of native points per capture; no parameter sharing across copies. '
              'Native NLL and old graph AD gradient checks are distinct. Not a fit or Hessian benchmark.')
    monitor = Resources(False, args.output)
    def save():
        report['resources'] = monitor.snapshot(); save_json(args.output, report)
    save()
    try:
        arrays = jax.device_put(dict(model.arrays)); resident = [jax.device_put(p) for p in points]
        jax.block_until_ready((arrays, resident))
        function = make_model_function(model)
        captures = dict(points=points, parameter_names=np.array(model.metadata['parameter_names']))
        for operation, fn in [('nll', function), ('value_gradient', jax.value_and_grad(function))]:
            monitor.stage = operation
            before = time.perf_counter(); trace = jax.make_jaxpr(fn)(resident[0], arrays)
            trace_seconds = time.perf_counter()-before
            assert not trace.consts
            before = time.perf_counter(); lowered = jax.jit(fn).lower(resident[0], arrays)
            lower_seconds = time.perf_counter()-before
            hlo_bytes = len(str(lowered.compiler_ir()).encode())
            before = time.perf_counter(); compiled = lowered.compile()
            compile_seconds = time.perf_counter()-before
            memory = compiled.memory_analysis()
            allocation = sum(getattr(memory, n) for n in ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes'))
            assert allocation < 8*1024**3
            before = time.perf_counter(); result = jax.block_until_ready(compiled(resident[0], arrays))
            first_seconds = time.perf_counter()-before
            results = [jax.device_get(result)] + [jax.device_get(compiled(p, arrays)) for p in resident[1:]]
            measured_values = np.array(results if operation == 'nll' else [v[0] for v in results])
            error = float(max(abs(measured_values-expected_values)))
            assert np.isfinite(measured_values).all() and error < 1e-6
            row = dict(top_level_equations=len(trace.jaxpr.eqns), recursive_equations=equation_count(trace.jaxpr),
                closed_over_arrays=len(trace.consts), stablehlo_text_bytes=hlo_bytes,
                trace_seconds=trace_seconds, lower_seconds=lower_seconds, compile_seconds=compile_seconds,
                trace_lower_compile_seconds=trace_seconds+lower_seconds+compile_seconds,
                first_execute_seconds=first_seconds, static_allocation_bytes=allocation,
                native_nll_max_absolute_error=error)
            captures[operation+'_values'] = measured_values
            if operation == 'value_gradient':
                actual = np.array([v[1] for v in results]); delta = abs(actual-expected_gradients)
                relative = float(np.max(delta/np.maximum(abs(expected_gradients), 1e-8)))
                assert np.isfinite(actual).all() and relative < 1e-4
                row.update(old_ad_gradient_max_absolute_error=float(delta.max()), old_ad_gradient_max_relative_error=relative)
                captures['gradients'] = actual
            report['operations'][operation] = row; save()
            print('measured', args.family, args.channels, operation, compile_seconds, flush=True)
        numeric = args.output.with_suffix('.npz'); assert not numeric.exists()
        np.savez_compressed(numeric, **captures)
        assert 'ROOT' not in sys.modules and all(sha(p) == digest for p, digest in report['sources'].items())
        report.update(status='measured_native_values_old_ad_gradients', arrays=str(numeric),
                      arrays_sha256=sha(numeric), root_imported=False)
        save()
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); save(); raise
    finally:
        monitor.close()


def campaign(args):
    assert not args.output.exists() and not args.directory.exists()
    args.directory.mkdir(parents=True)
    report = dict(status='running', started_unix=time.time(), script_sha256=sha(__file__),
                  manifest=str(args.manifest), manifest_sha256=sha(args.manifest), commands=[])
    save_json(args.output, report)
    try:
        for family in FAMILIES:
            for count in SIZES:
                for repetition in range(3):
                    output = args.directory/f'{family}_{count}_{repetition}.json'
                    command = [sys.executable, __file__, 'worker', '--manifest', str(args.manifest),
                               '--family', family, '--channels', str(count), '--output', str(output)]
                    row = dict(command=command, started_unix=time.time(), status='running')
                    report['commands'].append(row); save_json(args.output, report)
                    with output.with_suffix('.log').open('x') as stream:
                        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
                    row.update(returncode=result.returncode, finished_unix=time.time(),
                               status='passed' if result.returncode == 0 else 'failed')
                    save_json(args.output, report)
                    assert result.returncode == 0, output
                    print('completed', family, count, repetition, flush=True)
        report['status'] = 'completed_pending_audit'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report['finished_unix'] = time.time(); save_json(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'worker', 'campaign'))
    parser.add_argument('--directory', type=Path)
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--family', choices=tuple(FAMILIES))
    parser.add_argument('--channels', type=int, choices=SIZES)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    if args.action in ('prepare', 'campaign'): assert args.directory is not None
    if args.action != 'prepare': assert args.manifest is not None
    if args.action == 'worker': assert args.family is not None and args.channels is not None
    {'prepare': prepare, 'worker': worker, 'campaign': campaign}[args.action](args)


if __name__ == '__main__':
    main()
