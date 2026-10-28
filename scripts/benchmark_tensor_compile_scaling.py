"""Measure cold compilation as homogeneous channel count grows, in fresh processes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tensor_benchmark_common import provenance, save_json, sha


def worker(count, output):
    import jax
    import numpy as np
    from combine_jax.graph import Parameter
    from combine_jax.tensor.scalar import import_scalars
    from combine_jax.tensor.constraints import pack_constraints
    from combine_jax.tensor.counting import pack_counting_channels
    from combine_jax.tensor.model import compose_counting, make_model_function
    assert jax.config.x64_enabled and jax.default_backend() == 'cpu'
    assert 'ROOT' not in sys.modules
    jax.config.update('jax_enable_compilation_cache', False)
    parameters = [Parameter('rate', 12., .01, 100., 1., 'poi')]
    nodes = {'rate': ('RooRealVar', (), dict(value=12., constant=False)),
        'x': ('RooRealVar', (), dict(value=.5, minimum=0., maximum=1., constant=False)),
        'uniform': ('RooUniform', ('x',), {}),
        'pdf': ('RooAddPdf', ('uniform','rate'), dict(extended=True, pdfs=['uniform'], coefs=['rate']))}
    scalars = import_scalars(nodes, parameters, ['rate'], fixed_value=lambda n: nodes[n][2]['value'], observables=['x'])
    constraints = pack_constraints(nodes, [], {}, scalars)
    weights = np.linspace(8., 12., count)
    rows = [dict(name=f'channel_{i}', pdf='pdf', observables=['x'], weights=np.array([w])) for i, w in enumerate(weights)]
    channels = pack_counting_channels(nodes, rows, scalars)
    model = compose_counting(scalars, constraints, channels)
    function = make_model_function(model)
    theta = jax.device_put(np.array([12.]))
    arrays = jax.device_put(dict(model.arrays))
    jax.block_until_ready((theta, arrays))
    report = dict(provenance(), status='measuring', script_sha256=sha(__file__), channels=count,
        jax_version=jax.__version__, float64=True, backend='cpu', structure_key=model.structure_key,
        groups=len(channels.metadata['schedule']), buffers_bytes=sum(a.nbytes for a in model.arrays.values()),
        operations={}, cache_enabled=False)
    for name, fn in [('nll', function), ('value_gradient', jax.value_and_grad(function))]:
        before = time.perf_counter()
        trace = jax.make_jaxpr(fn)(theta, arrays)
        trace_seconds = time.perf_counter()-before
        assert not trace.consts
        before = time.perf_counter()
        lowered = jax.jit(fn).lower(theta, arrays)
        lower = time.perf_counter()-before
        before = time.perf_counter()
        compiled = lowered.compile()
        compile_time = time.perf_counter()-before
        before = time.perf_counter()
        result = jax.block_until_ready(compiled(theta, arrays))
        execution = time.perf_counter()-before
        value = float(result if name == 'nll' else result[0])
        # Same saturated-offset Poisson convention as the imported uniform channels.
        expected = float(np.sum(12.-weights-weights*np.log(12./weights)))
        assert abs(value-expected) < 1e-8
        row = dict(top_level_equations=len(trace.jaxpr.eqns), closed_over_arrays=len(trace.consts),
            trace_seconds=trace_seconds, lower_seconds=lower, compile_seconds=compile_time,
            trace_lower_compile_seconds=trace_seconds+lower+compile_time,
            first_execute_seconds=execution, nll_absolute_error=abs(value-expected))
        if name == 'value_gradient':
            gradient = float(np.asarray(result[1])[0])
            row['gradient_absolute_error'] = abs(gradient-float(np.sum(1.-weights/12.)))
            assert row['gradient_absolute_error'] < 1e-8
        report['operations'][name] = row
    report['status'] = 'measured_validated_synthetic'
    save_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--channels', type=int)
    parser.add_argument('--output', type=Path, default=Path('reports/tensor_compile_scaling.json'))
    args = parser.parse_args()
    assert not args.output.exists()
    if args.channels is not None:
        assert args.channels >= 2
        worker(args.channels, args.output)
        return
    directory = Path('build/tensor_compile_scaling')
    directory.mkdir(exist_ok=False)
    report = dict(status='running', script_sha256=sha(__file__), cases=[],
        scope='Synthetic one-structure uniform/Poisson channels sharing one floating rate; fixed parameter dimension. This does not establish scaling for every HHVV density family.',
        policy='Five sizes, three repetitions each, sequential fresh processes, persistent cache disabled. Explicit trace time precedes lowering; all three preparation times are reported.')
    for count in (8, 32, 128, 512, 2048):
        for repeat in range(3):
            output = directory/f'channels_{count}_{repeat}.json'
            command = [sys.executable, __file__, '--channels', str(count), '--output', str(output)]
            with output.with_suffix('.log').open('x') as stream:
                subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
            measured = json.loads(output.read_text())
            report['cases'].append(dict(channels=count, repeat=repeat, report=str(output), sha256=sha(output),
                groups=measured['groups'], operations=measured['operations']))
            save_json(args.output, report)
            print('measured', count, repeat, flush=True)
    for operation in ('nll', 'value_gradient'):
        assert len({c['operations'][operation]['top_level_equations'] for c in report['cases']}) == 1
    report['status'] = 'measured_constant_trace_size'
    save_json(args.output, report)


if __name__ == '__main__':
    main()
