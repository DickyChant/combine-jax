"""Inspect neutral buffer layouts and saved compiler estimates without execution."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np

from combine_jax.tensor.bundle import json_hash
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import require_blind_model
from tensor_benchmark_provenance import verified_source_record


def inspect_model(directory):
    directory = Path(directory)
    document = json.loads((directory/'model.json').read_text())
    metadata = document['metadata']
    assert json_hash(metadata) == document['metadata_sha256']
    require_blind_model(SimpleNamespace(metadata=metadata))
    assert sha(directory/'arrays.npz') == document['arrays_sha256']
    headers = {}
    with zipfile.ZipFile(directory/'arrays.npz') as archive:
        for name in archive.namelist():
            assert name.endswith('.npy') and name[:-4] not in headers
            with archive.open(name) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {(1, 0): np.lib.format.read_array_header_1_0,
                           (2, 0): np.lib.format.read_array_header_2_0}
                assert version in readers, version
                shape, fortran, dtype = readers[version](stream)
            assert not fortran and not dtype.hasobject and dtype.kind in 'bif'
            headers[name[:-4]] = dict(shape=list(shape), dtype=dtype.str,
                elements=math.prod(shape), bytes=math.prod(shape)*dtype.itemsize)
    structure = json_hash(dict(schema=metadata['schema'], kind=metadata['kind'],
        schedule=metadata['schedule'], buffers={n: {k: h[k] for k in ('shape', 'dtype')}
            for n, h in sorted(headers.items())}))
    assert structure == document['structure_key']
    by_component = defaultdict(int)
    by_dtype = defaultdict(int)
    for name, header in headers.items():
        by_component[name.split('__', 1)[0]] += header['bytes']
        by_dtype[header['dtype']] += header['bytes']
    result = dict(model=str(directory), model_sha256={n: sha(directory/n) for n in ('model.json', 'arrays.npz')},
        structure_key=structure, structure_reconstructed_from_headers=True,
        dataset=metadata.get('provenance', {}).get('dataset'), buffers=len(headers),
        runtime_buffer_bytes=sum(h['bytes'] for h in headers.values()),
        bytes_by_component=dict(by_component), bytes_by_dtype=dict(by_dtype),
        largest_buffers=[dict(name=n, **h) for n, h in sorted(headers.items(), key=lambda item: item[1]['bytes'], reverse=True)[:25]])
    binned = metadata['components'].get('binned')
    if binned is not None:
        layouts = [d for d in binned['data_layouts'] if not d['masked']]
        groups = []
        for batch in binned['program']:
            output = headers['binned__'+batch['buffers']['outputs']]
            groups.append(dict(operation=batch['operation'], depth=batch['depth'],
                operator_instances=output['shape'][0], rows=output['shape'][1],
                output_elements=output['elements'],
                index_buffer_bytes=sum(headers['binned__'+n]['bytes'] for k, n in batch['buffers'].items()
                    if k in ('children', 'outputs'))))
        result['binned'] = dict(channels=len(layouts),
            data_rows={key: sum(d[key] for d in layouts) for key in
                ('original_rows', 'positive_rows', 'retained_rows', 'packed_rows')},
            density_arena=headers['binned__initial_values'], query_arena=headers['binned__queries__initial_values'],
            dynamic_density_outputs=sum(g['output_elements'] for g in groups),
            fast_query_outputs=headers['binned__fast_outputs']['elements'],
            program_batches=len(groups), groups=groups,
            program_index_bytes=sum(g['index_buffer_bytes'] for g in groups),
            scope='Arenas count scalar density/query slots, not unique events. Program outputs include intermediate densities and integration queries. Header analysis does not attribute compiler temporaries or establish runtime liveness.')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--capture', type=Path, action='append', default=[])
    parser.add_argument('--source-revision')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError('Preserve prior evidence')
    result = inspect_model(args.model)
    captures = []
    for path in args.capture:
        capture = json.loads(path.read_text())
        assert capture['model_sha256'] == result['model_sha256']
        assert capture['float64'] and capture['mode'] == 'jit'
        assert capture['operation'] in ('nll', 'value_gradient', 'hessian')
        evidence = verified_source_record(capture, 'scripts/benchmark_tensor.py', args.source_revision)
        memory = capture['compiler_memory']
        assert all(type(v) is int and v >= 0 for v in memory.values())
        captures.append(dict(report=str(path), report_sha256=sha(path), status=capture['status'],
            backend=capture['backend'], operation=capture['operation'], source_evidence=evidence,
            compiler_memory=memory,
            guarded_allocation_bytes=sum(memory[k] for k in
                ('argument_size_in_bytes', 'output_size_in_bytes', 'temp_size_in_bytes')),
            compiler_allocation_limit_bytes=capture['compiler_allocation_limit_bytes'],
            runtime={k: capture.get(k) for k in ('host', 'devices', 'jax_version', 'jaxlib_version', 'environment')},
            measured_resources=capture['resources']))
    result.update(status='inspected_without_numerical_execution', script_sha256=sha(__file__),
        captures=captures,
        scope='Verified numeric archive headers and saved compiler estimates only. No likelihood or fit evaluated, and no observed-data fitted values read. Static compiler allocations are distinct from measured RSS/VRAM. No performance improvement or causal allocation attribution claimed.')
    save_json(args.output, result)
    print(result['status'], result['runtime_buffer_bytes'], 'bytes', result.get('binned', {}).get('program_batches'), 'density batches')


if __name__ == '__main__': main()
