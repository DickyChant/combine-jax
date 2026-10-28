"""Classify scalar dependencies at yield, shape and constraint consumer boundaries.

Reads neutral arrays only. Parameter names and dependency indices are reported;
no parameter values or fit results are read. This is a conservative structural
census, not a claim that all yield terms admit the same likelihood reduction.
"""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from combine_jax.tensor.bundle import json_hash
from tensor_benchmark_common import save_json, sha
from tensor_benchmark_dataset import require_blind_model


def scalar_dependencies(metadata, arrays):
    """Exact graph reachability, including aliases and repeated/shared inputs."""
    dependencies = defaultdict(int)
    for slot, parameter in zip(arrays['parameter_slots'], arrays['parameter_indices'], strict=True):
        dependencies[int(slot)] = 1 << int(parameter)
    for batch in metadata['schedule']:
        for output, inputs, mask in zip(arrays[batch['outputs']], arrays[batch['inputs']],
                                       arrays[batch['mask']], strict=True):
            bits = 0
            for slot in inputs[mask]:
                bits |= dependencies[int(slot)]
            dependencies[int(output)] = bits
    return [dependencies[int(slot)] for slot in arrays['output_slots']]


class Prefix:
    def __init__(self, arrays, prefix):
        self.arrays, self.prefix = arrays, prefix

    def __getitem__(self, name):
        return self.arrays[self.prefix + name]


def classify(metadata, arrays):
    components = metadata['components']
    dependencies = scalar_dependencies(components['scalars'], Prefix(arrays, 'scalars__'))
    roles = defaultdict(int)
    consumer_scalars = defaultdict(set)

    def add(role, indices, mask=None):
        indices = np.asarray(indices)
        if mask is not None:
            indices = indices[np.broadcast_to(mask, indices.shape)]
        for index in np.unique(indices):
            index = int(index)
            if not 0 <= index < len(dependencies):
                raise ValueError('Consumer scalar index out of range')
            roles[role] |= dependencies[index]
            consumer_scalars[role].add(index)

    ch = Prefix(arrays, 'channels__')
    add('yield', ch['rate_indices'], ch['rate_mask'])
    if 'templates' in components:
        t = Prefix(arrays, 'templates__')
        add('yield', t['coefficient_indices'], t['process_mask'])
        add('yield', t['outer_indices'], t['propagator_mode'])
        for batch in components['templates']['schedule'][:-1]:
            if batch['operation'] != 'cms_histfunc':
                raise ValueError('Unknown template morph consumer')
            add('shape', t[batch['inputs']], t[batch['buffers']['morph_mask']])
        indices, kinds = t['bin_parameter_indices'], t['bin_types']
        add('shape', indices[(kinds > 0) & (indices < len(dependencies))])

    top_mixtures = []
    if 'binned' in components:
        m, a = components['binned'], Prefix(arrays, 'binned__')
        # NLL consumers refer to complete output rows of a top-level mixture.
        # Verify the full index row and the ordered, masked rate bindings.
        roots = {}
        for ni, batch in enumerate(m['nll_program']):
            b = {k: a[n] for k, n in batch['buffers'].items() if k != 'weights'}
            add('yield', b['rates'], b['rate_mask'])
            for row, density in enumerate(b['densities']):
                key = (int(density[0]), int(density[-1]), len(density))
                if key in roots:
                    raise ValueError('Repeated NLL density row')
                roots[key] = (ni, row, density, b['rates'][row][b['rate_mask'][row]])
        matched = set()
        for pi, batch in enumerate(m['program']):
            fields = batch['buffers']
            if 'coefs' not in fields:
                continue
            coefs, mask = a[fields['coefs']], a[fields['coef_mask']]
            shape_mask = mask.copy()
            if batch['operation'] == 'mixture_yields':
                for row, output in enumerate(a[fields['outputs']]):
                    key = (int(output[0]), int(output[-1]), len(output))
                    if key not in roots:
                        continue
                    ni, nr, density, rates = roots[key]
                    if not np.array_equal(output, density) or not np.array_equal(coefs[row][mask[row]], rates):
                        raise ValueError('Top mixture does not match NLL bindings')
                    matched.add(key)
                    shape_mask[row] = False
                    top_mixtures.append(dict(program_batch=pi, program_row=row,
                        nll_batch=ni, nll_row=nr, processes=len(rates), rows=len(output)))
            add('shape', coefs, shape_mask)
        if set(roots) != matched:
            raise ValueError('Cannot certify every dynamic NLL root as a yield mixture')
        for batch in m['histograms']['schedule']:
            if batch['operation'] != 'normalized_fast_histograms':
                raise ValueError('Unknown fast histogram consumer')
            b = batch['buffers']
            add('shape', a['histograms__' + b['inputs']], a['histograms__' + b['morph_mask']])

    c = Prefix(arrays, 'constraints__')
    for batch in components['constraints']['schedule']:
        if batch['inputs'] is not None:
            indices = c[batch['inputs']]
            add('constraint', indices[indices < len(dependencies)])
    fixed = arrays['scalars__parameter_fixed']
    parameters = []
    for index, name in enumerate(metadata['parameter_names']):
        active = [role for role in ('yield', 'shape', 'constraint') if roles[role] & (1 << index)]
        category = ('fixed' if fixed[index] else 'yield_and_shape' if 'yield' in active and 'shape' in active
                    else 'shape_only' if 'shape' in active else 'yield_only' if 'yield' in active
                    else 'constraint_only' if 'constraint' in active else 'unused')
        parameters.append(dict(index=index, name=name, category=category, consumers=active))
    return dict(counts=dict(Counter(p['category'] for p in parameters)), parameters=parameters,
        scalar_consumers={role: sorted(indices) for role, indices in consumer_scalars.items()},
        certified_top_mixtures=top_mixtures,
        scope='Conservative reachability through scalar instructions into process yields, shape inputs and constraints. Fixed imports have no active dependency. Nested mixture fractions and parametric-bin coefficients count as shapes. Template yield coefficients still couple to statistical profiles; they are not certified for the simple unbinned score formula.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier census evidence')
    document = json.loads((args.model / 'model.json').read_text())
    metadata = document['metadata']
    assert json_hash(metadata) == document['metadata_sha256']
    require_blind_model(SimpleNamespace(metadata=metadata))
    assert sha(args.model / 'arrays.npz') == document['arrays_sha256']
    with np.load(args.model / 'arrays.npz', allow_pickle=False) as arrays:
        result = classify(metadata, arrays)
    result.update(status='structural_dependencies_classified', script_sha256=sha(__file__),
        model_sha256={n: sha(args.model / n) for n in ('model.json', 'arrays.npz')},
        structure_key=document['structure_key'], dataset=metadata.get('provenance', {}).get('dataset'))
    save_json(args.output, result)
    print(json.dumps(result['counts']))


if __name__ == '__main__':
    main()
