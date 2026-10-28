"""Reachability must keep shared rate/shape paths and ignore masked padding."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from inspect_tensor_gradient_roles import classify, scalar_dependencies


def fixture():
    names = ['rate', 'shape', 'shared', 'constraint', 'fixed', 'unused']
    scalar = dict(schedule=[dict(inputs='i', mask='m', outputs='o')])
    arrays = dict(parameter_slots=np.array([0, 1, 2, 3, 5]),
        parameter_indices=np.array([0, 1, 2, 3, 5]), output_slots=np.array([0, 1, 2, 3, 4, 5, 6, 6]),
        parameter_fixed=np.array([False, False, False, False, True, False]),
        i=np.array([[0, 2, 5]]), m=np.array([[True, True, False]]), o=np.array([6]))
    metadata = dict(parameter_names=names, components=dict(scalars=scalar,
        channels={}, constraints=dict(schedule=[dict(inputs='i')]),
        binned=dict(program=[dict(operation='mixture_yields', buffers=dict(coefs='c', coef_mask='cm', outputs='o')),
            dict(operation='coef_sum', buffers=dict(coefs='s', coef_mask='sm', outputs='so'))],
            nll_program=[dict(buffers=dict(densities='d', rates='r', rate_mask='rm'))],
            histograms=dict(schedule=[]))))
    model_arrays = {'scalars__' + k: v for k, v in arrays.items()}
    model_arrays.update(channels__rate_indices=np.array([[5]]), channels__rate_mask=np.array([[False]]),
        constraints__i=np.array([[3, 4]]), binned__c=np.array([[7]]), binned__cm=np.array([[True]]),
        binned__o=np.array([[10, 11]]), binned__s=np.array([[1, 2, 5]]),
        binned__sm=np.array([[True, True, False]]), binned__so=np.array([[8, 9]]),
        binned__d=np.array([[10, 11]]), binned__r=np.array([[7]]), binned__rm=np.array([[True]]))
    return metadata, model_arrays, scalar, arrays


def test_shared_paths_aliases_fixed_and_padding():
    metadata, arrays, scalar, scalar_arrays = fixture()
    assert scalar_dependencies(scalar, scalar_arrays) == [1, 2, 4, 8, 0, 32, 5, 5]
    result = classify(metadata, arrays)
    assert {p['name']: p['category'] for p in result['parameters']} == dict(
        rate='yield_only', shape='shape_only', shared='yield_and_shape',
        constraint='constraint_only', fixed='fixed', unused='unused')
    assert len(result['certified_top_mixtures']) == 1


@pytest.mark.parametrize('key,value', [('binned__d', [[10, 12]]), ('binned__r', [[0]])])
def test_uncertified_root_or_reordered_rate_is_rejected(key, value):
    metadata, arrays, _, _ = fixture()
    arrays[key] = np.array(value)
    with pytest.raises(ValueError, match='NLL'):
        classify(metadata, arrays)


def test_nested_mixture_coefficients_are_shapes():
    metadata, arrays, _, _ = fixture()
    m = metadata['components']['binned']
    m['program'].append(dict(operation='mixture_fractions', buffers=dict(coefs='nested', coef_mask='nested_mask')))
    arrays.update(binned__nested=np.array([[0]]), binned__nested_mask=np.array([[True]]))
    result = classify(metadata, arrays)
    assert result['parameters'][0]['category'] == 'yield_and_shape'
