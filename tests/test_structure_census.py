"""Protect distinctions needed for the batching feasibility decision."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from structure_census import tree_fingerprints, semantic_labels, leaf_alias_fingerprint


def graph(prefix='', value=1., expression=('argument', 0), conditional=False):
    return {
        prefix+'x': ('RooRealVar', (), dict(value=value, constant=False)),
        prefix+'f': ('RooFormulaVar', (prefix+'x',), dict(ast=expression, expression='unused')),
        prefix+'h': ('FastVerticalInterpHistPdf2D2', (prefix+'f',),
                     dict(nominal=np.array([value]), coefs=(prefix+'f',), conditional=conditional)),
    }


def test_names_values_and_bin_contents_do_not_split_class_trees():
    first, second = graph(), graph('other_', 18.)
    assert tree_fingerprints(first)['h'] == tree_fingerprints(second)['other_h']
    assert (tree_fingerprints(first, semantic_labels(first))['h'] ==
            tree_fingerprints(second, semantic_labels(second))['other_h'])


@pytest.mark.parametrize('expression', [
    ('binary', '*', ('argument', 0), ('argument', 0)),
    ('binary', '+', ('argument', 0), ('constant', 2.)),
])
def test_same_class_shape_does_not_imply_same_formula(expression):
    first, second = graph(), graph(expression=expression)
    assert tree_fingerprints(first)['h'] == tree_fingerprints(second)['h']
    assert (tree_fingerprints(first, semantic_labels(first))['h'] !=
            tree_fingerprints(second, semantic_labels(second))['h'])


def test_formula_literals_are_retained_in_conservative_refinement():
    first = graph(expression=('binary', '*', ('argument', 0), ('constant', 2.)))
    second = graph(expression=('binary', '*', ('argument', 0), ('constant', 3.)))
    assert (tree_fingerprints(first, semantic_labels(first))['h'] !=
            tree_fingerprints(second, semantic_labels(second))['h'])


def test_conditional_normalization_splits_semantics_but_not_class_shape():
    first, second = graph(), graph(conditional=True)
    assert tree_fingerprints(first)['h'] == tree_fingerprints(second)['h']
    assert (tree_fingerprints(first, semantic_labels(first))['h'] !=
            tree_fingerprints(second, semantic_labels(second))['h'])


def test_aliases_are_separate_from_unfolded_tree_shape():
    shared = {'x': ('RooRealVar', ()), 'y': ('RooRealVar', ()),
              'f': ('RooProduct', ('x', 'x'))}
    separate = {**shared, 'f': ('RooProduct', ('x', 'y'))}
    assert tree_fingerprints(shared)['f'] == tree_fingerprints(separate)['f']
    assert (leaf_alias_fingerprint(shared, 'f', {'x', 'y'}) !=
            leaf_alias_fingerprint(separate, 'f', {'x', 'y'}))


def test_order_and_arity_are_preserved():
    first = {'x': ('RooRealVar', ()), 'y': ('RooConstVar', ()),
             'f': ('RooFormulaVar', ('x', 'y'))}
    reverse = {**first, 'f': ('RooFormulaVar', ('y', 'x'))}
    shorter = {**first, 'f': ('RooFormulaVar', ('x',))}
    assert len({tree_fingerprints(g)['f'] for g in (first, reverse, shorter)}) == 3


def test_cycles_and_missing_dependencies_are_rejected():
    with pytest.raises(ValueError, match='Cycle'):
        tree_fingerprints({'a': ('RooProduct', ('b',)), 'b': ('RooProduct', ('a',))})
    with pytest.raises(KeyError):
        tree_fingerprints({'a': ('RooProduct', ('missing',))})
