"""Tensor construction avoids oracle execution without weakening import checks."""
from dataclasses import replace

import numpy as np
import pytest

from combine_jax.bin_statistics import build_profiles
from combine_jax.graph import UnsupportedModelError
from combine_jax.likelihood import Likelihood,from_workspace as old_import
from combine_jax.registry import Node
from combine_jax.root_io import open_workspace
from combine_jax.tensor.model import from_workspace,evaluate_model_numpy
from test_bin_statistics import template_path


def test_tensor_import_checks_lowered_program_without_eager_oracle(template_path,monkeypatch):
    with open_workspace(template_path) as workspace:
        oracle=old_import(workspace)
        expected=float(oracle._value(oracle.initial))
        def forbidden(*args,**kwargs):raise AssertionError('Eager oracle execution')
        with monkeypatch.context() as context:
            context.setattr(Likelihood,'_value',forbidden)
            with pytest.raises(AssertionError,match='Eager oracle execution'):old_import(workspace)
            model=from_workspace(workspace)
    np.testing.assert_allclose(evaluate_model_numpy(model,oracle.initial),expected,atol=1e-10,rtol=0.)


@pytest.mark.parametrize('kind',['extra_constraint','indirect_shape'])
def test_reverse_profile_dependencies_still_reject_invalid_sharing(template_path,kind):
    with open_workspace(template_path) as workspace:
        exported=old_import(workspace,analytic_barlow_beeston=False)
    nodes=dict(exported.nodes);name='prop_binch_bin0'
    # Use an indirect dependency, so a direct-server-only optimization fails.
    nodes['indirect']=Node('indirect','RooProduct',(name,),{})
    constraints=list(exported.constraints)
    if kind=='extra_constraint':
        nodes['additional']=Node('additional','RooProduct',('indirect',),{})
        constraints.append('additional');message='one matching Gaussian'
    else:
        prop=nodes['prop_binch'];attrs=dict(prop.attrs)
        attrs['coefs']=('indirect',*attrs['coefs'][1:]);nodes[prop.name]=replace(prop,attrs=attrs)
        message='also affects nominal histogram'
    with pytest.raises(UnsupportedModelError,match=message):
        build_profiles(nodes,exported.registry,exported.channels,exported.parameters,constraints)
