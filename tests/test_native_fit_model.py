import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from native_fit_model import load_native_fit_model
from combine_jax.root_io import open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.fit_policy import POLICY
from combine_jax.tensor.model import from_workspace
from combine_jax.tensor.parameter_info import model_parameter_info
from combine_jax.tensor.profile_point import configured_profile_model


def test_native_fit_view_never_decompresses_likelihood_arrays(counting_path,tmp_path,monkeypatch):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
    model.save(tmp_path/'model')
    requested=[]
    original=np.lib.npyio.NpzFile.__getitem__
    def checked(archive,name):
        requested.append(name)
        assert name.startswith('parameter_info__') or name in {
            'scalars__parameter_initial','scalars__parameter_fixed','scalars__parameter_indices',
            'scalars__parameter_slots','scalars__initial_values'}
        return original(archive,name)
    monkeypatch.setattr(np.lib.npyio.NpzFile,'__getitem__',checked)
    view,evidence=load_native_fit_model(tmp_path/'model')
    assert requested and len(requested)<evidence['archive_array_count']
    assert evidence['selected_array_bytes']==sum(a.nbytes for a in view.arrays.values())
    expected=model_parameter_info(model);actual=model_parameter_info(view)
    assert expected.metadata==actual.metadata
    for name in expected.arrays:np.testing.assert_array_equal(expected.arrays[name],actual.arrays[name])
    profiled=configured_profile_model(view,{'lumi':.5})
    assert profiled.metadata['profile_point']['fixed']=={'lumi':.5}
    assert not isinstance(view,TensorBundle)


@pytest.mark.parametrize('target',['metadata','arrays'])
def test_native_fit_view_rejects_changed_input(counting_path,tmp_path,target):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
    model.save(tmp_path/'model')
    if target=='arrays':
        with (tmp_path/'model/arrays.npz').open('ab') as stream:stream.write(b'changed')
    else:
        path=tmp_path/'model/model.json';doc=json.loads(path.read_text());doc['metadata']['kind']='changed'
        path.write_text(json.dumps(doc))
    with pytest.raises(ValueError,match='checksum'):load_native_fit_model(tmp_path/'model')
