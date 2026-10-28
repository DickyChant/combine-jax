import numpy as np
import pytest

from combine_jax.tensor.bundle import TensorBundle


def metadata():
    return dict(schema='combine-jax.tensor.v1',kind='test',schedule=[],nested={'value':1})


def test_public_input_copy_internal_sharing_and_transport(tmp_path):
    raw=np.arange(16,dtype=np.float64)
    original=TensorBundle(metadata(),{'x':raw})
    raw[0]=-1
    assert original.arrays['x'][0]==0. and raw.flags.writeable
    composite=TensorBundle._from_frozen(original.metadata,original.arrays)
    assert composite.arrays['x'] is original.arrays['x']
    composite.metadata['nested']['value']=2
    assert original.metadata['nested']['value']==1
    with pytest.raises(ValueError):composite.arrays['x'][0]=1.
    composite.save(tmp_path/'shared');loaded=TensorBundle.load(tmp_path/'shared')
    assert composite.metadata==loaded.metadata and composite.structure_key==loaded.structure_key
    assert composite.arrays['x'].tobytes()==loaded.arrays['x'].tobytes()


def test_internal_factory_rejects_unfrozen_and_noncontiguous_buffers():
    raw=np.arange(10,dtype=np.float64)
    with pytest.raises(ValueError,match='frozen'):TensorBundle._from_frozen(metadata(),{'x':raw})
    raw.flags.writeable=False
    with pytest.raises(ValueError,match='contiguous'):TensorBundle._from_frozen(metadata(),{'x':raw[::2]})


def test_composition_shares_components_but_copies_external_offset(counting_path):
    from pathlib import Path
    from combine_jax.root_io import load_caching_nll,open_workspace
    from combine_jax.tensor.model import from_workspace,unpack_counting,compose_counting
    load_caching_nll()
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path(__file__).parent/'cards/counting.txt')
    parts=unpack_counting(model);offset=np.array(3.)
    composed=compose_counting(**parts,offset=offset)
    assert offset.flags.writeable
    offset[...]=4.
    assert composed.arrays['offset']==3.
    for name,part in parts.items():
        for key,array in part.arrays.items():
            assert composed.arrays[name+'__'+key] is array
