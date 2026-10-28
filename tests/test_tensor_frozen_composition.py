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


def test_loaded_buffers_remain_owned_frozen_and_independent(tmp_path):
    original=TensorBundle(metadata(),{
        'scalar':np.array(3.), 'matrix':np.arange(12.).reshape(3,4),
        'empty':np.empty((0,2)), 'indices':np.array([1,2],dtype=np.int32),
        'mask':np.array([True,False])})
    original.save(tmp_path/'model')
    first=TensorBundle.load(tmp_path/'model');second=TensorBundle.load(tmp_path/'model')
    for name,source in original.arrays.items():
        array=first.arrays[name]
        assert array.shape==source.shape and array.dtype==source.dtype
        assert array.flags.c_contiguous and not array.flags.writeable
        np.testing.assert_array_equal(array,source)
        assert not np.shares_memory(array,second.arrays[name])
    # The NumPy read-only flag is not a security boundary. Even if a caller
    # explicitly removes it, separately loaded models and disk bytes stay intact.
    first.arrays['matrix'].flags.writeable=True
    first.arrays['matrix'][0,0]=-1
    assert second.arrays['matrix'][0,0]==0
    assert TensorBundle.load(tmp_path/'model').arrays['matrix'][0,0]==0


@pytest.mark.parametrize('bad', [np.array(['text']), np.array([1+2j])])
def test_loader_rejects_nonnumeric_archive_even_with_valid_checksum(tmp_path,bad):
    import json
    from combine_jax.tensor.bundle import file_hash
    model=TensorBundle(metadata(),{'x':np.array([1.])})
    path=tmp_path/'model';model.save(path)
    np.savez_compressed(path/'arrays.npz',x=bad)
    document=json.loads((path/'model.json').read_text())
    document['arrays_sha256']=file_hash(path/'arrays.npz')
    (path/'model.json').write_text(json.dumps(document))
    with pytest.raises(ValueError,match='Internal buffer'):
        TensorBundle.load(path)


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
