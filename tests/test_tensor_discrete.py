"""One tensor executable must reproduce every frozen RooMultiPdf oracle."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from test_discrete_masks import multipdf_path
from combine_jax.likelihood import from_workspace as oracle_from_workspace
from combine_jax.root_io import open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.model import from_workspace, make_model_function, evaluate_model_numpy


@pytest.mark.parametrize('freeze_shapes', [False, True])
def test_runtime_choices_match_frozen_oracle_and_ad(multipdf_path, freeze_shapes):
    with open_workspace(multipdf_path) as workspace:
        if freeze_shapes:
            for name in ('multi_slope','multi_c1','multi_c2'):
                workspace.var(name).setConstant(True)
        model=from_workspace(workspace,discrete_policy='profile')
        assert workspace.cat('pdf_index').getIndex()==0
        oracles=[]
        for index in (0,1):
            workspace.cat('pdf_index').setIndex(index)
            oracles.append(oracle_from_workspace(workspace))
    names=model.metadata['parameter_names']
    if not freeze_shapes:
        assert {'multi_slope','multi_c1','multi_c2'}<=set(names)
    choices=model.metadata['provenance']['discrete_choices']['shapeBkg_background_choice']
    assert choices['profiled'] and len(choices['corrections'])==2
    np.testing.assert_allclose(choices['corrections'],[1.4,2.1],rtol=0,atol=1e-15)
    info=model.arrays
    initial=info['parameter_info__initial']
    widths=info['parameter_info__upper']-info['parameter_info__lower']
    scales=np.minimum(np.where(info['parameter_info__error']>0,info['parameter_info__error'],.1),.02*widths)
    rng=np.random.default_rng(9261)
    points=[initial]+[np.clip(initial+rng.uniform(-1,1,len(initial))*scales,
        info['parameter_info__lower'],info['parameter_info__upper']) for _ in range(25)]
    functions={mode:jax.jit(jax.value_and_grad(make_model_function(model,binned_gradient=mode)))
        for mode in ('ad','structural_sequential')}
    structural_gradient=jax.grad(make_model_function(model,binned_gradient='structural_sequential'))
    hvp=jax.jit(lambda point,buffers,direction:jax.jvp(
        lambda x:structural_gradient(x,buffers),(point,),(direction,))[1])
    base=dict(model.arrays)
    for index,oracle in enumerate(oracles):
        arrays={**base,'binned__discrete_indices':np.array([index],dtype=np.int32)}
        buffers=jax.device_put(arrays)
        selected=[names.index(name) for name in oracle.parameter_names]
        for point in points:
            reference_value,reference_gradient=oracle.value_and_grad(point[selected])
            expected_gradient=np.zeros(len(names));expected_gradient[selected]=reference_gradient
            for function in functions.values():
                value,gradient=function(point,buffers)
                np.testing.assert_allclose(value,reference_value,rtol=0,atol=1e-9)
                np.testing.assert_allclose(gradient,expected_gradient,rtol=1e-9,atol=1e-9)
        np.testing.assert_allclose(evaluate_model_numpy(TensorBundle(model.metadata,arrays),initial),
            oracle.value(initial[selected]),rtol=0,atol=1e-9)
        direction=jnp.asarray(rng.normal(size=len(names))*scales)
        expected_hvp=np.zeros(len(names))
        expected_hvp[selected]=np.asarray(oracle.hessian(initial[selected]))@np.asarray(direction)[selected]
        np.testing.assert_allclose(hvp(initial,buffers,direction),expected_hvp,rtol=1e-9,atol=1e-8)
    assert all(function._cache_size()==1 for function in functions.values())


def test_constant_category_preserves_frozen_import(multipdf_path):
    with open_workspace(multipdf_path) as workspace:
        workspace.cat('pdf_index').setIndex(1)
        workspace.cat('pdf_index').setConstant(True)
        frozen=from_workspace(workspace)
        profiled=from_workspace(workspace,discrete_policy='profile')
    assert profiled.metadata['parameter_names']==frozen.metadata['parameter_names']
    assert 'binned__discrete_indices' not in profiled.arrays
    for name in frozen.arrays:
        np.testing.assert_array_equal(frozen.arrays[name],profiled.arrays[name])


def test_native_boundary_records_stored_penalties_without_changing_indices(multipdf_path):
    from combine_jax.runtime.native_asimov import discrete_snapshot
    from combine_jax.runtime.import_model import verify_discrete_penalties
    with open_workspace(multipdf_path) as workspace:
        workspace.cat('pdf_index').setIndex(1)
        before=discrete_snapshot(workspace)
        assert workspace.cat('pdf_index').getIndex()==1
        item=next(iter(before.values()))
        assert item['initial_index']==1 and not item['category_constant']
        np.testing.assert_allclose([r['correction'] for r in item['alternatives']],[1.4,2.1])
        assert item['alternatives'][0]['parameters']==['multi_slope']
        assert item['alternatives'][1]['parameters']==['multi_c1','multi_c2']
        imported=from_workspace(workspace,discrete_policy='profile').metadata['provenance']['discrete_choices']
        verify_discrete_penalties(imported,before)
        broken={n:{**a,'corrections':[0.,0.]} for n,a in imported.items()}
        with pytest.raises(ValueError,match='stored discrete penalties differ'):
            verify_discrete_penalties(broken,before)
        workspace.var('multi_slope').setConstant(True)
        after=discrete_snapshot(workspace)
        assert workspace.cat('pdf_index').getIndex()==1
        selected=next(iter(after.values()))['alternatives'][0]
        assert selected['parameters']==[] and selected['correction']==item['alternatives'][0]['correction']


def test_discrete_device_fit_finds_frozen_branch_envelope(multipdf_path):
    from combine_jax.tensor.discrete_fit import make_discrete_fit_function
    from combine_jax.tensor.parameter_info import model_parameter_info,fit_coordinates,COORDINATE_FIELDS
    from combine_jax.tensor.fitting import prepare_fit
    references=[]
    with open_workspace(multipdf_path) as workspace:
        for name,error in (('multi_slope',.005),('multi_c1',.1),('multi_c2',.1)):
            workspace.var(name).setError(error)
        for index in (0,1):
            workspace.cat('pdf_index').setIndex(index)
            frozen=from_workspace(workspace)
            fn,c,a=prepare_fit(frozen,fixed={'r':1.},covariance=False,
                stopping_criterion='edm',edm_validation='estimated',edm_tolerance=1e-8)
            result=jax.device_get(fn(c,a));assert result['status']==0
            references.append(float(result['value']))
        workspace.cat('pdf_index').setIndex(int(np.argmax(references)))
        model=from_workspace(workspace,discrete_policy='profile')
    info=model_parameter_info(model)
    c=fit_coordinates(info,fixed={'r':1.})
    union=model.arrays['discrete_profile__parameter_union']
    short=fit_coordinates(info,fixed={n:float(v) for n,v,use in
        zip(info.metadata['names'],info.arrays['initial'],union) if not use})
    fn=jax.jit(make_discrete_fit_function(model,stopping_criterion='edm',edm_validation='estimated',edm_tolerance=1e-8))
    result=jax.device_get(fn({n:c[n] for n in COORDINATE_FIELDS},
        {n:short[n] for n in COORDINATE_FIELDS},jax.device_put(dict(model.arrays))))
    assert result['status']==0 and result['candidate_failures']==0
    assert result['discrete_indices'][0]==np.argmin(references)
    assert result['short_fits']>=2 and result['continuous_fits']>=2
    np.testing.assert_allclose(result['value'],min(references),rtol=0,atol=1e-6)


def test_shape_scale_floor_preserves_physical_problem():
    from combine_jax.tensor.discrete_fit import condition_shape_scales
    c=dict(reference=jnp.array([.2,3.]),free_indices=jnp.array([0,1]),
        physical_lower=jnp.array([0.,-10.]),physical_upper=jnp.array([1.,10.]),
        scale=jnp.array([1e-8,.4]),lower=jnp.array([-2e7,-32.5]),upper=jnp.array([8e7,17.5]))
    updated=condition_shape_scales(c,{'discrete_profile__parameter_union':jnp.array([True,False])})
    np.testing.assert_array_equal(updated['scale'],[.01,.4])
    for name in ('reference','physical_lower','physical_upper','free_indices'):
        np.testing.assert_array_equal(updated[name],c[name])
    for side in ('lower','upper'):
        np.testing.assert_allclose(c['reference']+updated[side]*updated['scale'],c['physical_'+side],rtol=0,atol=1e-14)
