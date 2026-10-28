"""Matrix-free curvature preserves native values, gradients and oracle products."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from combine_jax.tensor.curvature import make_value_gradient_hvps
from combine_jax.tensor.model import from_workspace, make_model_function
from combine_jax.root_io import open_workspace, load_caching_nll
from combine_jax.likelihood import from_workspace as old_import
from combine_jax.validation import finite_difference
from test_bin_statistics import template_path


def test_coupled_hvps_runtime_directions_fixed_entries_and_no_dense_matrix():
    def objective(x, arrays):
        active=x.at[2].set(0.)
        return jnp.sum(arrays['weights']*jnp.exp(active)) + jnp.sum(active)**4
    theta=jnp.linspace(-.2,.5,5)
    directions=jnp.array([[1.,-.3,.7,.2,-.5],[-.2,.6,1.,.4,.8]])
    function=make_value_gradient_hvps(objective);compiled=jax.jit(function)
    for scale in (1.,1.3):
        arrays=dict(weights=jnp.linspace(.7,1.1,5)*scale)
        for vectors in (directions, directions[::-1]*.8):
            value,gradient,actual=map(np.asarray,compiled(theta,arrays,vectors))
            expected=jax.hessian(objective)(theta,arrays)
            np.testing.assert_allclose(actual,np.asarray(vectors)@expected.T,atol=1e-12,rtol=1e-12)
            np.testing.assert_allclose(value,objective(theta,arrays),atol=1e-13)
            np.testing.assert_allclose(gradient,jax.grad(objective)(theta,arrays),atol=1e-13)
            np.testing.assert_array_equal(actual[:,2],0.)
    trace=jax.make_jaxpr(function)(theta,arrays,directions)
    assert not trace.consts
    assert not any(getattr(v.aval,'shape',None)==(5,5) for eq in trace.jaxpr.eqns for v in eq.outvars)
    for bad in (np.ones(5),np.ones((0,5)),np.ones((1,4)),np.ones((1,5),dtype=np.float32)):
        with pytest.raises(ValueError,match='directions'):
            compiled(theta,arrays,bad)


@pytest.mark.parametrize('kind',['counting','automcstats','binned'])
@pytest.mark.parametrize('policy',['ad','structural','structural_sequential'])
@pytest.mark.parametrize('recompute',[False,True])
def test_native_cards_and_old_hessian(counting_path,template_path,tmp_path,kind,policy,recompute):
    if kind=='counting':path,card=counting_path,Path('tests/cards/counting.txt')
    elif kind=='automcstats':path,card=template_path,Path(template_path).parent/'card.txt'
    else:
        source=Path('scripts/build_binned_fixture.py')
        spec=importlib.util.spec_from_file_location('build_binned_fixture',source)
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        path=builder.build_binned_fixture(tmp_path/'fixture');card=path.parent/'card.txt'
    root=load_caching_nll()
    with open_workspace(path) as workspace:
        model=from_workspace(workspace,card);old=old_import(workspace)
        arrays=jax.device_put(dict(model.arrays));point=old.initial
        directions=np.random.default_rng(917).normal(size=(3,len(point)))
        function=jax.jit(make_value_gradient_hvps(make_model_function(model,binned_gradient=policy,row_chunk_size=3),recompute_point=recompute))
        value,gradient,actual=map(np.asarray,function(point,arrays,directions))
        reference=root.cacheutils.CachingSimNLL(workspace.obj('ModelConfig').GetPdf(),workspace.data('data_obs'),workspace.obj('ModelConfig').GetNuisanceParameters())
        reference.setAnalyticBarlowBeeston(True);reference.clearConstantZeroPoint();reference.clearZeroPoint()
        def native(theta):
            for n,v in zip(old.parameter_names,theta,strict=True):workspace.var(n).setVal(float(v))
            return float(reference.getVal())
        np.testing.assert_allclose(value,native(point),rtol=0.,atol=1e-9)
        np.testing.assert_allclose(gradient,finite_difference(native,point,np.full(len(point),2e-4)),atol=2e-8,rtol=1e-4)
        np.testing.assert_allclose(actual,directions@old.hessian(point).T,atol=1e-9,rtol=1e-10)
        reference.setAnalyticBarlowBeeston(False)


def test_cli_records_streamed_hvps_and_runtime_direction_identity(counting_path,tmp_path):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'));old=old_import(workspace)
        point=old.initial.copy();expected=old.hessian(point)
    model.save(tmp_path/'model')
    vectors=np.random.default_rng(42).normal(size=(3,len(point)))
    path=tmp_path/'directions.npz'
    np.savez(path,directions=vectors,parameter_names=np.array(old.parameter_names))
    output=tmp_path/'result.json'
    command=[sys.executable,'scripts/benchmark_tensor.py','--model',str(tmp_path/'model'),
        '--backend','cpu','--operation','hvp','--directions',str(path),'--batch-size','1',
        '--repeat','2','--output',str(output)]
    result=subprocess.run(command,capture_output=True,text=True,timeout=120)
    assert result.returncode==0,result.stdout+result.stderr
    report=json.loads(output.read_text());assert report['status']=='measured_pending_validation'
    assert report['directions']==3 and report['points']==1 and report['directions_source_sha256']
    with np.load(report['arrays']) as saved:
        np.testing.assert_array_equal(saved['directions'],vectors)
        np.testing.assert_array_equal(saved['points'],point[None,:])
        assert saved['values'].shape==(3,) and saved['gradients'].shape==(3,len(point))
        np.testing.assert_allclose(saved['hvps'],vectors@expected.T,atol=1e-10,rtol=1e-10)
        assert 'hessians' not in saved.files


@pytest.mark.parametrize('options', [[], ['--directions','missing'], ['--directions','missing','--operation','value_gradient']])
def test_invalid_hvp_arguments_rejected_before_loading(tmp_path,options):
    result=subprocess.run([sys.executable,'scripts/benchmark_tensor.py','--model','unused',
        '--backend','cpu','--operation','hvp','--output',str(tmp_path/'none.json'),*options],
        capture_output=True,text=True,timeout=30)
    assert result.returncode==2 and not (tmp_path/'none.json').exists()
