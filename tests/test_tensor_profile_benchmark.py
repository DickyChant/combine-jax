import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.bundle import TensorBundle
from combine_jax.tensor.fit_policy import POLICY,combine_v11_fit_context
from combine_jax.tensor.model import from_workspace


def test_profile_benchmark_captures_distinct_resident_points(counting_path,tmp_path):
    root=load_caching_nll()
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
        model.save(tmp_path/'model')
    output=tmp_path/'profile.json';grid=[-.5,0.,.5]
    command=[sys.executable,'scripts/benchmark_tensor_profile.py','--model',str(tmp_path/'model'),
        '--parameter','lumi','--values',*map(str,grid),'--backend','cpu','--cache',str(tmp_path/'cache'),
        '--output',str(output)]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1000:]+completed.stderr[-1000:]
    report=json.loads(output.read_text())
    assert report['status']=='measured_pending_validation' and report['all_converged']
    assert report['float64'] and not report['root_imported'] and report['dataset']=='data_obs'
    assert report['fit_options']['stopping_criterion']=='edm'
    assert report['evaluation_counts']['gradient_evaluations']>0
    assert report['evaluation_counts']['gradient_total']>=report['evaluation_counts']['gradient_evaluations']
    assert report['compilation_cache']['structure_key']==model.structure_key
    assert list((tmp_path/'cache').rglob('*-cache'))
    assert all(value>=0. for value in report['timings'].values())
    with np.load(report['arrays'],allow_pickle=False) as saved,open_workspace(counting_path) as workspace:
        names=list(saved['parameter_names']);index=names.index('lumi')
        np.testing.assert_array_equal(saved['grid'],grid)
        np.testing.assert_array_equal(saved['values'][:,index],grid)
        assert saved['values'].shape==(3,len(names)) and np.all(saved['status']==0)
        assert np.all(saved['edm_curvature_validated']) and np.all(saved['hessian_edm']<1e-4)
        with combine_v11_fit_context(workspace):
            config=workspace.obj('ModelConfig')
            native=root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('data_obs'),config.GetNuisanceParameters())
            native.clearConstantZeroPoint();native.clearZeroPoint()
            errors=[]
            for point,value in zip(saved['values'],saved['value'],strict=True):
                for name,x in zip(names,point,strict=True):workspace.var(str(name)).setVal(float(x))
                errors.append(abs(native.getVal()-value))
            assert max(errors)<1e-10
            del native
    print('PROFILE_BENCHMARK_NATIVE_POINT_MAX_ERROR',max(errors),flush=True)
    # Check a nonzero grid point through the same ROOT-only CLI boundary used
    # by large-model campaigns, including native constant flags and offsets.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from tensor_benchmark_common import sha
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(counting=dict(workspace=str(counting_path),
        workspace_sha256=sha(counting_path),model=str(tmp_path/'model'),configuration={},dataset='data_obs',hashes={}))))
    checked=tmp_path/'native_point.json'
    completed=subprocess.run([sys.executable,'scripts/check_tensor_fit_candidate.py','--fit',str(output),
        '--profile-index','2','--case','counting','--manifest',str(manifest),'--output',str(checked)],
        capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1000:]+completed.stderr[-1000:]
    point=json.loads(checked.read_text())
    assert point['status']=='passed_point_only' and point['profile_index']==2
    assert point['profile_point']['fixed']=={'lumi':.5}
    assert point['fit_state']['fixed_real_values']['lumi']==.5
    assert point['errors']['canonical_jax_absolute_error']<1e-10
    assert point['errors']['canonical_restoration_absolute_error']<1e-10
    assert not point['jax_imported'] and not point['torch_imported']
    assert point['covariance_status'] is None


def test_profile_entry_point_rejects_unblinded_hhvv_before_fit(counting_path,tmp_path):
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
    # A sentinel model label exercises the benchmark guard before preparation;
    # no HHVV workspace or observed fitted result is accessed.
    metadata={**model.metadata,'parameter_names':['r_hhvv',*model.metadata['parameter_names'][1:]]}
    TensorBundle._from_frozen(metadata,model.arrays).save(tmp_path/'unblinded')
    output=tmp_path/'rejected.json'
    completed=subprocess.run([sys.executable,'scripts/benchmark_tensor_profile.py','--model',str(tmp_path/'unblinded'),
        '--parameter','r_hhvv','--values','1','--backend','cpu','--output',str(output)],capture_output=True,text=True,timeout=60)
    assert completed.returncode!=0
    report=json.loads(output.read_text())
    assert report['status']=='failed' and report['stage']=='load'
    assert 'nominal pre-fit Asimov' in report['exception']
    assert not output.with_suffix('.npz').exists()


def test_default_fit_candidate_check_uses_native_fit_state_only(counting_path,tmp_path):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from tensor_benchmark_common import sha
    with open_workspace(counting_path) as workspace:
        model=from_workspace(workspace,Path('tests/cards/counting.txt'),fit_policy=POLICY)
    model.save(tmp_path/'model')
    manifest=tmp_path/'manifest.json'
    manifest.write_text(json.dumps(dict(counting=dict(workspace=str(counting_path),
        workspace_sha256=sha(counting_path),model=str(tmp_path/'model'),configuration={},dataset='data_obs',hashes={}))))
    output=tmp_path/'fit.json'
    command=[sys.executable,'scripts/benchmark_tensor.py','--model',str(tmp_path/'model'),
        '--backend','cpu','--operation','fit','--fit-stopping-criterion','edm','--single-fit',
        '--repeat','1','--output',str(output)]
    completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1000:]+completed.stderr[-1000:]
    checked=tmp_path/'checked.json'
    completed=subprocess.run([sys.executable,'scripts/check_tensor_fit_candidate.py','--fit',str(output),
        '--case','counting','--manifest',str(manifest),'--output',str(checked)],
        capture_output=True,text=True,timeout=180)
    assert completed.returncode==0,completed.stdout[-1000:]+completed.stderr[-1000:]
    point=json.loads(checked.read_text())
    assert point['status']=='passed_point_only' and point['profile_index'] is None
    assert point['profile_point'] is None and point['covariance_status']==0
    assert point['fit_state']==model.metadata['parameter_info']['fit_policy']
    assert point['native_fit_model_load']['selected_array_bytes']<sum(a.nbytes for a in model.arrays.values())
    assert point['errors']['canonical_jax_absolute_error']<1e-10
