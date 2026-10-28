"""Named dataset selection and fail-closed HHVV benchmark provenance."""
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from tensor_benchmark_common import sha
from tensor_benchmark_dataset import benchmark_dataset, require_blind_model
from combine_jax.dataset_identity import binding_identity, workspace_dataset_identity
from combine_jax.root_io import load_caching_nll, open_workspace
from combine_jax.tensor.model import from_workspace, evaluate_model_numpy
from combine_jax.tensor.parameter_info import model_parameter_info


def test_named_data_import_and_native_evaluation(counting_path):
    root = load_caching_nll()
    with open_workspace(counting_path) as workspace:
        original = workspace.data('data_obs')
        alternate = original.emptyClone('test_asimov')
        for i in range(original.numEntries()):
            row = original.get(i).snapshot()
            # Synthetic counting card only: changing event weights changes its
            # observed count while keeping all observable coordinates unchanged.
            alternate.add(row, 1.5*original.weight())
        assert not getattr(workspace, 'import')(alternate)
        card = Path(__file__).parent/'cards/counting.txt'
        default = from_workspace(workspace, card)
        model = from_workspace(workspace, card, data='test_asimov', fit_policy='combine_v11_default')
        assert model.metadata['provenance']['dataset']['identity'] == workspace_dataset_identity(workspace, 'test_asimov')
        assert model.metadata['provenance']['dataset']['identity'] != default.metadata['provenance']['dataset']['identity']
        info = model_parameter_info(model)
        config = workspace.obj('ModelConfig')
        native = root.cacheutils.CachingSimNLL(config.GetPdf(),workspace.data('test_asimov'),config.GetNuisanceParameters())
        native.clearConstantZeroPoint(); native.clearZeroPoint()
        errors = []
        for signal in (.4,1.,1.7):
            point = info.arrays['initial'].copy()
            point[info.metadata['names'].index('r')] = signal
            for name, value in zip(info.metadata['names'],point,strict=True):
                workspace.var(name).setVal(float(value))
            errors.append(abs(evaluate_model_numpy(model,point)-native.getVal()))
        assert max(errors)<1e-10
        print('NAMED_DATA_NATIVE_NLL_MAX_ERROR', max(errors), flush=True)
        del native


def test_identity_includes_zero_rows_and_coordinates():
    binding = SimpleNamespace(coordinates={'x':np.array([1.,2.])},weights=np.array([2.,0.]))
    original = binding_identity({'channel':binding})
    binding.coordinates['x'][1] = 3.
    assert original != binding_identity({'channel':binding})


@pytest.fixture
def asimov_case(tmp_path):
    workspace = tmp_path/'workspace.root'; workspace.write_bytes(b'synthetic test workspace')
    identity = dict(sha256='synthetic data digest',channels=1,rows=3)
    report = tmp_path/'generation.json'
    report.write_text(json.dumps(dict(status='generated_and_identity_verified',
        generation_kind='combine_prefit_asimov',dataset=dict(identity=identity),workspace_sha256=sha(workspace))))
    model = SimpleNamespace(metadata=dict(parameter_names=['r_hhvv'],provenance=dict(
        dataset=dict(name='asimov_prefit',identity=identity),
        dataset_generation=dict(kind='combine_prefit_asimov',MH=125.,dataset_identity=identity,
            generation_report_sha256=sha(report)))))
    case = dict(dataset='asimov_prefit',workspace=str(workspace),generation_report=str(report))
    return case,model


def test_verified_asimov_and_generic_defaults(asimov_case):
    assert benchmark_dataset(*asimov_case,label='hhvv') == 'asimov_prefit'
    generic=SimpleNamespace(metadata=dict(parameter_names=['r']))
    assert benchmark_dataset({},generic) == 'data_obs'
    require_blind_model(generic)
    with pytest.raises(ValueError,match='pre-fit Asimov'):
        benchmark_dataset({},generic,label='hhvv')


@pytest.mark.parametrize('change', ['observed','missing','identity','mass','generation','workspace'])
def test_rejects_unbound_or_observed_hhvv(asimov_case,change):
    case,model = copy.deepcopy(asimov_case)
    if change=='observed': case['dataset']='data_obs'
    if change=='missing': model.metadata['provenance'].pop('dataset_generation')
    if change=='identity': model.metadata['provenance']['dataset_generation']['dataset_identity']={}
    if change=='mass': model.metadata['provenance']['dataset_generation']['MH']=120.
    if change=='generation': Path(case['generation_report']).write_text('{}')
    if change=='workspace': Path(case['workspace']).write_bytes(b'different workspace')
    with pytest.raises(ValueError): benchmark_dataset(case,model,label='hhvv')
