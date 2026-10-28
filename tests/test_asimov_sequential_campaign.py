"""Blinding, stopping and conditional-curvature gates for authorized GPU fits."""
import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from run_asimov_sequential_fit import checked_reference, converged, fit_arguments
from tensor_benchmark_common import sha
from submit_asimov_sequential_fits import submit_description
from stage_asimov_sequential_fit import required_inputs, check_staged_references


@pytest.mark.parametrize('family', ['A100', 'H100'])
def test_same_node_and_gpu_only_submission(family):
    text = submit_description(Path('scripts/condor_tensor_edm.sub').read_text(), family, 'worker.cern.ch')
    assert f'regexp("{family}", GPUs_DeviceName)' in text
    assert 'TARGET.Machine == "worker.cern.ch"' in text
    assert 'request_gpus = 1' in text and '!regexp("MIG", GPUs_DeviceName)' in text
    assert '+JobFlavour = "tomorrow"' in text and 'request_cpus = 8' in text
    with pytest.raises(AssertionError):
        submit_description(Path('scripts/condor_tensor_edm.sub').read_text(), family, 'lxplus905.cern.ch')


def test_fit_is_single_gradient_only_edm_with_unchanged_guard():
    args = fit_arguments('model', 'result.json')
    pairs = {args[i]: args[i + 1] for i in range(len(args) - 1) if args[i].startswith('--')}
    assert pairs['--backend'] == 'gpu'
    assert pairs['--binned-gradient'] == 'structural_sequential'
    assert pairs['--fit-covariance'] == 'none' and pairs['--fit-edm-validation'] == 'estimated'
    assert pairs['--fit-stopping-criterion'] == 'edm' and float(pairs['--fit-edm-tolerance']) == 1e-4
    assert pairs['--compiler-memory-limit-gib'] == '8' and '--single-fit' in args


@pytest.mark.parametrize('edm,status,expected', [(9e-5, 0, True), (1e-4, 0, False),
    (float('nan'), 0, False), (-1., 0, False), (9e-5, 1, False)])
def test_covariance_requires_actual_optimizer_convergence(edm, status, expected):
    report = dict(status='measured_pending_validation', fit_result=dict(status=status,
        covariance_status=6, corrected_estimated_edm=edm))
    assert converged(report) is expected
    report['status'] = 'failed'
    assert not converged(report)


def test_saved_asimov_reference_and_corruption_rejection():
    manifest = Path('build/tensor_hhvv_asimov/manifest.json')
    if not manifest.exists():
        pytest.skip('HHVV Asimov inputs unavailable')
    case = json.loads(manifest.read_text())['hhvv']
    metadata = json.loads((Path(case['model']) / 'model.json').read_text())['metadata']
    native = json.loads(Path('reports/gpu_tensor_asimov_h100_completed_native_fit.json').read_text())
    assert checked_reference(case, metadata, native, manifest_sha256=sha(manifest))['minimum_comparison_eligible']
    observed = copy.deepcopy(case)
    observed['dataset'] = 'data_obs'
    with pytest.raises((ValueError, AssertionError)):
        checked_reference(observed, metadata, native, manifest_sha256=sha(manifest))
    changed = copy.deepcopy(native)
    changed['stopping_rule']['tolerance'] = 1.
    with pytest.raises(AssertionError):
        checked_reference(case, metadata, changed, manifest_sha256=sha(manifest))
    changed = copy.deepcopy(native)
    changed['fit_state']['runtime_flags']['OPTIMIZE_BOUNDS'] = 0
    with pytest.raises(AssertionError):
        checked_reference(case, metadata, changed, manifest_sha256=sha(manifest))


def test_payload_guard_runs_without_original_cwd(tmp_path):
    import os
    manifest = Path('build/tensor_hhvv_asimov/manifest.json')
    if not manifest.exists():
        pytest.skip('HHVV Asimov inputs unavailable')
    case = json.loads(manifest.read_text())['hhvv']
    native = {family: f'reports/gpu_tensor_asimov_{family.lower()}_completed_native_fit.json'
              for family in ('A100','H100')}
    inputs = required_inputs(manifest, case, native)
    assert case['workspace'] in inputs
    for name in inputs:
        dest = tmp_path/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        os.link(name,dest)
    for folder in ('src','scripts'):
        (tmp_path/folder).symlink_to(Path(folder).resolve(),target_is_directory=True)
    runtime = dict(manifest=str(manifest),native_references=native,input_files={p:sha(p) for p in inputs})
    (tmp_path/'build/sequential_fit_runtime.json').write_text(json.dumps(runtime))
    assert check_staged_references(tmp_path,runtime)['input_hashes_verified']
    (tmp_path/case['workspace']).unlink()
    with pytest.raises(RuntimeError,match='workspace.root'):
        check_staged_references(tmp_path,runtime)
