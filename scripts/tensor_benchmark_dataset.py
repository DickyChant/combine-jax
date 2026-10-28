"""Require documented pre-fit Asimov input for every new HHVV benchmark fit."""
import json
from pathlib import Path

from tensor_benchmark_common import sha


ASIMOV_KIND = 'combine_prefit_asimov'


def require_blind_model(model, *, hhvv=False):
    """ROOT-free guard for standalone tensor benchmarks; generic cards unchanged."""
    if not hhvv and 'r_hhvv' not in model.metadata['parameter_names']:
        return
    provenance = model.metadata.get('provenance', {})
    dataset = provenance.get('dataset', {})
    generation = provenance.get('dataset_generation', {})
    if dataset.get('name') != 'asimov_prefit' or generation.get('kind') != ASIMOV_KIND:
        raise ValueError('HHVV benchmarks require the nominal pre-fit Asimov dataset')
    if (generation.get('MH') != 125. or not dataset.get('identity')
            or generation.get('dataset_identity') != dataset['identity']):
        raise ValueError('HHVV Asimov generation and imported data identity differ')
    if not generation.get('generation_report_sha256'):
        raise ValueError('HHVV Asimov generation evidence is missing')


def benchmark_dataset(case, model, *, label=None):
    name = case.get('dataset', 'data_obs')
    is_hhvv = label == 'hhvv' or 'r_hhvv' in model.metadata['parameter_names']
    if is_hhvv:
        require_blind_model(model, hhvv=True)
        if name != 'asimov_prefit':
            raise ValueError('Observed-data HHVV benchmark execution is prohibited')
        generation = model.metadata['provenance']['dataset_generation']
        path = Path(case['generation_report'])
        if sha(path) != generation['generation_report_sha256']:
            raise ValueError('Asimov generation report hash differs')
        report = json.loads(path.read_text())
        if report['status'] != 'generated_and_identity_verified' or report['generation_kind'] != ASIMOV_KIND:
            raise ValueError('Asimov generation was not independently verified')
        if report['dataset']['identity'] != generation['dataset_identity']:
            raise ValueError('Asimov dataset identity differs')
        if sha(case['workspace']) != report['workspace_sha256']:
            raise ValueError('Asimov workspace differs from the generation artifact')
    imported = model.metadata.get('provenance', {}).get('dataset')
    if imported and imported['name'] != name:
        raise ValueError('Native and tensor dataset names differ')
    return name
