"""Assert shared-input identity and summarize the three-stack fixed-point controls."""
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def difference(left, right):
    delta = np.asarray(right) - np.asarray(left)
    return dict(nominal=float(delta[0]), max_absolute=float(np.max(np.abs(delta))),
                max_offset_subtracted=float(np.max(np.abs(delta - delta[0]))))


def main():
    directory = Path('build/variable_bin_path_audit')
    stacks = ('v10', 'official_v11', 'pip_v11')
    reports = {}
    for stack in stacks:
        path = Path('reports') / ('variable_bin_paths_' + stack + '.json')
        shutil.copyfile(directory / (stack + '.json'), path)
        reports[stack] = json.loads(path.read_text())
        assert reports[stack]['status'] == 'completed'
        assert reports[stack]['script_sha256'] == sha('scripts/probe_variable_bin_paths.py')
    result = dict(status='completed', script_sha256=sha(__file__),
        reports={s: dict(path='reports/variable_bin_paths_' + s + '.json',
                        sha256=sha('reports/variable_bin_paths_' + s + '.json')) for s in stacks},
        scope='Fixed-point synthetic controls; no fits. The VV4q archive uses fixed background Asimov rows.',
        combine_source_files_identical=all(reports[s]['source_sha256'] ==
            reports['v10']['source_sha256'] for s in stacks), cases={}, root_changes=[])
    assert result['combine_source_files_identical']
    for label in ('plain1d', 'plain2d', 'morpher'):
        baseline = reports['v10']['cases'][label]
        for stack in stacks:
            case = reports[stack]['cases'][label]
            for key in ('workspace_sha256', 'row_sha256', 'points', 'reference_rows'):
                assert baseline[key] == case[key], (label, stack, key)
        summary = dict(workspace_sha256=baseline['workspace_sha256'],
            row_sha256=baseline['row_sha256'], points=len(baseline['points']), comparisons={},
            max_direct_roohistpdf_density_error=0., pure_roofit={}, generation={})
        for stack in stacks:
            case = reports[stack]['cases'][label]
            comparison = difference(baseline['nll'], case['nll'])
            summary['comparisons'][stack + '_minus_v10'] = comparison
            if label != 'morpher':
                assert comparison['max_absolute'] < 1e-10
            densities = case['nominal_densities']
            for name, node in densities.items():
                if node['class_name'] == 'RooHistPdf':
                    error = float(np.max(np.abs(np.array(node['values']) -
                        baseline['nominal_densities'][name]['values'])))
                    summary['max_direct_roohistpdf_density_error'] = max(
                        summary['max_direct_roohistpdf_density_error'], error)
            expected = np.array(baseline['reference_rows'])[:, -1]
            actual = np.array(case['generated_rows'])[:, -1]
            summary['generation'][stack] = dict(total=float(actual.sum()),
                reference_total=float(expected.sum()), ratio_to_fixed_v10_rows=(actual / expected).tolist())
            pure = np.array(case['pure_roofit_rows'])[:, -1]
            expected_pure = expected / expected.sum() * 100.
            error = float(np.max(np.abs(pure - expected_pure)))
            # The morpher fixture reference includes TH3F generation rounding.
            assert error < 1e-5
            summary['pure_roofit'][stack] = dict(max_bin_error_vs_normalized_fixed_rows=error)
            mixture = next(name for name, node in densities.items() if node['class_name'] == 'RooAddPdf')
            ratio = np.array(densities[mixture]['values']) / baseline['nominal_densities'][mixture]['values']
            predicted_delta = float(-np.dot(expected, np.log(ratio)))
            comparison['nominal_from_density_ratio'] = predicted_delta
            comparison['nominal_density_reconstruction_error'] = abs(predicted_delta - comparison['nominal'])
            assert comparison['nominal_density_reconstruction_error'] < 1e-10
        summary['comparisons']['pip_minus_official'] = difference(
            reports['official_v11']['cases'][label]['nll'], reports['pip_v11']['cases'][label]['nll'])
        assert summary['max_direct_roohistpdf_density_error'] == 0.
        result['cases'][label] = summary
    assert result['cases']['morpher']['comparisons']['official_v11_minus_v10']['nominal'] > 20.

    archive_names = ('v10', 'v11', 'v11_official')
    archive = {s: json.loads(Path('reports/asymptotic_vv4q_noVH_MX4000_' +
                                  s + '_shared_nll.json').read_text()) for s in archive_names}
    for s in archive_names:
        for key in ('workspace_sha256', 'data_sha256', 'points_sha256', 'runtime_flags'):
            assert archive[s][key] == archive['v10'][key]
    nll = {s: [c['nll'] for c in archive[s]['cases']] for s in archive_names}
    result['vv4q_shared_input_recheck'] = dict(
        workspace_sha256=archive['v10']['workspace_sha256'],
        data_sha256=archive['v10']['data_sha256'], points_sha256=archive['v10']['points_sha256'],
        official_minus_v10=difference(nll['v10'], nll['v11_official']),
        official_minus_pip=difference(nll['v11'], nll['v11_official']))
    for commit in ('16b33f63e4f3d79df980be764e571f3b0a5817c5',
                   '2f06c041f99faa31c760b149b236b31e2d740910'):
        source = directory / (commit + '.json')
        api = json.loads(source.read_text())
        result['root_changes'].append(dict(commit=commit, url=api['html_url'],
            api_response_sha256=sha(source), title=api['commit']['message'].splitlines()[0],
            committed_utc=api['commit']['committer']['date'],
            changed_files=[f['filename'] for f in api['files']]))
    output = Path('reports/variable_bin_paths_comparison.json')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v['comparisons'] for k, v in result['cases'].items()}, indent=2))


if __name__ == '__main__':
    main()
