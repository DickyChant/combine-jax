import argparse
import json
import subprocess

import numpy as np
import pytest

from combine_jax.methods.multidim_output import write_multidim_tree
from combine_jax.root_io import load_root
from combine_jax.runtime.multidim_options import add_multidim_command, request
from combine_jax.runtime.native_save_workspace import SNAPSHOT, digest, save_workspace


DATASET = 'combine_jax_asimov_nominal'


def parse(*extra):
    parser = argparse.ArgumentParser()
    add_multidim_command(parser.add_subparsers(dest='command'))
    return parser.parse_args(['multidim', '-d', 'workspace.root', '--poi', 'r', '-t', '-1',
                              '--setParameters', 'r=1', *extra])


def test_save_workspace_option():
    assert request(parse('--algo', 'singles', '--saveWorkspace'))['save_workspace']
    assert not request(parse('--algo', 'singles'))['save_workspace']
    with pytest.raises(ValueError, match='batch-impacts'):
        request(parse('--algo', 'batch-impacts', '--parameters', 'lumi', '--saveWorkspace'))
    with pytest.raises(ValueError, match='profiled RooMultiPdf'):
        request(parse('--algo', 'singles', '--discrete-policy', 'profile', '--saveWorkspace'))


def generated(R, counting_path, tmp_path):
    """Stand-in for the native generation stage: the workspace plus a named dataset."""
    source = R.TFile.Open(str(counting_path))
    ws = source.Get('w')
    data = ws.data('data_obs').Clone(DATASET)
    getattr(ws, 'import')(data)
    path = tmp_path/'generated.root'
    ws.writeToFile(str(path))
    source.Close()
    return dict(workspace=str(path), workspace_sha256=digest(path), dataset=dict(name=DATASET))


def combine(tmp_path, name, *args):
    result = subprocess.run(['combine', '-M', 'MultiDimFit', '-n', name, '--trackParameters', 'lumi,bkg', *args],
                            cwd=tmp_path, check=True, capture_output=True, text=True)
    import uproot
    with uproot.open(tmp_path/f'higgsCombine{name}.MultiDimFit.mH120.root') as handle:
        rows = handle['limit'].arrays(['r', 'trackedParam_lumi', 'trackedParam_bkg'], library='np')
    rows['lumi'], rows['bkg'] = rows.pop('trackedParam_lumi'), rows.pop('trackedParam_bkg')
    return result.stdout, rows


def test_saved_snapshot_starts_combine_at_the_fit(counting_path, tmp_path):
    R = load_root()
    generation = generated(R, counting_path, tmp_path)
    # A best fit from Combine itself stands in for the combine-jax global fit.
    _, reference = combine(tmp_path, 'Reference', '-d', generation['workspace'], '-D', DATASET)
    fit = dict(global_values={n: float(reference[n][0]) for n in ('r', 'lumi', 'bkg')})
    output = tmp_path/'higgsCombineJax.MultiDimFit.mH120.root'
    write_multidim_tree(output, [dict(parameters=dict(r=fit['global_values']['r']), deltaNLL=0.,
                                      quantile=-1., nll0=0., nll=0.)], mass=120.)
    report = save_workspace(R, generation, fit, output)
    assert report['status'] == 'saved' and report['snapshot'] == SNAPSHOT
    assert report['fitted_parameters'] == 3 and report['dataset'] == DATASET

    handle = R.TFile.Open(str(output))
    ws = handle.Get('w')
    assert handle.Get('limit').GetEntries() == 1
    assert ws.data(DATASET) and ws.loadSnapshot(SNAPSHOT)
    for name, value in fit['global_values'].items():
        assert ws.var(name).getVal() == value
    handle.Close()

    # Combine reads the file as its own --saveWorkspace output, and refitting from
    # the snapshot stays at the fit.
    _, fresh = combine(tmp_path, 'Fresh', '-d', generation['workspace'], '-D', DATASET)
    _, warm = combine(tmp_path, 'Warm', '-d', str(output), '-w', 'w', '-D', DATASET,
                      '--snapshotName', SNAPSHOT)
    for name in ('r', 'lumi', 'bkg'):
        np.testing.assert_allclose(warm[name][0], fresh[name][0], atol=1e-3)
        np.testing.assert_allclose(warm[name][0], fit['global_values'][name], atol=1e-3)


def test_save_workspace_refuses_out_of_range_or_unknown(counting_path, tmp_path):
    R = load_root()
    generation = generated(R, counting_path, tmp_path)
    output = tmp_path/'out.root'
    write_multidim_tree(output, [dict(parameters=dict(r=1.), deltaNLL=0., quantile=-1., nll0=0., nll=0.)])
    with pytest.raises(ValueError, match='not a workspace variable'):
        save_workspace(R, generation, dict(global_values=dict(nonexistent=0.)), output)
    with pytest.raises(ValueError, match='outside the workspace range'):
        save_workspace(R, generation, dict(global_values=dict(r=1e9)), output)
    generation['workspace_sha256'] = '0'*64
    with pytest.raises(ValueError, match='changed after generation'):
        save_workspace(R, generation, dict(global_values=dict(r=1.)), output)


@pytest.mark.parametrize('save_fails', [False, True])
def test_controller_runs_native_save_after_the_fit(tmp_path, monkeypatch, save_fails):
    from combine_jax.runtime import controller
    prefix = tmp_path/'runtime'
    prefix.mkdir()
    (prefix/'runtime.json').write_text(json.dumps(dict(files={}, native_version='10.3.3',
        native_library_sha256='test', native_cmssw='/unused', lcg_view='/unused')))
    output = tmp_path/'out'/'higgsCombineTest.MultiDimFit.mH125.0.root'
    stages = []

    def stage(command, environment, log, *, rss_limit_mib):
        stages.append(log.stem)
        if log.stem == 'fit':
            (log.parent/'fit.json').write_text(json.dumps(dict(status='complete', all_fits_converged=True,
                global_values=dict(r=1.), rows=[dict(parameters=dict(r=1.), deltaNLL=0., quantile=-1.,
                                                     nll0=0., nll=0.)])))
        if log.stem == 'save_workspace':
            staged = command[-2]
            assert command[-4:] == [str(log.parent/'native/generation.json'), str(log.parent/'fit.json'),
                                    str(log.parent/output.name), str(log.parent/'save_workspace.json')]
            assert staged.endswith(output.name) and not output.exists()
            if save_fails:
                raise RuntimeError('native save failed')
            (log.parent/'save_workspace.json').write_text(json.dumps(dict(status='saved',
                snapshot=SNAPSHOT, dataset=DATASET)))
        return dict(wall_seconds=0.)

    monkeypatch.setattr(controller, 'stage', stage)
    parser = argparse.ArgumentParser()
    add_multidim_command(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['multidim', '-d', str(tmp_path/'workspace.root'), '--poi', 'r', '-t', '-1',
        '--setParameters', 'r=1', '--algo', 'singles', '--saveWorkspace', '--backend', 'cpu',
        '--runtime', str(prefix), '--workdir', str(tmp_path/'work'), '--output', str(output)])
    (tmp_path/'workspace.root').write_bytes(b'stand-in')
    if save_fails:
        with pytest.raises(RuntimeError, match='native save failed'):
            controller.run_limits(args, method='MultiDimFit')
        assert not output.exists()
        return
    controller.run_limits(args, method='MultiDimFit')
    assert stages == ['device_check', 'native_asimov', 'import', 'fit', 'save_workspace']
    receipt = json.loads(output.with_suffix('.json').read_text())
    assert receipt['saved_workspace']['snapshot'] == SNAPSHOT
    assert output.exists() and receipt['output_sha256'] == digest(output)
