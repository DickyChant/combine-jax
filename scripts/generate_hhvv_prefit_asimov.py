"""Generate Combine's nominal pre-fit HHVV Asimov without fitting observed data."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from tensor_benchmark_common import save_json, sha


def nominal_parameters(workspace):
    config = workspace.obj('ModelConfig')
    return {str(v.GetName()): float(v.getVal()) for collection in
        (config.GetParametersOfInterest(), config.GetNuisanceParameters(), config.GetGlobalObservables())
        for v in collection if v.InheritsFrom('RooAbsReal')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    assert not any('/lcg/views/' in p for p in sys.path)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    from combine_jax.root_io import load_caching_nll, open_workspace
    from combine_jax.dataset_identity import workspace_dataset_identity
    from hhvv_setup import configure_hhvv
    root = load_caching_nll()
    assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
    args.directory.mkdir(parents=True)
    prepared = args.directory / 'nominal_input.root'
    with open_workspace(args.workspace) as workspace:
        configure_hhvv(workspace)
        workspace.var('MH').setVal(125.)
        workspace.var('MH').setConstant(True)
        nominal = nominal_parameters(workspace)
        assert 'r_hhvv' in nominal
        pois = {str(v.GetName()): float(v.getVal()) for v in workspace.obj('ModelConfig').GetParametersOfInterest()}
        workspace.writeToFile(str(prepared))
    # Explicit nonstandard POI values select the signal ModelConfig. Without
    # --setParameters, Combine can choose ModelConfig_bonly even at r_hhvv=1.
    command = [sys.executable, str(Path(sys.prefix) / 'bin/combine'), '-M', 'GenerateOnly',
        '-d', str(prepared.resolve()), '-t', '-1', '--mass', '125', '--saveToys', '--saveWorkspace',
        '--setParameters', ','.join(name + '=' + format(value, '.17g') for name, value in sorted(pois.items())),
        '--freezeParameters', 'MX', '-n', '.prefit_asimov', '-v', '1']
    assert '--toysFrequentist' not in command and '--snapshotName' not in command
    report = dict(status='generating', script_sha256=sha(__file__), command=command,
        source_workspace=str(args.workspace), source_workspace_sha256=sha(args.workspace),
        root_version=root.gROOT.GetVersion(), combine_version=version('higgs-combine-tool'),
        generation_kind='combine_prefit_asimov', nominal_parameters=nominal, nominal_pois=pois,
        MH=125., configuration={'MX': {'value': 2000., 'constant': True}},
        scope='Workspace nominal nuisance/POI/global values; no fitted snapshot, no frequentist prefit, no observed likelihood evaluation. GenerateOnly -t -1 uses Combine standard generation. The named Asimov dataset is copied unchanged into a new workspace.')
    save_json(args.output, report)
    log = args.directory / 'generate.log'
    child = None
    started = time.monotonic()
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=args.directory, stdout=stream, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            while child.poll() is None:
                if time.monotonic() - started > 1800:
                    raise RuntimeError('Asimov generation exceeded 30 minutes')
                ps = subprocess.run(['ps', '--no-headers', '-g', str(child.pid), '-o', 'rss='],
                                    capture_output=True, text=True)
                if ps.returncode not in (0, 1):
                    raise RuntimeError('Cannot inspect Asimov generator resource use')
                rss = sum(float(v) for v in ps.stdout.split()) / 1024
                combined = rss + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
                report['combined_peak_rss_mib'] = max(combined, report.get('combined_peak_rss_mib', 0.))
                if combined > 11264:
                    raise RuntimeError('Asimov generation exceeded the combined 11 GiB RSS guard')
                time.sleep(.25)
        report.update(returncode=child.returncode, generation_wall_seconds=time.monotonic() - started,
                      log_sha256=sha(log))
        assert child.returncode == 0
        text = log.read_text()
        assert 'Doing initial fit:' not in text and 'Minuit2Minimizer : Valid minimum' not in text
        outputs = list(args.directory.glob('higgsCombine*.root'))
        assert len(outputs) == 1
        with open_workspace(outputs[0]) as generated:
            assert generated.getSnapshot('clean')
            generated.loadSnapshot('clean')
            assert nominal_parameters(generated) == nominal
            assert generated.var('MH').getVal() == 125.
        source = root.TFile.Open(str(outputs[0]))
        asimov = source.Get('toys/toy_asimov')
        assert asimov and asimov.InheritsFrom('RooAbsData')
        target = args.directory / 'workspace_asimov.root'
        with open_workspace(prepared) as workspace:
            before = workspace_dataset_identity(workspace, asimov)
            assert not workspace.data('asimov_prefit')
            assert not getattr(workspace, 'import')(asimov, root.RooFit.Rename('asimov_prefit'))
            assert workspace_dataset_identity(workspace, 'asimov_prefit') == before
            workspace.writeToFile(str(target))
        source.Close()
        with open_workspace(target) as workspace:
            assert nominal_parameters(workspace) == nominal
            assert workspace_dataset_identity(workspace, 'asimov_prefit') == before
            data = workspace.data('asimov_prefit')
            report['dataset'] = dict(name='asimov_prefit', kind='combine_prefit_asimov',
                entries=int(data.numEntries()), sum_weights=float(data.sumEntries()), identity=before)
        report.update(status='generated_and_identity_verified', workspace=str(target),
            workspace_sha256=sha(target), native_toy_file=str(outputs[0]), native_toy_sha256=sha(outputs[0]),
            prepared_workspace_sha256=sha(prepared))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try: child.wait(timeout=10)
            except subprocess.TimeoutExpired: os.killpg(child.pid, signal.SIGKILL); child.wait()
        save_json(args.output, report)
    print(report['status'], report['dataset']['entries'], report['dataset']['identity']['sha256'], flush=True)


if __name__ == '__main__':
    main()
