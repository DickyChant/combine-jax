"""Native v10 Migrad on the explicit HHVV nominal Asimov, Python 3.9 compatible.

The fit configuration and dataset are exported with the corrected neutral model.
No observed dataset is selected and no fitted snapshot supplies the start.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import shlex
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--configuration', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--tensor-fit', type=Path, help='Optional completed Asimov GPU fit NPZ for a same-point NLL check')
    args = parser.parse_args()
    assert not args.output.exists() and not args.directory.exists()
    assert platform.node().split('.')[0] != 'lxplus905'
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    import ROOT as R
    import numpy as np
    base = Path(os.environ['CMSSW_BASE'])
    assert base.name == 'CMSSW_14_1_0_pre4' and R.gROOT.GetVersion() == '6.30/07'
    library = base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
    R.gInterpreter.AddIncludePath(str(base/'src'))
    assert R.gSystem.Load(str(library)) >= 0
    R.gROOT.SetBatch(True)
    assert R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    config = json.loads(args.configuration.read_text())
    for field in ('lower', 'upper', 'profile_lower', 'profile_upper'):
        config[field] = [float(v) for v in config[field]]
    assert config['dataset'] == 'asimov_prefit' and config['template_convention'] == 'v10_density'
    assert sha(args.workspace) == config['workspace_sha256']
    assert sha(library) == config['native_library_sha256']
    names = config['names']
    fixed = [n for n, lo, hi in zip(names, config['lower'], config['upper']) if lo == hi]
    freeze = sorted(set(fixed + ['MX'] + list(config['fit_state']['category_indices'])))
    command = [str(base/'bin/el9_amd64_gcc12/combine'), '-M', 'MultiDimFit',
        '-d', str(args.workspace.resolve()), '-D', 'asimov_prefit', '--mass', '125',
        '--algo', 'none', '--saveWorkspace', '--cminDefaultMinimizerStrategy', '1',
        '--cminDefaultMinimizerTolerance', '0.1', '--cminDefaultMinimizerType', 'Minuit2',
        '--cminDefaultMinimizerAlgo', 'Migrad', '--freezeParameters', ','.join(freeze),
        '-v', '3', '-n', '.v10_asimov']
    report = dict(status='preflight', script_sha256=sha(__file__), host=platform.node(),
        root_version=R.gROOT.GetVersion(), combine_version='10.3.3',
        library=str(library.resolve()), library_sha256=sha(library),
        workspace_sha256=sha(args.workspace), configuration_sha256=sha(args.configuration),
        dataset='asimov_prefit', dataset_identity=config['dataset_identity'], command=command,
        initial_parameters=names, initial_values=config['initial'],
        stopping_rule=dict(type='Minuit2', algorithm='Migrad', strategy=1, tolerance=.1,
                           error_definition=.5, raw_nll_edm_target=1e-4),
        scope='One native-v10 Migrad fit, explicit nominal Asimov, unchanged start and effective bounds. Snapshot errors remain uncertified at the four accepted HHVV clipping bins.')
    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2)+'\n')
        temporary.replace(args.output)
    save()
    preflight = args.directory/'preflight'
    preflight.mkdir()
    precommand = command + ['--skipInitialFit']
    with (preflight/'combine.log').open('x') as stream:
        subprocess.run(precommand, cwd=preflight, stdout=stream, stderr=subprocess.STDOUT,
                       check=True, timeout=600)
    text = (preflight/'combine.log').read_text()
    assert 'Doing initial fit:' not in text and 'Minuit2Minimizer : Valid minimum' not in text
    results = list(preflight.glob('higgsCombine*.root'))
    assert len(results) == 1
    handle = R.TFile.Open(str(results[0]), 'READ')
    ws = handle.Get('w')
    snap = ws.getSnapshot('MultiDimFit')
    assert snap
    for i, name in enumerate(names):
        v = snap.find(name)
        assert v and v.getVal() == config['initial'][i], name
        bounds = (v.getVal(), v.getVal()) if v.isConstant() else (v.getMin(), v.getMax())
        assert list(bounds) == [config['lower'][i], config['upper'][i]], name
    for name, value in config['fit_state']['fixed_real_values'].items():
        v = snap.find(name)
        assert v and v.isConstant() and v.getVal() == value, name
    for name, value in config['fit_state']['category_indices'].items():
        assert ws.cat(name).getIndex() == value
    for name, lower, upper in zip(config['profile_names'], config['profile_lower'], config['profile_upper']):
        v = snap.find(name)
        assert v and v.getMin() == lower and v.getMax() == upper, name
    data = ws.data('asimov_prefit')
    assert data.numEntries() == config['dataset_entries']
    assert abs(data.sumEntries()-config['dataset_sum_weights']) < 1e-6
    handle.Close()
    report['preflight'] = dict(status='passed', command=precommand,
        result_sha256=sha(results[0]), log_sha256=sha(preflight/'combine.log'))
    if args.preflight_only:
        report['status'] = 'passed_preflight_only'
        save()
        return
    # Use the existing transparent counter wrappers with this stack's headers.
    source = Path('scripts/native_minuit_counters.cc').resolve()
    counter = (args.directory/'native_minuit_counters.so').resolve()
    libdir = subprocess.check_output(['root-config', '--libdir'], text=True).strip()
    compile_command = ['g++', '-O2', '-fPIC', '-shared', str(source),
        *shlex.split(subprocess.check_output(['root-config', '--cflags'], text=True)),
        '-L'+libdir, '-Wl,-rpath,'+libdir, '-lMinuit2', '-ldl', '-o', str(counter)]
    build = subprocess.run(compile_command, capture_output=True, text=True)
    (args.directory/'counter_build.log').write_text(build.stdout+build.stderr)
    assert build.returncode == 0
    counter_prefix = (args.directory/'native_calls').resolve()
    env = dict(os.environ, LD_PRELOAD=str(counter), COMBINE_JAX_NATIVE_COUNTER_PREFIX=str(counter_prefix))
    report['call_instrumentation'] = dict(source_sha256=sha(source), library_sha256=sha(counter), command=compile_command)
    report['status'] = 'running_native_fit'
    save()
    log = args.directory/'combine.log'
    started = time.perf_counter()
    child = None
    try:
        with log.open('x') as stream:
            child = subprocess.Popen(command, cwd=args.directory, stdout=stream, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            while child.poll() is None:
                elapsed = time.perf_counter()-started
                ps = subprocess.run(['ps', '--no-headers', '-g', str(child.pid), '-o', 'rss='],
                                    capture_output=True, text=True)
                rss = sum(float(v) for v in ps.stdout.split())/1024 + resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
                report['peak_combined_rss_mib'] = max(rss, report.get('peak_combined_rss_mib', 0.))
                report['elapsed_seconds'] = elapsed
                save()
                if elapsed > 43200 or rss > 11264:
                    raise RuntimeError('Native v10 fit exceeded the 12-hour or 11-GiB RSS guard')
                time.sleep(2)
        report.update(returncode=child.returncode, executable_wall_seconds=time.perf_counter()-started)
        assert child.returncode == 0
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        save()
    text = log.read_text()
    summaries = re.findall(r'Minuit2Minimizer : (Valid|Invalid) minimum - status = (\d+).*?Edm\s*=\s*([\deE.+-]+).*?Nfcn\s*=\s*(\d+)', text, re.S)
    assert len(summaries) == 1, summaries
    valid, status, edm, nfcn = summaries[0]
    results = list(args.directory.glob('higgsCombine*.root'))
    assert len(results) == 1
    handle = R.TFile.Open(str(results[0]), 'READ')
    saved = handle.Get('w')
    snapshot = saved.getSnapshot('MultiDimFit')
    values = [float(snapshot.find(n).getVal()) for n in names]
    errors = [float(snapshot.find(n).getError()) if n not in fixed else 0. for n in names]
    handle.Close()
    report['fit'] = dict(status=int(status), edm=float(edm), edm_target_met=0 <= float(edm) < 1e-4,
        parameters=names, values=values, errors=errors)
    events = [json.loads(line) for p in args.directory.glob('native_calls.*.jsonl') for line in p.read_text().splitlines()]
    phases = [r for r in events if r['event'] == 'minimize_return']
    assert len(phases) == 1
    p = phases[0]
    report['evaluation_counts'] = dict(function=int(nfcn), actual_nll_evaluations=p['phase_nll'],
        numerical_gradient_evaluations=p['phase_numerical_gradients']+p['phase_hessian_gradients'])
    logger = args.directory/'combine_logger.out'
    matches = set(float(v) for v in re.findall(r'Minimized in ([\d.]+) seconds', text+(logger.read_text() if logger.exists() else '')))
    assert len(matches) == 1
    report['migrad_seconds'] = matches.pop()
    handle = R.TFile.Open(str(args.workspace), 'READ')
    ws = handle.Get('w')
    mc = ws.obj('ModelConfig')
    for name in config['fit_state']['nuisance_bounds_removed']:
        ws.var(name).removeMin()
        ws.var(name).removeMax()
    for name, value in zip(names, values):
        ws.var(name).setVal(value)
    for flag in config['runtime_flags']:
        R.runtimedef.set(flag, config['runtime_flags'][flag])
    nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), ws.data('asimov_prefit'), mc.GetNuisanceParameters())
    nll.setChannelMasks(mc.GetPdf().channelMasks())
    nll.setAnalyticBarlowBeeston(True)
    nll.clearConstantZeroPoint()
    nll.clearZeroPoint()
    R.RooAbsReal.setEvalErrorLoggingMode(R.RooAbsReal.CountErrors)
    R.RooAbsReal.clearEvalErrorLog()
    report['fit']['canonical_nll'] = float(nll.getVal())
    assert np.isfinite(report['fit']['canonical_nll']) and R.RooAbsReal.numEvalErrors() == 0
    if args.tensor_fit:
        with np.load(args.tensor_fit, allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved['parameter_names'], names)
            np.testing.assert_array_equal(saved['points'][0], config['initial'])
            tensor_values = saved['fit_values'].copy()
            tensor_nll = float(saved['fit_value'])
        for name, value in zip(names, tensor_values):
            ws.var(name).setVal(float(value))
        R.RooAbsReal.clearEvalErrorLog()
        native_nll = float(nll.getVal())
        report['tensor_point'] = dict(arrays_sha256=sha(args.tensor_fit),
            native_nll=native_nll, tensor_nll=tensor_nll, error=abs(native_nll-tensor_nll),
            roofit_errors=int(R.RooAbsReal.numEvalErrors()))
        assert np.isfinite(native_nll) and R.RooAbsReal.numEvalErrors() == 0
    nll.setAnalyticBarlowBeeston(False)
    report.update(status='converged_pending_gpu_comparison' if valid == 'Valid' and int(status) == 0 and report['fit']['edm_target_met'] else 'failed_native_convergence',
                  log_sha256=sha(log), result_sha256=sha(results[0]))
    save()
    print(report['status'], report['migrad_seconds'], flush=True)


if __name__ == '__main__':
    main()
