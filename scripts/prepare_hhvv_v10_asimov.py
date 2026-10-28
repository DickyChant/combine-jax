"""Fresh native-v10 nominal HHVV Asimov and 26 fixed-point NLLs; no fitting.

Standalone Python 3.9/CMSSW boundary, with no pip ROOT or JAX import.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('workspace', 'points', 'directory'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists()
    args.directory.mkdir(parents=True)
    import ROOT as R
    import numpy as np
    base = Path(os.environ['CMSSW_BASE'])
    assert base.name == 'CMSSW_14_1_0_pre4' and R.gROOT.GetVersion() == '6.30/07'
    R.gInterpreter.AddIncludePath(str(base/'src'))
    library = base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
    assert R.gSystem.Load(str(library)) >= 0
    R.gROOT.SetBatch(True)
    executable = base/'bin/el9_amd64_gcc12/combine'
    loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
              if 'libHiggsAnalysisCombinedLimit.so' in line}
    assert len(loaded) == 1 and Path(next(iter(loaded))).resolve() == library.resolve()
    with np.load(args.points, allow_pickle=False) as f:
        names, points = f['parameter_names'], f['points']
    assert points.shape == (26, len(names))
    handle = R.TFile.Open(str(args.workspace), 'READ')
    ws = handle.Get('w')
    for name, value in [('MX', 2000.), ('MH', 125.)]:
        ws.var(name).setVal(value)
        ws.var(name).setConstant(True)
    np.testing.assert_array_equal([ws.var(str(n)).getVal() for n in names], points[0])
    mc = ws.obj('ModelConfig')
    pois = {p.GetName(): float(p.getVal()) for p in mc.GetParametersOfInterest()}
    prepared = args.directory/'nominal_input.root'
    ws.writeToFile(str(prepared))
    handle.Close()
    command = [str(executable), '-M', 'GenerateOnly', '-d', str(prepared.resolve()),
        '-t', '-1', '--mass', '125', '--saveToys', '--saveWorkspace',
        '--setParameters', ','.join(n+'='+format(v, '.17g') for n, v in sorted(pois.items())),
        '--freezeParameters', 'MX', '-n', '.v10_prefit_asimov', '-v', '1']
    output = args.directory/'generation.json'
    report = dict(status='generating', script_sha256=sha(__file__), command=command,
        source_workspace=str(args.workspace), source_workspace_sha256=sha(args.workspace),
        points_sha256=sha(args.points), root_version=R.gROOT.GetVersion(), combine_version='10.3.3',
        cmssw=str(base), library=str(library.resolve()), library_sha256=sha(library),
        executable_sha256=sha(executable), generation_kind='combine_prefit_asimov',
        nominal_pois=pois, MH=125., MX=2000., dataset_name='asimov_prefit',
        scope='Native v10 GenerateOnly -t -1 at workspace nominal values. No fit, no observed likelihood evaluation. NLL references use the saved Asimov only.')
    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    start = time.perf_counter()
    with (args.directory/'generate.log').open('x') as stream:
        result = subprocess.run(command, cwd=args.directory, stdout=stream, stderr=subprocess.STDOUT, timeout=1800)
    assert result.returncode == 0
    report['generation_wall_seconds'] = time.perf_counter()-start
    generated = list(args.directory.glob('higgsCombine*.root'))
    assert len(generated) == 1
    log = (args.directory/'generate.log').read_text()
    assert 'Doing initial fit:' not in log and 'Minuit2Minimizer : Valid minimum' not in log
    source = R.TFile.Open(str(generated[0]), 'READ')
    data = source.Get('toys/toy_asimov')
    handle = R.TFile.Open(str(prepared), 'READ')
    ws = handle.Get('w')
    assert data and not ws.data('asimov_prefit')
    assert not getattr(ws, 'import')(data, R.RooFit.Rename('asimov_prefit'))
    target = args.directory/'workspace_asimov.root'
    ws.writeToFile(str(target))
    np.testing.assert_array_equal([ws.var(str(n)).getVal() for n in names], points[0])
    report.update(status='generated_native_v10', workspace=str(target), workspace_sha256=sha(target),
        native_toy_sha256=sha(generated[0]), dataset=dict(name='asimov_prefit',
        entries=int(data.numEntries()), sum_weights=float(data.sumEntries())))
    save()
    assert R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    flags = ('RECURSIVE GAUSSNLL HISTNLL CBNLL ROOREALSUM_FACTOR ROOREALSUM_NONORM '
             'ROOREALSUM_BASICINT ROOREALSUM_KEEPZEROS PRODNLL HFNLL HISTFUNCNLL ROOREALSUM_CHEAPPROD').split()
    for flag in flags:
        R.runtimedef.set('ADDNLL_'+flag, 1)
    mc = ws.obj('ModelConfig')
    for v in mc.GetNuisanceParameters():
        if v.InheritsFrom('RooRealVar') and v.getAttribute('optimizeBounds'):
            v.removeMin()
            v.removeMax()
    nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), ws.data('asimov_prefit'), mc.GetNuisanceParameters())
    nll.setChannelMasks(mc.GetPdf().channelMasks())
    nll.setAnalyticBarlowBeeston(True)
    nll.setNoDeepLogEvalError(False)
    nll.clearConstantZeroPoint()
    nll.clearZeroPoint()
    R.RooAbsReal.setEvalErrorLoggingMode(R.RooAbsReal.CountErrors)
    values = []
    for i, point in enumerate(points):
        for name, value in zip(names, point):
            ws.var(str(name)).setVal(float(value))
        R.RooAbsReal.clearEvalErrorLog()
        value = float(nll.getVal())
        assert np.isfinite(value) and value < 9e9 and R.RooAbsReal.numEvalErrors() == 0
        values.append(value)
        print('v10 Asimov NLL point', i, value, flush=True)
    np.savez_compressed(args.directory/'reference.npz', parameter_names=names, points=points, values=values)
    report.update(status='generated_native_v10_nll_reference',
        reference_sha256=sha(args.directory/'reference.npz'), nll_points=len(values))
    save()
    nll.setAnalyticBarlowBeeston(False)
    print(report['status'], report['dataset'], flush=True)


if __name__ == '__main__':
    main()
