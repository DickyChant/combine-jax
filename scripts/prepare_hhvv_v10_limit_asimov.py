"""Native-v10 pre-fit background Asimov for a full-HHVV expected limit.

Standalone CMSSW Python 3.9 process. No JAX/pip ROOT and no data fit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--mass', type=int, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    assert not args.directory.exists()
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    import ROOT as R
    base = Path(os.environ['CMSSW_BASE'])
    assert base.name == 'CMSSW_14_1_0_pre4' and R.gROOT.GetVersion() == '6.30/07'
    library = base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
    assert R.gSystem.Load(str(library)) >= 0
    R.gInterpreter.AddIncludePath(str(base/'src'))
    assert R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"\n'
                                 '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    R.gROOT.SetBatch(True)
    handle = R.TFile.Open(str(args.workspace), 'READ')
    assert handle and not handle.IsZombie()
    ws = handle.Get('w')
    assert ws and ws.var('MX').getVal() == args.mass and ws.var('MX').isConstant()
    assert ws.var('MH').getVal() == 125. and ws.var('MH').isConstant()
    mc = ws.obj('ModelConfig')
    assert [v.GetName() for v in mc.GetParametersOfInterest()] == ['r_hhvv']
    r = ws.var('r_hhvv')
    r.setMin(0.)
    # These disable pseudo-Asimov sampling; every native integration bin is kept.
    flags = {'TMCSO_AdaptivePseudoAsimov': 0, 'TMCSO_PseudoAsimov': 0}
    for name, value in flags.items():
        R.runtimedef.set(name, value)
    nominal = {v.GetName(): float(v.getVal()) for collection in
               (mc.GetNuisanceParameters(), mc.GetGlobalObservables()) for v in collection
               if v.InheritsFrom('RooAbsReal')}
    categories = {v.GetName(): int(v.getIndex()) for v in ws.allCats()
                  if v.GetName().startswith('pdfindex_')}
    output = args.directory/'generation.json'
    report = dict(status='generating', script_sha256=sha(__file__), mass=args.mass, MH=125.,
        poi='r_hhvv', signal_strength=0., mode='prefit', run='blind', no_fit_asimov=True,
        source_workspace_sha256=sha(args.workspace), source_workspace=str(args.workspace),
        root_version=R.gROOT.GetVersion(), combine_version='10.3.3', library_sha256=sha(library),
        cmssw=str(base), runtime_flags=flags, nominal_discrete_categories=categories,
        initial_r_max=float(r.getMax()), generation_kind='combine_prefit_background_asimov',
        scope='Native asimovDatasetNominal(ModelConfig, 0), the --run blind/--noFitAsimov path. No input data is read by the generator and no fit is run. Discrete categories are nominal for generation; fit treatment is recorded separately.')
    def save():
        output.write_text(json.dumps(report, indent=2)+'\n')
    save()
    start = time.perf_counter()
    data = R.asimovutils.asimovDatasetNominal(mc, 0., -1)
    assert data and data.sumEntries() > 0
    for name, value in nominal.items():
        assert ws.var(name).getVal() == value, name
    name = 'combine_jax_asimov_bonly'
    assert not ws.data(name)
    # A full simultaneous dataset has many inactive observable columns. Its
    # vector store can exceed ROOT's 1 GiB per-object byte-count limit; TTree
    # baskets avoid serialising every row into one object buffer.
    data.convertToTreeStore()
    assert not getattr(ws, 'import')(data, R.RooFit.Rename(name))
    r.setVal(0.); r.setConstant(False)
    target = args.directory/'workspace_asimov.root'
    assert not ws.writeToFile(str(target))
    report.update(status='saved_pending_reopen',
        generation_seconds=time.perf_counter()-start, workspace=str(target), workspace_sha256=sha(target),
        dataset=dict(name=name, entries=int(data.numEntries()), sum_weights=float(data.sumEntries())),
        globals={v.GetName(): float(v.getVal()) for v in mc.GetGlobalObservables()},
        storage_class=str(ws.data(name).store().ClassName()),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save()
    check = R.TFile.Open(str(target), 'READ')
    assert check and not check.IsZombie()
    saved = check.Get('w').data(name)
    assert saved.numEntries() == data.numEntries() and saved.sumEntries() == data.sumEntries()
    for index in [0, 1, data.numEntries()//3, data.numEntries()//2, data.numEntries()-1]:
        original_row, saved_row = data.get(index), saved.get(index)
        assert saved.weight() == data.weight()
        for variable in original_row:
            if variable.InheritsFrom('RooAbsReal'):
                assert saved_row.find(variable.GetName()).getVal() == variable.getVal()
            elif variable.InheritsFrom('RooAbsCategory'):
                assert saved_row.find(variable.GetName()).getCurrentIndex() == variable.getCurrentIndex()
    report.update(status='generated_native_v10_background_asimov', reopen_verified=True,
                  generation_seconds=time.perf_counter()-start,
                  peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    save()
    print(report['status'], args.mass, report['dataset'], flush=True)


if __name__ == '__main__':
    main()
