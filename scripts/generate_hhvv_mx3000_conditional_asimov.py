"""Native v10 generation at the GPU conditional fit; separate CMSSW process."""
import json
import math
import os
from pathlib import Path
import resource
import time
from hhvv_mx3000_authorization import authorize, check_workspace, sha


def main():
    manifest = json.loads(Path('build/hhvv_observed_runtime.json').read_text())
    source = Path('build/hhvv_observed/input/workspace.root')
    authorize(manifest, source, 'build/hhvv_observed/input/card.txt')
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    output = Path('reports/gpu_hhvv_MX3000_conditional_generation.json')
    assert not output.exists()
    report = dict(status='running', mass=3000, script_sha256=sha(__file__))
    def save():
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n'); temp.replace(output)
    save()
    started = time.perf_counter()
    try:
        import ROOT as R
        base = Path(os.environ['CMSSW_BASE'])
        assert base.name == 'CMSSW_14_1_0_pre4' and R.gROOT.GetVersion() == '6.30/07'
        library = base/'lib/el9_amd64_gcc12/libHiggsAnalysisCombinedLimit.so'
        assert sha(library) == manifest['native_library_sha256']
        assert R.gSystem.Load(str(library)) >= 0
        R.gInterpreter.AddIncludePath(str(base/'src'))
        assert R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"\n'
            '#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
            '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
        R.gROOT.SetBatch(True)
        handle = R.TFile.Open(str(source), 'READ')
        ws = handle.Get('w')
        check_workspace(ws)
        fit_path = Path('reports/gpu_hhvv_MX3000_background_fit.json')
        fit = json.loads(fit_path.read_text())
        assert fit['status'] == 'converged' and fit['fit']['status'] == 0
        assert fit['fit']['corrected_estimated_edm'] < 1e-4 and fit['poi_value'] == 0.
        assert fit['manifest_sha256'] == sha('build/hhvv_observed_runtime.json')
        for name, value in zip(fit['names'], fit['point']):
            ws.var(name).setVal(value)
        ws.var('r_hhvv').setMin(0.); ws.var('r_hhvv').setVal(0.)
        mc = ws.obj('ModelConfig')
        for flag in ('RECURSIVE GAUSSNLL HISTNLL CBNLL ROOREALSUM_FACTOR ROOREALSUM_NONORM '
                     'ROOREALSUM_BASICINT ROOREALSUM_KEEPZEROS PRODNLL HFNLL HISTFUNCNLL ROOREALSUM_CHEAPPROD').split():
            R.runtimedef.set('ADDNLL_'+flag, 1)
        for flag in ('TMCSO_AdaptivePseudoAsimov', 'TMCSO_PseudoAsimov'):
            R.runtimedef.set(flag, 0)
        nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), ws.data('data_obs'), mc.GetNuisanceParameters())
        nll.setChannelMasks(mc.GetPdf().channelMasks())
        nll.setAnalyticBarlowBeeston(True)
        nll.clearConstantZeroPoint(); nll.clearZeroPoint()
        value = float(nll.getVal())
        report['native_same_point_nll_error'] = abs(value-fit['fit']['value'])
        report['native_nll'] = value
        save()
        assert math.isfinite(value) and report['native_same_point_nll_error'] < 1e-6
        # Retain the native analytically minimized BB-lite values for generation.
        nll.setAnalyticBarlowBeeston(False)
        snapshot = R.RooArgSet()
        data = R.asimovutils.asimovDatasetWithFit(mc, ws.data('data_obs'), snapshot, False, 0., -1)
        assert data and data.sumEntries() > 0
        for variable in snapshot:
            ws.var(variable.GetName()).setVal(variable.getVal())
        name = 'combine_jax_asimov_conditional_bonly'
        data.convertToTreeStore()
        assert not ws.data(name) and not getattr(ws, 'import')(data, R.RooFit.Rename(name))
        ws.var('r_hhvv').setConstant(False)
        target = Path('build/hhvv_observed/workspace_conditional_asimov.root')
        assert not target.exists() and not ws.writeToFile(str(target))
        check = R.TFile.Open(str(target), 'READ')
        saved = check.Get('w').data(name)
        assert saved.numEntries() == data.numEntries() and saved.sumEntries() == data.sumEntries()
        report.update(status='generated_conditional_asimov', workspace=str(target), workspace_sha256=sha(target),
            dataset=dict(name=name, entries=int(data.numEntries()), sum_weights=float(data.sumEntries())),
            conditional_fit_report_sha256=sha(fit_path), globals={v.GetName():float(v.getVal()) for v in snapshot},
            native_library_sha256=sha(library), root_version=R.gROOT.GetVersion(), combine_version='10.3.3',
            no_fit_asimov=False, needs_fit=False, generation='asimovDatasetWithFit', reopen_verified=True)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report.update(wall_seconds=time.perf_counter()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save()


if __name__ == '__main__':
    main()
