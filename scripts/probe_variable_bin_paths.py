"""Separate plain RooHistPdf generation from morpher evaluation, with fixed data.

Reads only synthetic v10-built workspaces. Does not rebuild a workspace or fit.
Run the identical script/inputs in isolated v10, official v11 and pip v11 stacks.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plain1d', 'plain2d', 'morpher', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--pip-stack', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    import ROOT as R
    if args.pip_stack:
        from cms_combine.cli import runtime_environment
        env = runtime_environment()
        for p in env['ROOT_INCLUDE_PATH'].split(':'):
            R.gInterpreter.AddIncludePath(p)
        library = next(Path(p) / 'libHiggsAnalysisCombinedLimit.so'
                       for p in env['LD_LIBRARY_PATH'].split(':')
                       if (Path(p) / 'libHiggsAnalysisCombinedLimit.so').is_file())
        source = Path('vendor/combine_pypi/vendor/CombinedLimit').resolve()
    else:
        base = Path(os.environ['CMSSW_BASE'])
        R.gInterpreter.AddIncludePath(str(base / 'src'))
        library = base / 'lib' / os.environ['SCRAM_ARCH'] / 'libHiggsAnalysisCombinedLimit.so'
        source = base / 'src/HiggsAnalysis/CombinedLimit'
    assert R.gSystem.Load(str(library)) >= 0
    R.gROOT.SetBatch(True)
    assert R.gInterpreter.Declare(
        '#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/CachingNLL.h"\n'
        '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"')
    for flag in ('ADDNLL_RECURSIVE', 'ADDNLL_GAUSSNLL', 'ADDNLL_HISTNLL',
                 'ADDNLL_CBNLL', 'ADDNLL_ROOREALSUM_FACTOR', 'ADDNLL_ROOREALSUM_NONORM',
                 'ADDNLL_ROOREALSUM_BASICINT', 'ADDNLL_ROOREALSUM_KEEPZEROS',
                 'ADDNLL_PRODNLL', 'ADDNLL_HFNLL', 'ADDNLL_HISTFUNCNLL',
                 'ADDNLL_ROOREALSUM_CHEAPPROD'):
        R.runtimedef.set(flag, 1)
    R.runtimedef.set('TMCSO_PseudoAsimov', 0)
    R.runtimedef.set('TMCSO_AdaptivePseudoAsimov', 0)
    report = dict(status='running', script_sha256=sha(__file__),
                  root_version=R.gROOT.GetVersion(), root_commit=R.gROOT.GetGitCommit(),
                  cmssw=os.environ.get('CMSSW_VERSION'), library=str(library.resolve()),
                  library_sha256=sha(library), combine_source_directory=str(source),
                  scope='Synthetic data only; identical v10 workspace/data bytes, no fits.',
                  source_sha256={}, cases={})
    for name in ('src/AsimovUtils.cc', 'src/ToyMCSamplerOpt.cc',
                 'src/VerticalInterpHistPdf.cc', 'interface/FastTemplate.h',
                 'interface/FastTemplate.hpp'):
        report['source_sha256'][name] = sha(source / name)

    def rows(data, names):
        result = []
        for i in range(data.numEntries()):
            row = data.get(i)
            result.append([*[float(row.getRealValue(n)) for n in names], float(data.weight())])
        return sorted(result)

    held = []
    for label in ('plain1d', 'plain2d', 'morpher'):
        path = getattr(args, label)
        handle = R.TFile.Open(str(path), 'READ')
        ws = handle.Get('w')
        mc = ws.obj('ModelConfig')
        obs = mc.GetObservables()
        axes = sorted([a for a in obs if a.InheritsFrom('RooRealVar')], key=lambda a: a.GetName())
        names = [a.GetName() for a in axes]
        data_name = 'asimov_bonly' if label == 'morpher' else 'data_obs'
        data = ws.data(data_name)
        assert data and ws
        reference_rows = rows(data, names)
        nuisance = 'shape' if label == 'morpher' else 'lumi'
        initial_r = 0. if label == 'morpher' else 1.
        ws.var('r').setVal(initial_r)
        ws.var(nuisance).setVal(0.)
        ws.var('r').setConstant(False)
        ws.var(nuisance).setConstant(False)
        components = mc.GetPdf().getComponents()
        densities = {}
        for pdf in components:
            if pdf.ClassName() not in ('RooHistPdf', 'FastVerticalInterpHistPdf3D', 'RooAddPdf'):
                continue
            values = []
            for row in reference_rows:
                for a, x in zip(axes, row):
                    a.setVal(x)
                values.append(float(pdf.getVal(obs)))
            densities[pdf.GetName()] = dict(class_name=pdf.ClassName(), values=values)
        generated = R.asimovutils.asimovDatasetNominal(mc, initial_r, -1)
        generated_rows = rows(generated, names)
        ws.var('r').setConstant(False)

        # A direct RooHistPdf control bypasses Combine's generation code.
        direct = next(p for p in components if p.ClassName() == 'RooHistPdf')
        pure_roofit = direct.generateBinned(R.RooArgSet(*axes),
                                          R.RooFit.NumEvents(100.), R.RooFit.ExpectedData(True))
        direct_obs = R.RooArgSet(*axes)
        histogram_args = []
        if len(axes) > 1:
            histogram_args.append(R.RooFit.YVar(axes[1]))
        if len(axes) > 2:
            histogram_args.append(R.RooFit.ZVar(axes[2]))
        histogram_ratios = {}
        # Compare ROOT's default with an explicit density conversion. For the
        # extended mixture the expected yield is present in both and cancels.
        targets = [direct] + [p for p in components if p.ClassName() == 'RooAddPdf']
        for pdf in targets:
            default = pdf.createHistogram('default', axes[0], *histogram_args)
            density = pdf.createHistogram('density', axes[0], *histogram_args, R.RooFit.Scaling(False))
            default.SetDirectory(0)
            density.SetDirectory(0)
            histogram_ratios[pdf.GetName()] = [
                float(default.GetBinContent(default.FindBin(*row[:-1]))) /
                float(density.GetBinContent(density.FindBin(*row[:-1]))) for row in reference_rows]
            held.extend((default, density))
        nll = R.cacheutils.CachingSimNLL(mc.GetPdf(), data, mc.GetNuisanceParameters())
        nll.clearConstantZeroPoint()
        nll.clearZeroPoint()
        points = [(initial_r, 0.)] + [(0.2 + i * .19, -1.8 + i * .15) for i in range(25)]
        values = []
        for r, t in points:
            ws.var('r').setVal(r)
            ws.var(nuisance).setVal(t)
            values.append(float(nll.getVal()))
        report['cases'][label] = dict(workspace_sha256=sha(path), data_name=data_name,
            row_sha256=hashlib.sha256(json.dumps(reference_rows).encode()).hexdigest(),
            names=names, reference_rows=reference_rows, generated_rows=generated_rows,
            pure_roofit_pdf=direct.GetName(), pure_roofit_rows=rows(pure_roofit, names),
            nominal_densities=densities, default_to_scaling_false=histogram_ratios,
            parameter_names=['r', nuisance], points=points, nll=values)
        held.extend((handle, ws, data, generated, direct_obs, pure_roofit, nll, components))
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    report['status'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(report['root_version'], {k: v['nll'][0] for k, v in report['cases'].items()}, flush=True)


if __name__ == '__main__':
    main()
