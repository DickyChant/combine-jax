"""Standalone native Combine-v10 pre-fit generator (CMSSW Python 3.9 compatible)."""
import fnmatch
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def discrete_snapshot(workspace):
    """Export native stored penalties and active names, without fitting data.

    ROOT 6.30's Combine RooMultiPdf has no public indexCategory accessor, so
    locate its category among its variables. Restore the index after inspection.
    """
    mc=workspace.obj('ModelConfig')
    result={}
    components=mc.GetPdf().getComponents()
    for pdf in components:
        if pdf.ClassName()!='RooMultiPdf':
            continue
        variables=pdf.getVariables()
        categories=[v for v in variables if v.InheritsFrom('RooAbsCategory')]
        if len(categories)!=1:
            raise ValueError('RooMultiPdf must have one index category: '+pdf.GetName())
        category=categories[0]
        initial=int(category.getIndex())
        alternatives=[]
        try:
            for index in range(pdf.getNumPdfs()):
                category.setIndex(index)
                alternative=pdf.getPdf(index)
                penalty=float(pdf.getCorrection())
                if not math.isfinite(penalty):
                    raise ValueError('Nonfinite native discrete penalty')
                parameters=alternative.getParameters(mc.GetObservables())
                alternatives.append(dict(index=index,name=str(alternative.GetName()),
                    pdf_class=str(alternative.ClassName()),correction=penalty,
                    parameters=sorted(str(v.GetName()) for v in parameters
                        if v.InheritsFrom('RooRealVar') and not v.isConstant())))
        finally:
            category.setIndex(initial)
        result[str(pdf.GetName())]=dict(category=str(category.GetName()),
            initial_index=initial,category_constant=bool(category.isConstant()),alternatives=alternatives)
    return result


def generate(request_path, directory):
    config = json.loads(Path(request_path).read_text())
    multidim = config.get('method') == 'MultiDimFit'
    if config['run'] not in ('blind','expected') or not config['no_fit_asimov']:
        raise ValueError('Native generation is blinded, pre-fit background-only')
    if not multidim and config['signal_strength'] != 0.:
        raise ValueError('Background-only Asimov is required')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    os.sched_setaffinity(0, [min(os.sched_getaffinity(0))])
    import ROOT as R
    base = Path(os.environ['CMSSW_BASE'])
    library = base/'lib'/os.environ['SCRAM_ARCH']/'libHiggsAnalysisCombinedLimit.so'
    # Combine v10 has no --version CLI switch. Bind the installer's explicit
    # release declaration to the actual shared library, including worker copies.
    version = config['native_version']
    if version != '10.3.3' or not R.gROOT.GetVersion().startswith('6.30'):
        raise RuntimeError('Native v10.3.3 / ROOT 6.30 required; got '+version+' / '+R.gROOT.GetVersion())
    if digest(library) != config['native_library_sha256']:
        raise RuntimeError('Native library differs from the configured runtime reference')
    if R.gSystem.Load(str(library)) < 0:
        raise RuntimeError('Cannot load the configured native Combine library')
    R.gInterpreter.AddIncludePath(str(base/'src'))
    if not R.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"\n'
                                 '#include "HiggsAnalysis/CombinedLimit/interface/ProfilingTools.h"'):
        raise RuntimeError('Cannot load native Asimov declarations')
    R.gROOT.SetBatch(True)
    handle = R.TFile.Open(config['workspace'], 'READ')
    if not handle or handle.IsZombie():
        raise OSError(config['workspace'])
    ws = handle.Get('w')
    mc = ws.obj('ModelConfig')
    requested_pois = config.get('pois', [config['poi']])
    poi = ws.var(requested_pois[0])
    if not poi:
        raise ValueError('Unknown POI '+config['poi'])
    current_pois = [p.GetName() for p in mc.GetParametersOfInterest()]
    if multidim:
        new_pois = R.RooArgSet()
        for poi_name in requested_pois:
            variable = ws.var(poi_name)
            if not variable:
                raise ValueError('Unknown POI '+poi_name)
            if mc.GetNuisanceParameters() and mc.GetNuisanceParameters().find(poi_name):
                raise NotImplementedError('Promoting a constrained nuisance to a POI awaits ignoreConstraint validation: '+poi_name)
            new_pois.add(variable)
            variable.setConstant(False)
        mc.SetParametersOfInterest(new_pois)
    elif current_pois != [config['poi']]:
        raise ValueError('Requested POI must equal the single workspace POI: '+repr(current_pois))
    for name, limits in config['ranges'].items():
        variable = ws.var(name)
        if not variable:
            raise ValueError('Unknown ranged parameter '+name)
        variable.setRange(*limits)
    for name, value in config['parameters'].items():
        variable = ws.var(name)
        if not variable:
            raise ValueError('Unknown assigned parameter '+name)
        if not variable.getMin() <= value <= variable.getMax():
            raise ValueError('Assigned value outside range for '+name)
        variable.setVal(value)
    variables = {v.GetName(): v for v in ws.allVars()}
    frozen = set()
    for pattern in config['freeze']:
        matches = [name for name in variables if fnmatch.fnmatchcase(name, pattern)]
        if not matches:
            raise ValueError('Freeze pattern matches no variable: '+pattern)
        frozen.update(matches)
    for group in config['freeze_groups']:
        members = ws.set('group_'+group)
        if not members:
            raise ValueError('Unknown nuisance group '+group)
        frozen.update(v.GetName() for v in members)
    if set(requested_pois) & frozen:
        raise ValueError('Cannot freeze a requested POI')
    for name in frozen:
        variables[name].setConstant(True)
    if ws.var('MH'):
        ws.var('MH').setVal(config['mass'])
        ws.var('MH').setConstant(True)
    poi.setConstant(False)
    if not multidim:
        poi.setMin(0.)
    if not multidim and config['r_max'] is not None:
        poi.setMax(config['r_max'])
    flags = {'TMCSO_AdaptivePseudoAsimov': 0, 'TMCSO_PseudoAsimov': 0}
    for name, value in flags.items():
        R.runtimedef.set(name, value)
    discrete=discrete_snapshot(ws)
    nominal = {v.GetName(): float(v.getVal()) for collection in
               (mc.GetNuisanceParameters(), mc.GetGlobalObservables()) for v in (collection or ())
               if v.InheritsFrom('RooAbsReal')}
    start = time.perf_counter()
    generation_values = {n: float(ws.var(n).getVal()) for n in requested_pois}
    signal_strength = generation_values[requested_pois[0]] if multidim else 0.
    generation_values[requested_pois[0]] = signal_strength
    data = R.asimovutils.asimovDatasetNominal(mc, signal_strength, -1)
    if not data or data.sumEntries() <= 0:
        raise RuntimeError('Empty native Asimov dataset')
    for name, value in nominal.items():
        if ws.var(name).getVal() != value:
            raise RuntimeError('Generation changed nominal '+name)
    if discrete_snapshot(ws)!=discrete:
        raise RuntimeError('Generation changed nominal discrete configuration')
    name = 'combine_jax_asimov_nominal' if multidim else 'combine_jax_asimov_bonly'
    if ws.data(name):
        raise ValueError('Reserved Asimov name already exists')
    data.convertToTreeStore()
    if getattr(ws, 'import')(data, R.RooFit.Rename(name)):
        raise RuntimeError('Asimov workspace import failed')
    poi.setVal(signal_strength); poi.setConstant(False)
    nll0 = None
    removed_bounds = {}
    if multidim:
        # FitterAlgoBase applies OPTIMIZE_BOUNDS before constructing its NLL.
        for variable in mc.GetNuisanceParameters() or ():
            if variable.InheritsFrom('RooRealVar') and variable.getAttribute('optimizeBounds'):
                removed_bounds[variable.GetName()] = (variable.getMin(), variable.getMax())
                variable.removeMin(); variable.removeMax()
        R.gInterpreter.Declare('''
            #include "HiggsAnalysis/CombinedLimit/interface/Combine.h"
            double combine_jax_initial_nll(RooStats::ModelConfig &mc, RooAbsData &data) {
                auto nll = combineCreateNLL(*mc.GetPdf(), data, mc.GetNuisanceParameters(), true);
                return nll->getVal();
            }
        ''')
        nll0 = float(R.combine_jax_initial_nll(mc, data))
        if not math.isfinite(nll0):
            raise RuntimeError('Nonfinite native initial offset NLL')
        for variable_name, bounds in removed_bounds.items():
            variable = ws.var(variable_name)
            if math.isfinite(bounds[0]): variable.setMin(bounds[0])
            if math.isfinite(bounds[1]): variable.setMax(bounds[1])
    target = directory/'workspace.root'
    if ws.writeToFile(str(target)):
        raise RuntimeError('Asimov workspace write failed')
    check = R.TFile.Open(str(target), 'READ')
    saved = check.Get('w').data(name)
    if saved.numEntries() != data.numEntries() or saved.sumEntries() != data.sumEntries():
        raise RuntimeError('Saved Asimov differs on reopen')
    report = dict(status='generated_native_v10_nominal_asimov' if multidim else 'generated_native_v10_background_asimov',
        generation_kind='combine_prefit_nominal_asimov' if multidim else 'combine_prefit_background_asimov',
        config=config, workspace=str(target), workspace_sha256=digest(target),
        source_workspace_sha256=digest(config['workspace']),
        dataset=dict(name=name, entries=data.numEntries(), sum_weights=data.sumEntries()),
        poi=config['poi'], pois=requested_pois, generation_poi_values=generation_values,
        signal_strength=signal_strength, mode='prefit', frozen_parameters=sorted(frozen),
        native_initial_offset_nll=nll0, optimized_nuisance_bounds=sorted(removed_bounds),
        minos_delta_nll=.5 if mc.GetPdf().InheritsFrom('RooSimultaneousOpt') else None,
        root_version=R.gROOT.GetVersion(), combine_version=version,
        version_source='installer declaration, bound to native library SHA256', library_sha256=digest(library),
        cmssw=str(base), runtime_flags=flags,discrete=discrete,
        globals={v.GetName(): float(v.getVal()) for v in (mc.GetGlobalObservables() or ())},
        generation_seconds=time.perf_counter()-start, reopen_verified=True,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (directory/'generation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    generate(*sys.argv[1:])
