"""Combine v11 Asimov generation at the ROOT import boundary."""
from importlib.metadata import version
import hashlib
import json
from pathlib import Path
import sys
import time


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare_expected_workspace(path, card, directory, *, poi=None, run='blind',
                               no_fit_asimov=False, data='data_obs', asimov_report=None,
                               r_max=None, parameters=None, freeze=(), mass=125., template_convention='v10_density'):
    """Generate native Asimov rows/globals; conditional nuisance fits use JAX.

    No original ROOT file is written. HHVV posterior-expected input requires a
    hash-bound Asimov generation report, never just a renamed observed dataset.
    """
    if run not in ('blind', 'expected'):
        raise ValueError('Only blind/expected limits are supported')
    from ..template_conventions import check_convention
    check_convention(template_convention)
    if any('/lcg/views/' in p for p in sys.path):
        raise RuntimeError('Prepare workspaces in the isolated Combine v11/ROOT 6.40 environment; GPU workers load the neutral artifact')
    from ..root_io import load_caching_nll, open_workspace
    from ..dataset_identity import workspace_dataset_identity
    from ..tensor.model import from_workspace
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('Preserve existing method preparation evidence')
    directory.mkdir(parents=True)
    start = time.perf_counter()
    root = load_caching_nll()
    if root.gROOT.GetVersion() != '6.40.00' or version('higgs-combine-tool') != '11.0.0':
        raise RuntimeError('Expected the isolated Combine v11 / ROOT 6.40 reference stack')
    if not root.gInterpreter.Declare('#include "HiggsAnalysis/CombinedLimit/interface/AsimovUtils.h"'):
        raise RuntimeError('Cannot load Combine Asimov generation declarations')
    prefit = run == 'blind' or no_fit_asimov
    report = dict(status='preparing', method='AsymptoticLimits', run=run, no_fit_asimov=prefit,
        workspace=str(path), workspace_sha256=digest(path), card=str(card), card_sha256=digest(card),
        combine_version='11.0.0', root_version='6.40.00', signal_strength=0.,template_convention=template_convention)
    with open_workspace(path) as workspace:
        mc = workspace.obj('ModelConfig')
        from ..graph import walk
        from .contracts import require_continuous_model
        require_continuous_model({'provenance': {'discrete_choices': {
            name: {} for name, node in walk(mc.GetPdf()).items() if node.ClassName() == 'RooMultiPdf'}}})
        pois = list(mc.GetParametersOfInterest())
        if len(pois) != 1:
            raise ValueError('First AsymptoticLimits implementation requires exactly one POI')
        poi = poi or str(pois[0].GetName())
        if poi != str(pois[0].GetName()):
            raise ValueError('Requested POI differs from the workspace ModelConfig')
        for name, value in (parameters or {}).items():
            variable = workspace.var(name)
            if not variable:
                raise ValueError(f'Unknown configuration parameter {name}')
            variable.setVal(float(value))
        for name in freeze:
            variable = workspace.var(name)
            if not variable or name == poi:
                raise ValueError(f'Cannot freeze parameter {name}')
            variable.setConstant(True)
        if workspace.var('MH'):
            workspace.var('MH').setVal(float(mass))
            workspace.var('MH').setConstant(True)
        r = workspace.var(poi)
        if r.isConstant():
            raise ValueError('POI is fixed in the supplied workspace')
        r.setMin(0.)
        if r_max is not None:
            if not 0 < r_max < float('inf'):
                raise ValueError('rMax must be finite and positive')
            r.setMax(float(r_max))
        report.update(poi=poi, initial_r_max=float(r.getMax()), parameters=parameters or {},
                      frozen_parameters=list(freeze), mass=mass)
        conditional = None
        from .template_asimov import variable_bin_channels, correct_asimov
        affected = variable_bin_channels(workspace) if template_convention == 'v10_density' else {}
        if not prefit:
            if workspace.var('r_hhvv'):
                if asimov_report is None:
                    raise ValueError('HHVV --run expected requires a verified Asimov input report, or --noFitAsimov/--run blind')
                proof = json.loads(Path(asimov_report).read_text())
                if (proof.get('workspace_sha256') != report['workspace_sha256']
                        or proof.get('dataset', {}).get('name') != data
                        or not str(proof.get('generation_kind', '')).startswith('combine_')
                        or 'asimov' not in str(proof.get('generation_kind', ''))):
                    raise ValueError('Input is not a verified Asimov workspace/dataset')
                if workspace_dataset_identity(workspace, data) != proof['dataset']['identity']:
                    raise ValueError('Asimov input identity mismatch')
                report['input_asimov_report_sha256'] = digest(asimov_report)
            if not workspace.data(data):
                raise ValueError(f'Workspace has no input dataset {data}')
            if affected:
                raise NotImplementedError('Conditional expected variable-bin models require a v10 background-fit validation; use blind or --noFitAsimov for the certified pre-fit path')
            from ..tensor.limit_profile import LimitProfiler
            # AsymptoticLimits derives from LimitAlgo, not FitterAlgoBase: it
            # preserves workspace nuisance bounds (no optimizeBounds pass).
            source = from_workspace(workspace, card, data=data,template_convention=template_convention)
            profiler = LimitProfiler(source, poi, r_max=float(r.getMax()))
            then = time.perf_counter()
            value = profiler(0.)
            for name, x in zip(profiler.names, profiler.point(), strict=True):
                workspace.var(name).setVal(float(x))
            # Restore analytically profiled bin nuisances too: they are omitted
            # from the explicit JAX parameter vector but affect generated rows.
            native_nll = root.cacheutils.CachingSimNLL(mc.GetPdf(), workspace.data(data), mc.GetNuisanceParameters())
            native_nll.setAnalyticBarlowBeeston(True)
            native_nll.clearConstantZeroPoint()
            native_nll.clearZeroPoint()
            native_value = float(native_nll.getVal())
            native_nll.setAnalyticBarlowBeeston(False)
            if abs(native_value-value) >= 1e-6:
                raise RuntimeError('Conditional background fit fails native same-point NLL gate')
            conditional = dict(nll=value, records=profiler.records, timings=profiler.timings,
                execution_seconds=time.perf_counter()-then, compiler_memory=profiler.compiler_memory,
                input_dataset=data, input_identity=workspace_dataset_identity(workspace, data),
                native_same_point_nll_error=abs(native_value-value))
            del source, profiler
        globals_snapshot = root.RooArgSet()
        if prefit:
            generated = root.asimovutils.asimovDatasetNominal(mc, 0., -1)
        else:
            # The conditional JAX fit above supplies nuisance values. Native
            # generation computes constraint/global-observable conventions only.
            generated = root.asimovutils.asimovDatasetWithFit(mc, workspace.data(data), globals_snapshot, False, 0., -1)
            for variable in globals_snapshot:
                workspace.var(variable.GetName()).setVal(float(variable.getVal()))
        name = 'combine_jax_asimov_bonly'
        generation_correction = None
        if affected:
            generated, generation_correction = correct_asimov(workspace,generated,affected)
        if workspace.data(name):
            raise ValueError('Workspace already contains the reserved method dataset')
        identity = workspace_dataset_identity(workspace, generated)
        if getattr(workspace, 'import')(generated, root.RooFit.Rename(name)):
            raise RuntimeError('Cannot import generated Asimov dataset')
        r.setConstant(False)
        r.setVal(0.)
        prepared = directory / 'workspace.root'
        workspace.writeToFile(str(prepared))
        model = from_workspace(workspace, card, data=name,template_convention=template_convention)
        if model.metadata['provenance']['dataset']['identity'] != identity:
            raise RuntimeError('Asimov dataset changed during neutral import')
        model.metadata['asymptotic_asimov'] = dict(poi=poi, signal_strength=0.,
            mode='prefit' if prefit else 'conditional_background_fit', dataset=name,
            fit_bounds_policy='workspace_ranges_asymptotic_v11',
            dataset_identity=identity, source_workspace_sha256=report['workspace_sha256'],
            conditional_fit=conditional, globals={str(v.GetName()): float(v.getVal()) for v in (mc.GetGlobalObservables() or ())},
            root_version='6.40.00', combine_version='11.0.0',template_convention=template_convention,
            generation_correction=generation_correction)
        model.save(directory / 'model')
        report.update(status='prepared_expected_asimov', dataset=dict(name=name, identity=identity),
            prepared_workspace=str(prepared), prepared_workspace_sha256=digest(prepared),
            model=str(directory / 'model'), model_sha256={n: digest(directory / 'model' / n) for n in ('model.json','arrays.npz')},
            conditional_fit=conditional, generation_correction=generation_correction, preparation_seconds=time.perf_counter()-start)
    (directory / 'preparation.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return model, report
