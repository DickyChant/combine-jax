"""Public workspace-to-tensor import with explicit data and fit configuration."""
from importlib.metadata import version
from pathlib import Path
import time

from .datasets import dataset_selection
from .runtime.resources import save_json, sha


def hig_kl_options():
    """Published HIG-23-006 fixed-coupling scan settings, without input paths."""
    fixed='r r_gghh r_qqhh r_vhh kt kF CV C2V kc ktau ks kmu kb'.split()
    return dict(snapshot='MultiDimFit',poi='kl',mass=125.38,
        parameters={n:1. for n in fixed},freeze=[*fixed,'MH'],
        ranges={'kl':[-15.,26.]},freeze_attributes=['all'],
        runtime_defines={'MINIMIZER_MaxCalls':9999999,'MINIMIZER_analytic':1,
            'FAST_VERTICAL_MORPH':1,'OPTIMIZE_BOUNDS':0,'SIMNLL_GROUPCONSTRAINTS':10,
            'CACHINGPDF_NOCLONE':1,'MINIMIZER_freezeDisassociatedParams':1,
            'MINIMIZER_multiMin_hideConstants':1,'MINIMIZER_multiMin_maskConstraints':1,
            'MINIMIZER_multiMin_maskChannels':2},
        template_convention='v10_density',cbshape_convention='root630',profile_bounds_policy='clip')


def import_workspace(path, output, *, data, analysis, dataset_kind=None, workspace_name='w',
                     snapshot=None, poi=None, parameters=None, freeze=(), ranges=None,
                     freeze_attributes=(), mass=None, runtime_defines=None, discrete='fixed',
                     template_convention='v10_density',cbshape_convention='native',
                     profile_bounds_policy='reject',discrete_profile_mode='independent',
                     compact_density=True):
    """Read a workspace or file-level dataset and save a framework-neutral directory.

    ROOT stays in this CPU import process. No source file or data are modified;
    an external dataset is cloned into the in-memory workspace under an alias.
    Setting a POI here occurs before constant folding, unlike fixing a fit point.
    Density storage compaction is exact and enabled by default. Disabling it is
    intended for same-input memory/numerical comparisons only.
    """
    if type(compact_density) is not bool:raise ValueError('Density storage compaction must be boolean')
    if discrete_profile_mode not in ('independent','native_batched'):
        raise ValueError('Unknown discrete profile mode')
    if discrete!='profile' and discrete_profile_mode!='independent':
        raise ValueError('Native-batched mode requires discrete profiling')
    runtime_defines=dict(runtime_defines or {})
    if discrete_profile_mode=='native_batched':
        if runtime_defines.get('MINIMIZER_multiMin_maskChannels',2)!=2:
            raise ValueError('Native-batched mode requires MINIMIZER_multiMin_maskChannels=2')
        runtime_defines['MINIMIZER_multiMin_maskChannels']=2
    import jax
    if not jax.config.x64_enabled or any(d.platform!='cpu' for d in jax.devices()):
        raise RuntimeError('Workspace import requires float64 CPU JAX in the ROOT environment')
    path=Path(path);output=Path(output)
    if output.exists():raise FileExistsError('Preserve existing imported model')
    kind=dataset_kind or ('observed' if data=='data_obs' else 'asimov')
    selection=dataset_selection(data,analysis=analysis,kind=kind)
    from .root_io import load_caching_nll,REFERENCE_FLAGS
    from .tensor.model import from_workspace
    start=time.perf_counter();root=load_caching_nll()
    flags={**REFERENCE_FLAGS,**(runtime_defines or {})}
    flags['MINIMIZER_skipDiscreteIterations']=int(discrete=='fixed')
    for name,value in flags.items():root.runtimedef.set(name,int(value))
    handle=root.TFile.Open(str(path),'READ')
    if not handle or handle.IsZombie():raise OSError(path)
    try:
        w=handle.Get(workspace_name)
        if not w:raise ValueError('Workspace not found: '+workspace_name)
        selection=dataset_selection(data,analysis=analysis,kind=kind,
            parameter_names=[str(v.GetName()) for v in w.allVars()])
        if snapshot:
            if not w.getSnapshot(snapshot):raise ValueError('Snapshot not found: '+snapshot)
            w.loadSnapshot(snapshot)
        mc=w.obj('ModelConfig')
        if not mc:raise ValueError('ModelConfig not found')
        def var(name):
            v=w.var(name)
            if not v:raise ValueError('Unknown parameter: '+name)
            return v
        for name,(low,high) in (ranges or {}).items():
            if not low<high:raise ValueError('Invalid range: '+name)
            var(name).setRange(low,high)
        for name,value in (parameters or {}).items():
            v=var(name)
            if not v.getMin()<=value<=v.getMax():raise ValueError('Assigned value outside bounds: '+name)
            v.setVal(value)
        for name in freeze:var(name).setConstant(True)
        for v in mc.GetNuisanceParameters() or ():
            if any(v.getAttribute(attr) for attr in freeze_attributes):v.setConstant(True)
        if mass is not None:var('MH').setVal(mass);var('MH').setConstant(True)
        if poi:
            if poi in freeze:raise ValueError('POI cannot also be frozen')
            var(poi).setConstant(False);mc.SetParametersOfInterest(root.RooArgSet(var(poi)))
        selected=w.data(data)
        imported_name=data
        if not selected:
            selected=handle.Get(data)
            if not selected or not selected.InheritsFrom('RooAbsData'):
                raise ValueError('Dataset not found: '+data)
            imported_name='combine_jax_selected_data'
            if w.data(imported_name):raise ValueError('Reserved dataset alias already exists')
            if getattr(w,'import')(selected,root.RooFit.Rename(imported_name),root.RooFit.Silence()):
                raise RuntimeError('Cannot import external dataset into memory')
        model=from_workspace(w,data=imported_name,analysis=analysis,dataset_kind=kind,
            discrete_policy=discrete,template_convention=template_convention,
            cbshape_convention=cbshape_convention,profile_bounds_policy=profile_bounds_policy,
            discrete_profile_mode=discrete_profile_mode,compact_density_storage=compact_density)
        array_bytes=sum(a.nbytes for a in model.arrays.values())
        binned=model.metadata['components'].get('binned',{})
        compaction=binned.get('density_storage_compaction')
        discrete=model.metadata.get('discrete_profile',{}).get('channels',{})
        transformations={name:summary for name,summary in (
            ('binned',compaction),('discrete_channels',discrete.get('density_storage_compaction')))
            if summary is not None}
        removed=sum(item['removed_float64_bytes'] for item in transformations.values())
        density_storage=dict(enabled=compact_density,changed=bool(removed),
            transformation=compaction,transformations=transformations,
            model_array_bytes_before=array_bytes+removed,
            model_array_bytes_after=array_bytes,removed_bytes=removed,
            scope='Main and discrete-candidate density arenas; all runtime reads, writes and alternatives retained, float64 unchanged')
        receipt=dict(source_sha256=sha(path),dataset=selection,
            dataset_identity=model.metadata['provenance']['dataset']['identity'],
            workspace=workspace_name,snapshot=snapshot,poi=poi,parameters=parameters or {},
            freeze=list(freeze),ranges=ranges or {},freeze_attributes=list(freeze_attributes),mass=mass,
            runtime_defines=flags,discrete=discrete,discrete_profile_mode=discrete_profile_mode,
            template_convention=template_convention,
            cbshape_convention=cbshape_convention,profile_bounds_policy=profile_bounds_policy,
            density_storage=density_storage,
            root_version=str(root.gROOT.GetVersion()),combine_version=version('higgs-combine-tool'),
            validation='Import only; native NLL, gradient and method gates are separate')
        model.metadata['workflow_import']=receipt
        model.save(output)
        report=dict(status='imported',**receipt,model_sha256={n:sha(output/n) for n in ('model.json','arrays.npz')},
            parameters_count=len(model.metadata['parameter_names']),channels=len(model.metadata['channel_names']),
            array_bytes=array_bytes,wall_s=time.perf_counter()-start)
        save_json(output/'import.json',report)
        return model,report
    finally:handle.Close()
