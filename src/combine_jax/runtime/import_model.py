"""Isolated pip ROOT import; consume only the native pre-fit Asimov dataset."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time


def verify_discrete_penalties(choices,native_discrete):
    if not native_discrete:
        raise ValueError('Discrete profiling requires native-v10 category and penalty metadata')
    for pdf,choice in choices.items():
        native=native_discrete[pdf]
        if choice['category']!=native['category'] or choice['index']!=native['initial_index']:
            raise ValueError('Native/imported discrete category mismatch: '+pdf)
        penalties=choice.get('corrections',[choice['correction']])
        expected=([a['correction'] for a in native['alternatives']] if choice.get('profiled')
            else [native['alternatives'][native['initial_index']]['correction']])
        if list(penalties)!=expected:
            raise ValueError('Native/imported stored discrete penalties differ: '+pdf)


def main(request_path, generation_path, directory):
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise RuntimeError('ROOT import requires CPU JAX')
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    if not jax.config.x64_enabled or any(d.platform != 'cpu' for d in jax.devices()):
        raise RuntimeError('Importer requires float64 CPU JAX')
    from ..root_io import load_caching_nll, open_workspace
    from ..tensor.model import from_workspace
    from ..dataset_identity import workspace_dataset_identity
    from ..methods.workspace import digest
    config = json.loads(Path(request_path).read_text())
    generation = json.loads(Path(generation_path).read_text())
    multidim = config.get('method') == 'MultiDimFit'
    assert generation['status'] == ('generated_native_v10_nominal_asimov' if multidim
                                    else 'generated_native_v10_background_asimov')
    assert generation['config'] == config
    if not multidim:
        assert generation['signal_strength'] == 0.
    assert digest(generation['workspace']) == generation['workspace_sha256']
    root = load_caching_nll()
    if root.gROOT.GetVersion() != '6.40.00' or version('higgs-combine-tool') != '11.0.0':
        raise RuntimeError('Importer must use pip Combine 11.0.0 / ROOT 6.40.00')
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    with open_workspace(generation['workspace']) as workspace:
        name = generation['dataset']['name']
        identity = workspace_dataset_identity(workspace, name)
        data = workspace.data(name)
        assert data.numEntries() == generation['dataset']['entries']
        assert data.sumEntries() == generation['dataset']['sum_weights']
        model = from_workspace(workspace, config['datacard'], data=name, template_convention='v10_density',
            fit_policy='combine_v11_default' if multidim else 'workspace',
            discrete_policy='profile' if config['discrete_policy']=='profile' else 'fixed')
    assert model.metadata['provenance']['dataset']['identity'] == identity
    choices = model.metadata['provenance']['discrete_choices']
    if choices and config['discrete_policy'] == 'reject':
        raise NotImplementedError('RooMultiPdf requires explicit --discrete-policy profile or fixed')
    if choices and config['discrete_policy']=='profile':
        verify_discrete_penalties(choices,generation.get('discrete'))
    model.metadata['multidim_asimov' if multidim else 'asymptotic_asimov'] = dict(
        poi=config['poi'], signal_strength=generation['signal_strength'], mode='prefit',
        dataset=name, dataset_identity=identity, source_workspace_sha256=generation['source_workspace_sha256'],
        native_asimov_workspace_sha256=generation['workspace_sha256'],
        globals=generation['globals'], root_version=generation['root_version'], combine_version='10.3.3',
        template_convention='v10_density', native_generation=True,
        fit_bounds_policy='optimize_bounds_multidimfit' if multidim else 'workspace_ranges_asymptotic',
        conditional_fit=None,
        discrete_policy=config['discrete_policy'] if choices else 'profile',
        fixed_discrete_choices=choices if config['discrete_policy']=='fixed' else {},
        native_discrete=generation.get('discrete',{}))
    model.save(directory/'model')
    report = dict(status='imported_native_v10_nominal_asimov' if multidim else 'imported_native_v10_background_asimov',
        model=str(directory/'model'),
        model_sha256={n: digest(directory/'model'/n) for n in ('model.json','arrays.npz')},
        dataset=dict(name=name, identity=identity), generation=generation,
        import_root_version=root.gROOT.GetVersion(), import_combine_version=version('higgs-combine-tool'),
        template_convention='v10_density', channels=len(model.metadata['channel_names']),
        parameters=len(model.metadata['parameter_names']), array_bytes=sum(a.nbytes for a in model.arrays.values()),
        import_seconds=time.perf_counter()-start,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    (directory/'preparation.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main(*sys.argv[1:])
