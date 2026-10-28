"""Isolated pip ROOT import; consume only the native pre-fit Asimov dataset."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time


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
    assert generation['status'] == 'generated_native_v10_background_asimov'
    assert generation['config'] == config and generation['signal_strength'] == 0.
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
        model = from_workspace(workspace, config['datacard'], data=name, template_convention='v10_density')
    assert model.metadata['provenance']['dataset']['identity'] == identity
    choices = model.metadata['provenance']['discrete_choices']
    if choices and config['discrete_policy'] != 'fixed':
        raise NotImplementedError('RooMultiPdf profiling is not validated; --discrete-policy fixed must be explicit')
    model.metadata['asymptotic_asimov'] = dict(poi=config['poi'], signal_strength=0., mode='prefit',
        dataset=name, dataset_identity=identity, source_workspace_sha256=generation['source_workspace_sha256'],
        native_asimov_workspace_sha256=generation['workspace_sha256'],
        globals=generation['globals'], root_version=generation['root_version'], combine_version='10.3.3',
        template_convention='v10_density', native_generation=True,
        fit_bounds_policy='workspace_ranges_asymptotic', conditional_fit=None,
        discrete_policy='fixed' if choices else 'profile', fixed_discrete_choices=choices)
    model.save(directory/'model')
    report = dict(status='imported_native_v10_background_asimov', model=str(directory/'model'),
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
