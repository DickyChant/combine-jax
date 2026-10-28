"""Isolated ROOT 6.40 import for the authorized MX3000 data or conditional Asimov."""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time
from hhvv_mx3000_authorization import authorize, check_workspace
from tensor_benchmark_common import save_json, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind', required=True, choices=['observed', 'conditional_asimov'])
    args = p.parse_args()
    if os.environ.get('JAX_PLATFORMS') != 'cpu':
        raise RuntimeError('The isolated ROOT importer requires JAX_PLATFORMS=cpu')
    manifest = json.loads(Path('build/hhvv_observed_runtime.json').read_text())
    authorize(manifest, 'build/hhvv_observed/input/workspace.root', 'build/hhvv_observed/input/card.txt')
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    output = Path('reports')/('gpu_hhvv_MX3000_'+args.kind+'_import.json')
    assert not output.exists()
    report = dict(status='running', kind=args.kind, mass=3000, script_sha256=sha(__file__))
    save_json(output, report)
    start = time.perf_counter()
    try:
        # The import boundary evaluates the old JAX oracle on CPU. Check that
        # backend before traversing the large workspace; CUDA belongs to fits.
        import jax
        assert jax.config.x64_enabled and all(d.platform == 'cpu' for d in jax.devices())
        report['import_jax_platforms'] = [d.platform for d in jax.devices()]
        from combine_jax.root_io import open_workspace, load_caching_nll
        from combine_jax.tensor.model import from_workspace
        root = load_caching_nll()
        assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
        generation = None
        if args.kind == 'observed':
            source, name = Path('build/hhvv_observed/input/workspace.root'), 'data_obs'
        else:
            generation = json.loads(Path('reports/gpu_hhvv_MX3000_conditional_generation.json').read_text())
            assert generation['status'] == 'generated_conditional_asimov' and generation['native_same_point_nll_error'] < 1e-6
            source, name = Path(generation['workspace']), generation['dataset']['name']
            assert sha(source) == generation['workspace_sha256']
        with open_workspace(source) as ws:
            check_workspace(ws)
            ws.var('r_hhvv').setMin(0.)
            ws.var('r_hhvv').setConstant(False)
            model = from_workspace(ws, 'build/hhvv_observed/input/card.txt', data=name,
                                   template_convention='v10_density')
        assert len(model.metadata['channel_names']) == 459
        choices = model.metadata['provenance']['discrete_choices']
        assert choices == manifest['fixed_discrete_choices']
        proof = dict(mass=3000, combination='HHVV_v4_final', authorized_workspace_sha256=manifest['workspace_sha256'],
                     manifest_sha256=sha('build/hhvv_observed_runtime.json'), data=name,
                     dataset_identity=model.metadata['provenance']['dataset']['identity'])
        model.metadata['mx3000_observed_authorization'] = proof
        if generation:
            model.metadata['asymptotic_asimov'] = dict(poi='r_hhvv', signal_strength=0.,
                mode='conditional_background_fit', dataset=name, dataset_identity=proof['dataset_identity'],
                source_workspace_sha256=manifest['workspace_sha256'], native_generation=True,
                combine_version='10.3.3', root_version='6.30/07', mass=3000,
                conditional_fit_report_sha256=generation['conditional_fit_report_sha256'],
                globals=generation['globals'], discrete_policy='fixed', fixed_discrete_choices=choices)
        target = Path('build/hhvv_observed')/(args.kind+'_model')
        model.save(target)
        report.update(status='imported', model=str(target), dataset=model.metadata['provenance']['dataset'],
            workspace_sha256=sha(source), model_sha256={n:sha(target/n) for n in ('model.json','arrays.npz')},
            channels=len(model.metadata['channel_names']), parameters=len(model.metadata['parameter_names']),
            array_bytes=sum(a.nbytes for a in model.arrays.values()), fixed_discrete_choices=choices,
            root_version=root.gROOT.GetVersion(), combine_version=version('higgs-combine-tool'),
            template_convention='v10_density')
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        report.update(wall_seconds=time.perf_counter()-start,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save_json(output, report)


if __name__ == '__main__':
    main()
