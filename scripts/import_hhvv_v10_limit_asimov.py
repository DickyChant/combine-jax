"""Import native-v10 background Asimov with explicit density correction.

This is the separate pip ROOT 6.40 import process; generation stays native v10.
Preserve discrete metadata and workspace nuisance bounds for the method driver.
"""
import argparse
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import time

from tensor_benchmark_common import save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--fixed-discrete', action='store_true',
        help='Record the selected branches for an explicitly exploratory fixed-discrete limit')
    args = parser.parse_args()
    assert not args.directory.exists()
    args.directory.mkdir(parents=True)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    generation = json.loads(args.generation.read_text())
    assert generation['status'] == 'generated_native_v10_background_asimov'
    assert generation['signal_strength'] == 0. and generation['MH'] == 125.
    source = Path(generation['workspace'])
    assert sha(source) == generation['workspace_sha256']
    output = args.directory/'preparation.json'
    report = dict(status='importing', script_sha256=sha(__file__), generation=generation,
        generation_report_sha256=sha(args.generation), card_sha256=sha(args.card),
        template_convention='v10_density',
        scope='Native-v10 pre-fit background Asimov, imported in isolated pip-v11/ROOT6.40 with explicit v10_density. Workspace nuisance bounds retained; discrete choices retained, not silently profiled or discarded.')
    save_json(output, report)
    started = time.perf_counter()
    try:
        from combine_jax.root_io import load_caching_nll, open_workspace
        from combine_jax.tensor.model import from_workspace
        from combine_jax.tensor.parameter_info import model_parameter_info
        from combine_jax.dataset_identity import workspace_dataset_identity
        root = load_caching_nll()
        assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
        with open_workspace(source) as ws:
            assert ws.var('MX').getVal() == generation['mass'] and ws.var('MX').isConstant()
            assert ws.var('r_hhvv').getVal() == 0. and not ws.var('r_hhvv').isConstant()
            name = generation['dataset']['name']
            data = ws.data(name)
            assert data.numEntries() == generation['dataset']['entries']
            assert data.sumEntries() == generation['dataset']['sum_weights']
            identity = workspace_dataset_identity(ws, name)
            model = from_workspace(ws, args.card, data=name, template_convention='v10_density')
        assert model.metadata['provenance']['dataset']['identity'] == identity
        assert not model.metadata.get('parameter_info', {}).get('fit_policy', {}).get('nuisance_bounds_removed')
        model.metadata['asymptotic_asimov'] = dict(poi='r_hhvv', signal_strength=0., mode='prefit',
            dataset=name, dataset_identity=identity, source_workspace_sha256=generation['source_workspace_sha256'],
            native_asimov_workspace_sha256=generation['workspace_sha256'],
            generation_report_sha256=sha(args.generation), fit_bounds_policy='workspace_ranges_asymptotic',
            conditional_fit=None, globals=generation['globals'], root_version='6.30/07', combine_version='10.3.3',
            template_convention='v10_density', native_generation=True, mass=generation['mass'])
        if args.fixed_discrete:
            model.metadata['asymptotic_asimov'].update(discrete_policy='fixed',
                fixed_discrete_choices=model.metadata['provenance'].get('discrete_choices', {}))
        model.save(args.directory/'model')
        info = model_parameter_info(model)
        report.update(status='imported_native_v10_background_asimov', model=str(args.directory/'model'),
            model_sha256={n: sha(args.directory/'model'/n) for n in ('model.json', 'arrays.npz')},
            import_root_version=root.gROOT.GetVersion(), import_combine_version=version('higgs-combine-tool'),
            dataset=dict(name=name, identity=identity), parameters=len(info.metadata['names']),
            discrete_choices=model.metadata['provenance'].get('discrete_choices', {}),
            structure_key=model.structure_key, array_bytes=sum(a.nbytes for a in model.arrays.values()))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        report.update(import_seconds=time.perf_counter()-started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        save_json(output, report)
    print(report['status'], generation['mass'], report['parameters'], flush=True)


if __name__ == '__main__':
    main()
