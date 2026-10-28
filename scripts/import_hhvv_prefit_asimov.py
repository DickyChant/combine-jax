"""Import the exact saved Combine pre-fit Asimov into a fresh neutral model."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time
import resource

import numpy as np

from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--generation-report', type=Path, required=True)
    parser.add_argument('--card', type=Path, required=True)
    parser.add_argument('--points', type=Path, required=True)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace-memory',action='store_true')
    args = parser.parse_args()
    assert not args.directory.exists() and not args.output.exists()
    generation = json.loads(args.generation_report.read_text())
    assert generation['status'] == 'generated_and_identity_verified'
    assert generation['generation_kind'] == 'combine_prefit_asimov' and generation['MH'] == 125.
    assert sha(args.workspace) == generation['workspace_sha256']
    report = dict(provenance(), status='importing', script_sha256=sha(__file__),
        workspace_sha256=sha(args.workspace), generation_report_sha256=sha(args.generation_report),
        scope='Fresh named-data import; all model parameters start at workspace nominal. No fits or observed-data references. Numerical validation follows separately.')
    save_json(args.output, report); monitor = Resources(False, args.output)
    started = time.perf_counter()
    try:
        if args.trace_memory:
            from combine_jax.tensor.bundle import TensorBundle
            original=TensorBundle.__post_init__
            def traced(bundle):
                size=sum(a.nbytes for a in bundle.arrays.values())
                if size>8*1024**2:
                    print('bundle_memory_before',bundle.metadata['kind'],size,
                          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)
                original(bundle)
                if size>8*1024**2:
                    print('bundle_memory_after',bundle.metadata['kind'],size,
                          resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,flush=True)
            TensorBundle.__post_init__=traced
        from combine_jax.root_io import load_caching_nll, open_workspace
        from combine_jax.tensor.model import from_workspace
        from combine_jax.tensor.fit_policy import POLICY, check_native_fit_state
        from combine_jax.tensor.parameter_info import model_parameter_info
        from tensor_benchmark_dataset import benchmark_dataset
        root = load_caching_nll()
        assert root.gROOT.GetVersion() == '6.40.00' and version('higgs-combine-tool') == '11.0.0'
        with open_workspace(args.workspace) as workspace:
            assert workspace.var('MH').getVal() == 125. and workspace.var('MH').isConstant()
            assert workspace.var('MX').getVal() == 2000. and workspace.var('MX').isConstant()
            model = from_workspace(workspace, args.card, data='asimov_prefit', fit_policy=POLICY)
            check_native_fit_state(workspace, model)
        assert model.metadata['provenance']['dataset']['identity'] == generation['dataset']['identity']
        model.metadata['provenance']['dataset_generation'] = dict(kind='combine_prefit_asimov',
            MH=125., dataset_identity=generation['dataset']['identity'],
            generation_report_sha256=sha(args.generation_report))
        info = model_parameter_info(model)
        with np.load(args.points, allow_pickle=False) as old:
            names = old['parameter_names']; points = old['points']
        np.testing.assert_array_equal(names, info.metadata['names'])
        np.testing.assert_array_equal(points[0], info.arrays['initial'])
        assert points.shape == (26,len(names))
        args.directory.mkdir(parents=True)
        model.save(args.directory/'model')
        np.savez_compressed(args.directory/'points.npz', parameter_names=names, points=points)
        case = dict(workspace=str(args.workspace), dataset='asimov_prefit', configuration={},
            generation_report=str(args.generation_report), model=str(args.directory/'model'),
            points=str(args.directory/'points.npz'), reference=str(args.directory/'reference.npz'),
            hashes={str(p):sha(p) for p in [args.workspace,args.generation_report,
                args.directory/'points.npz',*(args.directory/'model').iterdir()]})
        benchmark_dataset(case,model,label='hhvv')
        manifest=args.directory/'manifest.pending.json'; save_json(manifest,dict(hhvv=case))
        report.update(status='imported_identity_verified_numerical_gates_pending',
            manifest=str(manifest), manifest_sha256=sha(manifest), structure_key=model.structure_key,
            dataset=model.metadata['provenance']['dataset'], parameters=len(names),
            array_bytes=sum(a.nbytes for a in model.arrays.values()),
            root_version=root.gROOT.GetVersion(), combine_version=version('higgs-combine-tool'),
            artifacts=case['hashes'])
    except BaseException as exc:
        report.update(status='failed',error=repr(exc)); raise
    finally:
        monitor.close(); report.update(elapsed_seconds=time.perf_counter()-started,resources=monitor.snapshot())
        save_json(args.output,report)
    print(report['status'],report['dataset']['identity']['sha256'],flush=True)


if __name__=='__main__': main()
