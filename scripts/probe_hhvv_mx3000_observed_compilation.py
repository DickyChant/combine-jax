"""CPU-only bounded compilation/one-step check, never a converged fit campaign."""
import argparse
import json
import os
from pathlib import Path
import time

from hhvv_mx3000_authorization import authorize
from tensor_benchmark_common import Resources, provenance, save_json, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--release-cache', action='store_true')
    parser.add_argument('--compile-only', action='store_true',
        help='Compile the actual 10,000-iteration executable without running a fit')
    args = parser.parse_args()
    assert not args.output.exists()
    manifest_path = Path('build/hhvv_observed_runtime.json')
    manifest = json.loads(manifest_path.read_text())
    authorize(manifest)
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import numpy as np
    assert jax.config.x64_enabled and all(d.platform == 'cpu' for d in jax.devices())
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.tensor.limit_profile import LimitProfiler
    prep = json.loads(Path('reports/gpu_hhvv_MX3000_observed_import.json').read_text())
    assert prep['status'] == 'imported'
    for name, digest in prep['model_sha256'].items():
        assert sha(Path(prep['model'])/name) == digest
    report = dict(provenance(), status='running', jax_version=jax.__version__,
        mass=3000, run='compile_only' if args.compile_only else 'compile_and_one_iteration_only',
        float64=True, static_guard_gib=8.,
        release_cache=args.release_cache, events=[], model_sha256=prep['model_sha256'],
        scope='Same authorized observed MX3000 model; compilation diagnostic, no convergence claim.')
    monitor = Resources(False, args.output)
    started = time.perf_counter()
    def progress(name, stage, timings, memory):
        monitor.stage = name+':'+stage
        # Current RSS, unlike ru_maxrss, exposes the effect of cache release.
        rss = int(Path('/proc/self/statm').read_text().split()[1])*os.sysconf('SC_PAGE_SIZE')/2**20
        event = dict(executable=name, stage=stage, rss_mib=rss, resources=monitor.snapshot())
        report['events'].append(event)
        report.update(timings=timings, compiler_memory=memory)
        save_json(args.output, report)
        print(json.dumps(event), flush=True)
    save_json(args.output, report)
    try:
        model = TensorBundle.load(prep['model'])
        proof = model.metadata['mx3000_observed_authorization']
        assert proof['mass'] == 3000 and proof['manifest_sha256'] == sha(manifest_path)
        assert model.metadata['provenance']['discrete_choices'] == manifest['fixed_discrete_choices']
        # Compile the actual loop, or explicitly select a one-iteration probe.
        # No rows/parameters are removed and the likelihood is unchanged.
        fit = LimitProfiler(model, 'r_hhvv', global_up=.5, global_start_fraction=.1,
            compile_kinds=('conditional',), max_iterations=10000 if args.compile_only else 1,
            release_compile_cache=args.release_cache, compile_progress=progress)
        del model
        if args.compile_only:
            report.update(status='passed_compile_check', max_iterations=10000, fits_executed=0)
            return
        monitor.stage = 'one_iteration'
        begin = time.perf_counter()
        result = jax.block_until_ready(fit.executables['conditional'](fit.coordinates['conditional'], fit.arrays))
        # Status 1 (iteration cap) is expected and is never called convergence.
        summary = {k: jax.device_get(result[k][0]).item() for k in
            ('status','iteration','value','corrected_estimated_edm','function_evaluations','gradient_evaluations')}
        assert summary['status'] in (0, 1) and np.isfinite(summary['value'])
        assert np.isfinite(jax.device_get(result['values'])).all()
        values_file = args.output.with_suffix('.npz')
        np.savez(values_file, values=jax.device_get(result['values']),
                 gradient=jax.device_get(result['free_gradient']))
        report.update(status='passed_bounded_check', one_iteration=summary,
            execution_seconds=time.perf_counter()-begin, values_sha256=sha(values_file))
    except BaseException as exc:
        report.update(status='failed', error=repr(exc)); raise
    finally:
        monitor.close()
        report.update(wall_seconds=time.perf_counter()-started, resources=monitor.snapshot())
        save_json(args.output, report)


if __name__ == '__main__':
    main()
