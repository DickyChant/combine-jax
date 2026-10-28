"""Measured ROOT-free MultiDimFit execution on an explicitly generated Asimov."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time

from .fit import verify
from .resources import Resources, provenance, save_json, sha


def main(request_path, preparation_path, output):
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK, (stack[1], stack[1]))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    config = json.loads(Path(request_path).read_text())
    if config['method'] != 'MultiDimFit' or config['run'] != 'expected':
        raise ValueError('Explicit Asimov MultiDimFit request required')
    devices = verify(config['backend'])
    import jax
    from ..tensor.bundle import TensorBundle
    from ..tensor.multidim_profile import MultiDimProfiler
    from ..methods.multidim import scan, impacts, DELTA68
    from ..local import release_compilation_cache
    preparation = json.loads(Path(preparation_path).read_text())
    if preparation['generation']['config'] != config:
        raise ValueError('Generation request differs from the fit request')
    if preparation['status'] != 'imported_native_v10_nominal_asimov':
        raise ValueError('Only the native nominal Asimov artifact is accepted')
    path = Path(preparation['model'])
    for name, checksum in preparation['model_sha256'].items():
        if sha(path/name) != checksum:
            raise ValueError('Model checksum mismatch: '+name)
    model = TensorBundle.load(path)
    if model.metadata['multidim_asimov']['dataset_identity'] != preparation['dataset']['identity']:
        raise ValueError('Asimov dataset identity mismatch')
    monitor = Resources(config['backend'] == 'gpu', output)
    report = dict(provenance(), status='running', config=config, preparation=preparation,
        devices=[dict(platform=d.platform, kind=d.device_kind) for d in devices],
        versions={n: version(n) for n in ('jax', 'jaxlib', 'numpy', 'scipy')},
        profiles=[], jax_allocator_samples=[], float64=True, static_guard_gib=8.,
        curvature_linearization='recompute', curvature_column_batch_size=1)
    if config['backend'] == 'gpu':
        report['versions']['jax-cuda12-plugin'] = version('jax-cuda12-plugin')
    def save():
        report['resources'] = monitor.snapshot()
        save_json(output, report)
    def compile_progress(name, stage, timings, memory):
        monitor.stage = name+'_'+stage
        report.update(timings=timings, compiler_memory=memory)
        print('compile', name, stage, flush=True)
        save()
    def fit_progress(record):
        monitor.stage = 'profiles'
        report['profiles'].append(record)
        report['jax_allocator_samples'].append(devices[0].memory_stats())
        save()
    driver = None
    try:
        groups = [tuple(config['pois'])] if config['algo'] == 'grid' else [(n,) for n in config['impact_parameters']]
        driver = MultiDimProfiler(model, conditional_parameters=groups,
            compiler_memory_limit_gib=config['static_guard_gib'], defer_compile=True,
            edm_validation=config['edm_validation'],
            global_edm_validation=config.get('global_edm_validation'),
            compile_progress=compile_progress, fit_progress=fit_progress)
        del model
        release_compilation_cache()
        driver.compile()
        begin = time.perf_counter()
        driver.global_minimum()
        report['global_values'] = dict(zip(driver.names, jax.device_get(driver.global_point).tolist()))
        nll0 = preparation['generation']['native_initial_offset_nll']
        if config['algo'] == 'grid':
            report['rows'] = scan(driver, config['pois'], config['points'], nll0=nll0)
        else:
            delta = DELTA68 if config['robust_fit'] else preparation['generation']['minos_delta_nll']
            if delta is None:
                raise NotImplementedError('Native Minos error definition has not been audited for this PDF type')
            result = impacts(driver, config['poi'], config['impact_parameters'], nll0=nll0, delta_nll=delta)
            report['impact_results'] = result
            if config['algo'] != 'batch-impacts':
                report['rows'] = next(iter(result.values()))['rows']
        report.update(status='complete', all_fits_converged=all(r['status'] == 0 for r in driver.records),
            execution_wall_seconds=time.perf_counter()-begin)
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), all_fits_converged=False)
        raise
    finally:
        if driver is not None:
            report.update(timings=driver.timings, compiler_memory=driver.compiler_memory)
        monitor.close()
        report.update(process_seconds=time.perf_counter()-started,
            jax_allocator_final=devices[0].memory_stats(), root_imported='ROOT' in sys.modules)
        save()


if __name__ == '__main__':
    main(*sys.argv[1:])
