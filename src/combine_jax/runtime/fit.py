"""ROOT-free execution and measurement of a blinded expected-limit artifact."""
from importlib.metadata import version
import json
import os
from pathlib import Path
import resource
import sys
import time

from .resources import Resources, provenance, save_json, sha


def verify(backend):
    if backend == 'gpu' and 'MIG-' in os.environ.get('CUDA_VISIBLE_DEVICES', ''):
        raise RuntimeError('MIG allocations are unsupported; request a full A100/H100 GPU')
    import jax
    devices = jax.devices()
    if not jax.config.x64_enabled or 'ROOT' in sys.modules:
        raise RuntimeError('Fitting requires float64 and an isolated ROOT-free interpreter')
    if backend == 'gpu':
        if len(devices) != 1 or devices[0].platform != 'gpu':
            raise RuntimeError('Exactly one assigned GPU is required')
        if not any(name in devices[0].device_kind for name in ('H100','A100')):
            raise RuntimeError('Only H100/A100 is supported for this blinded campaign')
    elif any(d.platform != 'cpu' for d in devices):
        raise RuntimeError('CPU backend selected but non-CPU devices found')
    print('Devices verified:', [(d.platform,d.device_kind) for d in devices],
          'JAX',jax.__version__,'float64',jax.config.x64_enabled,flush=True)
    return devices


def main(request_path, preparation_path, output):
    started = time.perf_counter()
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    stack = resource.getrlimit(resource.RLIMIT_STACK)
    resource.setrlimit(resource.RLIMIT_STACK,(stack[1],stack[1]))
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    config = json.loads(Path(request_path).read_text())
    devices = verify(config['backend'])
    import jax
    from ..tensor.bundle import TensorBundle
    from ..methods.runner import run_expected_model
    from ..methods.asymptotic import LimitOptions
    from ..methods.root_output import audit_crossings
    preparation = json.loads(Path(preparation_path).read_text())
    model_path = Path(preparation['model'])
    for name, checksum in preparation['model_sha256'].items():
        if sha(model_path/name) != checksum:
            raise ValueError('Model checksum mismatch: '+name)
    report = dict(provenance(), status='running', config=config, preparation=preparation,
        devices=[dict(platform=d.platform,kind=d.device_kind) for d in devices],
        versions={name:version(name) for name in ('jax','jaxlib','numpy','scipy')},
        profiles=[], jax_allocator_samples=[], float64=True, static_guard_gib=8.)
    if config['backend'] == 'gpu':
        report['versions']['jax-cuda12-plugin'] = version('jax-cuda12-plugin')
    monitor = Resources(config['backend']=='gpu', output)
    def save():
        report['resources'] = monitor.snapshot()
        save_json(output, report)
    def compile_progress(name, stage, timings, memory):
        monitor.stage = name+'_'+stage
        report.update(timings=timings,compiler_memory=memory)
        print('compile',name,stage,flush=True)
        save()
    def fit_progress(record):
        monitor.stage = 'profiles'
        report['profiles'].append(record)
        report['jax_allocator_samples'].append(devices[0].memory_stats())
        print('profile',{k:v for k,v in record.items()
            if k not in ('parameter_names','parameter_values')},flush=True)
        save()
    save()
    try:
        model = TensorBundle.load(model_path)
        result = run_expected_model(model, config['poi'],
            options=LimitOptions(confidence_level=config['confidence_level'],
                r_abs_accuracy=config['r_abs_accuracy'],r_rel_accuracy=config['r_rel_accuracy']),
            r_max=config['r_max'],compiler_memory_limit_gib=config['static_guard_gib'],
            discrete_policy='fixed' if config['discrete_policy']=='fixed' else 'profile',
            release_compile_cache=True,compile_progress=compile_progress,fit_progress=fit_progress)
        rows = audit_crossings(result)
        report.update(result, rows=rows, all_fits_converged=all(r['status']==0 for r in result['profile_fits']),
            status='complete' if all(r['status']=='resolved' for r in rows) else 'unresolved_crossing')
    except BaseException as exc:
        report.update(status='failed',error=repr(exc))
        raise
    finally:
        monitor.close()
        report.update(process_seconds=time.perf_counter()-started,
            jax_allocator_final=devices[0].memory_stats(), root_imported='ROOT' in sys.modules)
        save()


if __name__ == '__main__':
    if sys.argv[1] == '--verify':
        verify(sys.argv[2])
    else:
        main(*sys.argv[1:])
