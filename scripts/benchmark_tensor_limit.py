"""Limit operation for the tensor benchmark harness; no ROOT import."""
import os
from pathlib import Path
import sys
import time
import traceback

from tensor_benchmark_common import Resources, provenance, save_json, sha


def benchmark_limit(args):
    if args.mode != 'jit' or args.compile_only or args.points or args.cache:
        raise ValueError('Limit benchmark currently requires jit, one prepared Asimov model, no external points/cache/compile-only')
    if args.output.exists():
        raise FileExistsError('Preserve existing limit evidence')
    if args.backend == 'gpu' and not os.environ.get('_CONDOR_SCRATCH_DIR'):
        raise ValueError('GPU limits run only in authorized HTCondor A100/H100 jobs')
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jaxlib
    devices = jax.devices()
    assert jax.config.x64_enabled and all(d.platform == args.backend for d in devices)
    if args.backend == 'gpu':
        assert len(devices) == 1 and any(x in devices[0].device_kind for x in ('A100','H100'))
    from combine_jax.tensor.bundle import TensorBundle
    from combine_jax.methods.runner import run_expected_model
    from combine_jax.methods.asymptotic import LimitOptions
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(provenance(), status='running', operation='limit', script_sha256=sha(__file__),
        benchmark_dispatch_sha256=sha('scripts/benchmark_tensor.py'), jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__, backend=args.backend, float64=True,
        model_sha256={n:sha(args.model/n) for n in ('model.json','arrays.npz')},
        devices=[dict(platform=d.platform,kind=d.device_kind) for d in devices],
        method_sources={str(p):sha(p) for p in Path('src/combine_jax/methods').glob('*.py')})
    monitor = Resources(args.backend == 'gpu', args.output)
    start = time.perf_counter()
    save_json(args.output, report)
    try:
        model = TensorBundle.load(args.model)
        result = run_expected_model(model, args.poi, r_max=args.r_max,
            max_iterations=args.fit_max_iterations, compiler_memory_limit_gib=args.compiler_memory_limit_gib,
            options=LimitOptions(confidence_level=args.limit_cl, strict_bounds=args.limit_strict_bounds))
        report.update(result, operation='limit')
        report['capture_policy'] = 'One limit calculation; no repeated limit fits or uncertainty calculation.'
        report['jax_allocator'] = [d.memory_stats() for d in devices]
    except BaseException as exc:
        report.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        monitor.close()
        report.update(resources=monitor.snapshot(), total_wall_seconds=time.perf_counter()-start,
                      root_imported='ROOT' in sys.modules)
        save_json(args.output, report)
    print(report['status'], report['limits'], flush=True)
