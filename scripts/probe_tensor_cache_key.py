"""Capture the unmodified JAX cache key before cache lookup or compilation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys


class KeyCaptured(BaseException):
    """Stop before compiling or loading an executable."""


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=('cpu', 'gpu'), required=True)
    parser.add_argument('--producer-root', type=Path, default=Path('.'))
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--model', type=Path)
    inputs.add_argument('--toy', action='store_true')
    parser.add_argument('--operation', choices=('nll', 'value_gradient'), default='nll')
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--xla-caches', choices=('inherited', 'none'), default='inherited',
                        help='Control optional auxiliary XLA caches; executable cache remains enabled')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    os.environ['JAX_PLATFORMS'] = 'cuda' if args.backend == 'gpu' else 'cpu'
    os.environ['JAX_ENABLE_X64'] = 'true'
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[key] = '1'
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:8])
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    from jax._src import cache_key
    from jax._src.lib.mlir import ir
    from jax.experimental.compilation_cache import compilation_cache
    devices = jax.devices()
    assert devices and all(d.platform == args.backend for d in devices)
    assert jax.config.x64_enabled
    assert jax.__version__ in ('0.10.1', '0.12.0'), 'Private diagnostic API has not been checked for this version'
    print('verified devices', devices, flush=True)
    producer = args.producer_root.resolve()
    assert 'combine_jax' not in sys.modules
    sys.path.insert(0, str(producer/'src'))
    sources = {}
    model_hashes = None
    if args.toy:
        def function(theta, arrays):
            return jnp.sum(theta*arrays['scale'])
        function.__name__ = 'nll'
        theta = np.array([1., 2.])
        arrays = dict(scale=np.array([3., 4.]))
    else:
        from combine_jax.tensor.bundle import TensorBundle
        from combine_jax.tensor.model import make_model_function
        from combine_jax.tensor.parameter_info import model_parameter_info
        assert Path(sys.modules['combine_jax.tensor.model'].__file__).resolve() == producer/'src/combine_jax/tensor/model.py'
        sources = {str(p.relative_to(producer)):sha(p) for p in (producer/'src/combine_jax/tensor').glob('*.py')}
        model = TensorBundle.load(args.model)
        theta = model_parameter_info(model).arrays['initial']
        arrays = {n:a for n,a in model.arrays.items() if not n.startswith('parameter_info__')}
        function = make_model_function(model)
        model_hashes = {n:sha(args.model/n) for n in ('model.json', 'arrays.npz')}
    compilation_cache.reset_cache()
    compilation_cache.set_cache_dir(str(args.cache.resolve()))
    if args.xla_caches == 'none':
        jax.config.update('jax_persistent_cache_enable_xla_caches', 'none')
    jax.config.update('jax_enable_compilation_cache', True)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', -1)
    arguments = jax.device_put((theta, arrays))
    jax.block_until_ready(arguments)
    if args.operation == 'value_gradient':
        function = jax.value_and_grad(function)
    report = dict(status='probing', script_sha256=sha(__file__), jax_version=jax.__version__,
        jaxlib_version=jaxlib.__version__, cache_key_source_sha256=sha(cache_key.__file__),
        producer_root=str(producer), sources=sources, model_sha256=model_hashes,
        backend=args.backend, operation=args.operation, float64=True, toy=args.toy,
        devices=[dict(kind=d.device_kind, platform=d.platform, id=d.id) for d in devices],
        environment={k:os.environ.get(k) for k in ('COMBINE_JAX_LCG_VIEW','CUDA_VISIBLE_DEVICES','XLA_FLAGS',
                    'JAX_COMPILATION_CACHE_INCLUDE_METADATA_IN_KEY')},
        metadata_in_key=bool(jax.config.jax_compilation_cache_include_metadata_in_key),
        auxiliary_xla_caches=jax.config.jax_persistent_cache_enable_xla_caches,
        components=[], scope='Diagnostic only: original JAX key generation is unchanged. Stop at the outer key before cache lookup or executable compilation. No model evaluation, numerical agreement, or timing claim.')
    original_get = cache_key.get
    original_log = cache_key._log_cache_key_hash
    active = False

    def record_component(cumulative, name, hashfn):
        if active:
            digest = hashlib.sha256(); hashfn(digest)
            report['components'].append(dict(name=name, sha256=digest.hexdigest(), cumulative_sha256=cumulative.hexdigest()))

    def capture(module, devices, compile_options, backend, *extra, **kwargs):
        nonlocal active
        name = ir.StringAttr(module.operation.attributes['sym_name']).value
        if name != 'jit_nll':
            return original_get(module, devices, compile_options, backend, *extra, **kwargs)
        active = True
        key = original_get(module, devices, compile_options, backend, *extra, **kwargs)
        active = False
        report.update(status='captured_before_compile', key=key, platform_version=backend.platform_version,
                      compile_options_serialized_sha256=hashlib.sha256(compile_options.SerializeAsString()).hexdigest())
        debug = compile_options.executable_build_options.debug_options
        report['compile_options_cache_paths'] = {
            name: getattr(debug, name) for name in
            ('xla_gpu_kernel_cache_file', 'xla_gpu_per_fusion_autotune_cache_dir')}
        raise KeyCaptured()

    cache_key.get = capture
    cache_key._log_cache_key_hash = record_component
    try:
        jax.jit(function).lower(*arguments).compile()
    except KeyCaptured:
        pass
    finally:
        cache_key.get = original_get
        cache_key._log_cache_key_hash = original_log
    assert report['status'] == 'captured_before_compile'
    assert len(report['components']) == 8 and 'ROOT' not in sys.modules
    assert report['key'].endswith(report['components'][-1]['cumulative_sha256'])
    if not args.toy:
        assert sources == {n:sha(producer/n) for n in sources}
    report['root_imported'] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(report['status'], report['key'], flush=True)


if __name__ == '__main__':
    main()
