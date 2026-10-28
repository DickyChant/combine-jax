"""Opt-in persistent JAX compilation caching for neutral model structures."""
from pathlib import Path


def configure_model_cache(model,directory):
    """Configure before JIT dispatch, with no concurrent compilation in progress.

    The caller owns and trusts the cache directory. The neutral structure key
    separates layouts; JAX's executable key additionally covers lowered code,
    argument shapes/dtypes, compiler options and device/version compatibility.
    Buffer values remain runtime arguments and do not select a cache namespace.
    This persists compiled executables, not Python traces or model data.
    Auxiliary XLA caches are disabled: JAX 0.10.1 includes their absolute paths
    in its executable key, preventing reuse when batch scratch paths change.
    """
    import jax
    from jax.experimental.compilation_cache import compilation_cache
    namespace=Path(directory).expanduser().resolve()/model.structure_key
    namespace.mkdir(parents=True,exist_ok=True,mode=0o700)
    # Reset only JAX's disk-cache handle so sequential models cannot silently
    # keep using the previous model's namespace. No cache files are deleted.
    compilation_cache.reset_cache()
    compilation_cache.set_cache_dir(str(namespace))
    jax.config.update('jax_enable_compilation_cache',True)
    # Keep JAX's complete compatibility key unchanged. Its executable cache is
    # sufficient for warm jobs; per-fusion autotuning files are optional.
    jax.config.update('jax_persistent_cache_enable_xla_caches','none')
    jax.config.update('jax_persistent_cache_min_compile_time_secs',0.)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes',-1)
    jax.config.update('jax_raise_persistent_cache_errors',True)
    return dict(directory=str(namespace),structure_key=model.structure_key,jax_version=jax.__version__,
        minimum_compile_seconds=0.,minimum_entry_bytes=-1,
        auxiliary_xla_caches='none',
        key_policy='Neutral structure namespace plus JAX executable compatibility key; numeric buffer contents excluded.')
