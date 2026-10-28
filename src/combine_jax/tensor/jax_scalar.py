"""JAX adapter for framework-neutral scalar batches; model arrays stay arguments."""
from .scalar import operation, validate_program


def make_function(bundle):
    import jax
    import jax.numpy as jnp
    import jax.scipy.special as special

    jax.config.update('jax_enable_x64', True)
    validate_program(bundle)
    # No bundle/array closure: compilation sees only the small operation schedule.
    schedule = tuple(dict(batch) for batch in bundle.metadata['schedule'])

    def evaluate(theta, arrays):
        if theta.shape != arrays['parameter_initial'].shape:
            raise ValueError('Parameter vector shape mismatch')
        values = arrays['initial_values'].at[arrays['parameter_slots']].set(theta[arrays['parameter_indices']])
        for batch in schedule:
            arguments = values[arrays[batch['inputs']]]
            result = operation(batch['operation'], arguments, arrays[batch['mask']], jnp, special)
            values = values.at[arrays[batch['outputs']]].set(result)
        valid = jnp.all(jnp.where(arrays['parameter_fixed'], theta == arrays['parameter_initial'], True))
        return jnp.where(valid, values[arrays['output_slots']], jnp.nan)

    return evaluate
