"""Binned likelihood pullback with streamed yield scores and local shape AD.

Only the scalar input is differentiated. Neutral model arrays are immutable
runtime data. The primal density arena is retained once; per-operator AD tapes
are reconstructed locally in reverse order instead of retaining every gather.
"""


def make_structural_pullback(evaluate, densities, histogram_queries, program, nll_program,
                             *, row_chunk_size=1024, sequential_shapes=False):
    import jax
    import jax.numpy as jnp
    from .binned_channels import operation, nll_batch, program_fields

    if type(row_chunk_size) is not int or row_chunk_size < 1:
        raise ValueError('row_chunk_size must be a positive integer')
    if type(sequential_shapes) is not bool:
        raise ValueError('sequential_shapes must be boolean')

    def gather(values, indices):
        return values.at[indices].get(mode='promise_in_bounds', wrap_negative_indices=False)

    def accumulate(values, indices, increments):
        # Child/scalar aliases can repeat, including across channels.
        return values.at[indices].add(increments, mode='promise_in_bounds', wrap_negative_indices=False)

    def local_fields(b, fields):
        result = dict(b)
        for field in fields:
            if field in result:
                indices = result[field]
                result[field] = jnp.arange(indices.size, dtype=jnp.int32).reshape(indices.shape)
        return result

    def shape_pullback(op, values, scalars, b, value_bar, scalar_bar):
        local = local_fields(b, ('children', 'coefs'))
        children = gather(values, b['children']).reshape(-1) if 'children' in b else jnp.empty(0, dtype=jnp.float64)
        coefs = scalars[b['coefs']].reshape(-1) if 'coefs' in b else jnp.empty(0, dtype=jnp.float64)
        def term(child_values, coefficients):
            return operation(op, child_values, coefficients, local, jnp)
        _, pullback = jax.vjp(term, children, coefs)
        children_bar, coefs_bar = pullback(gather(value_bar, b['outputs']))
        if 'children' in b:
            value_bar = accumulate(value_bar, b['children'], children_bar.reshape(b['children'].shape))
        if 'coefs' in b:
            scalar_bar = accumulate(scalar_bar, b['coefs'], coefs_bar.reshape(b['coefs'].shape))
        return value_bar, scalar_bar

    def mixture_pullback(values, scalars, b, value_bar):
        """Accumulate process scores over rows without a process×row AD tape."""
        coefs = jnp.where(b['coef_mask'], scalars[b['coefs']], 0.)
        total = jnp.sum(coefs, axis=1)
        rows = b['outputs'].shape[1]
        if not rows:
            return value_bar, jnp.zeros_like(coefs)
        width = min(row_chunk_size, rows)
        count = (rows + width - 1) // width

        def step(index, state):
            bar, score = state
            row = index * width + jnp.arange(width, dtype=jnp.int32)
            valid = row < rows
            row = jnp.minimum(row, rows - 1)
            children = b['children'][:, :, row]
            shape = jnp.where(b['child_mask'][:, :, None], gather(values, children), 0.)
            output = b['outputs'][:, row]
            cotangent = jnp.where(valid[None, :], gather(bar, output), 0.) / total[:, None]
            mixture = gather(values, output)
            score = score + jnp.sum(cotangent[:, None, :] * (shape - mixture[:, None, :]), axis=2)
            increments = jnp.where(b['child_mask'][:, :, None],
                cotangent[:, None, :] * coefs[:, :, None], 0.)
            return accumulate(bar, children, increments), score

        value_bar, score = jax.lax.fori_loop(0, count, step, (value_bar, jnp.zeros_like(coefs)))
        return value_bar, jnp.where(b['coef_mask'], score, 0.)

    @jax.custom_vjp
    def function(scalars, arrays):
        return evaluate(scalars, arrays)

    def forward(scalars, arrays):
        values = densities(scalars, arrays)
        output = arrays['initial_nll']
        for batch in nll_program:
            b = program_fields(batch, arrays)
            output = output.at[b['outputs']].set(nll_batch(values, scalars, b, jnp), unique_indices=True)
        return output, (scalars, arrays, values)

    def backward(residual, output_bar):
        scalars, arrays, values = residual
        scalar_bar, value_bar = jnp.zeros_like(scalars), jnp.zeros_like(values)
        for batch in nll_program:
            b = program_fields(batch, arrays)
            local = local_fields(b, ('densities', 'rates'))
            def term(density, rates):
                return nll_batch(density, rates, local, jnp)
            _, pullback = jax.vjp(term, gather(values, b['densities']).reshape(-1),
                                  scalars[b['rates']].reshape(-1))
            density_bar, rate_bar = pullback(output_bar[b['outputs']])
            value_bar = accumulate(value_bar, b['densities'], density_bar.reshape(b['densities'].shape))
            scalar_bar = accumulate(scalar_bar, b['rates'], rate_bar.reshape(b['rates'].shape))

        for batch in reversed(program):
            b = program_fields(batch, arrays)
            op = batch['operation']
            if op == 'mixture_yields':
                value_bar, score = mixture_pullback(values, scalars, b, value_bar)
                scalar_bar = accumulate(scalar_bar, b['coefs'], score)
                continue
            if sequential_shapes and b['outputs'].shape[0] > 1:
                # Same-structure operator instances share a loop body. Only
                # the current instance's local shape AD tape is materialized;
                # both cotangent accumulators stay on device throughout.
                def step(index, state):
                    single = {k: jax.lax.dynamic_slice_in_dim(v, index, 1, axis=0) for k, v in b.items()}
                    return shape_pullback(op, values, scalars, single, *state)
                value_bar, scalar_bar = jax.lax.fori_loop(0, b['outputs'].shape[0], step, (value_bar, scalar_bar))
            else:
                value_bar, scalar_bar = shape_pullback(op, values, scalars, b, value_bar, scalar_bar)

        # Exact normalized histogram and query derivatives; repeated query/bin
        # destinations require accumulation, not an overwrite or a uniqueness hint.
        fast, pullback = jax.vjp(lambda s: histogram_queries(s, arrays), scalars)
        fast_bar = accumulate(jnp.zeros_like(fast), arrays['fast_inputs'],
                              gather(value_bar, arrays['fast_outputs']))
        scalar_bar = scalar_bar + pullback(fast_bar)[0]
        return scalar_bar, None

    function.defvjp(forward, backward)
    return function
