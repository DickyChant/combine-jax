"""Float64 BFGS rank-two update with row-blocked matrix intermediates.

This is the dense BFGS formula, including its final symmetrization, not a
limited-memory approximation. The previous and next matrices remain dense.
The fit driver uses this only with an explicit positive update block size.
"""


def deferred_identity(dependency, *, block_size=128):
    """Construct an exact identity after a runtime vector has been evaluated.

    The row loop overwrites every element, even when dependency contains NaN
    or infinities. Its arbitrary initial contents make the allocation depend
    on the completed initial gradient instead of overlapping its working set.
    """
    import jax
    import jax.numpy as jnp
    if type(block_size) is not int or block_size<=0:
        raise ValueError('block_size must be a positive integer')
    if dependency.ndim!=1 or not dependency.size or dependency.dtype!=jnp.float64:
        raise ValueError('Dependency must be a nonempty float64 vector')
    n=dependency.shape[0];width=min(n,block_size);columns=jnp.arange(n)
    initial=jnp.broadcast_to(jnp.sum(dependency),(n,n))
    def body(i,matrix):
        first=jnp.minimum(i*width,n-width);rows=first+jnp.arange(width)
        values=(rows[:,None]==columns[None,:]).astype(dependency.dtype)
        return jax.lax.dynamic_update_slice_in_dim(matrix,values,first,axis=0)
    return jax.lax.fori_loop(0,(n+width-1)//width,body,initial)


def blocked_bfgs_update(matrix, step, change, *, block_size=128,
                        blocked=None, reset=False, update=True):
    """Return the updated inverse, relative L1 change and finiteness.

    The caller has already checked the secant curvature. ``blocked`` decouples
    bound-blocked rows/columns with unit diagonal; ``reset`` selects the identity.
    These base-matrix operations are applied in each slice, avoiding another
    dense matrix. ``update=False`` returns that base without a secant update.
    Reduction association may differ from the dense
    expression. No tolerance, precision, matrix element or symmetrization is
    dropped. Both old matrix orientations are read, so asymmetric input is
    handled in the same way as the existing dense update.
    """
    import jax
    import jax.numpy as jnp
    if type(block_size) is not int or block_size <= 0:
        raise ValueError('block_size must be a positive integer')
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.dtype != jnp.float64:
        raise ValueError('base must be a square float64 matrix')
    n = matrix.shape[0]
    if n == 0 or any(v.shape != (n,) or v.dtype != matrix.dtype for v in (step, change)):
        raise ValueError('Nonempty float64 step and change must match base')
    blocked = jnp.zeros(n, dtype=bool) if blocked is None else jnp.asarray(blocked)
    reset, update = jnp.asarray(reset), jnp.asarray(update)
    if blocked.shape != (n,) or blocked.dtype != jnp.bool_ or any(v.shape or v.dtype != jnp.bool_ for v in (reset,update)):
        raise ValueError('blocked must be a bool vector; reset and update must be bool scalars')
    width = min(block_size, n)
    hy = jnp.where(reset, change,
        jnp.where(blocked, change, matrix @ jnp.where(blocked, 0., change)))
    ys = jnp.where(update, jnp.dot(step, change), 1.)
    factor = (ys+jnp.dot(change, hy))/(ys*ys)
    columns = jnp.arange(n)

    def body(i, state):
        result, delta, magnitude, finite = state
        # The last slice may overlap the preceding block. Its values are
        # identical, and the overlap is excluded from all reductions.
        first = jnp.minimum(i*width, n-width)
        rows = first+jnp.arange(width)
        old = jax.lax.dynamic_slice_in_dim(matrix, first, width, axis=0)
        old_transpose = jax.lax.dynamic_slice_in_dim(matrix, first, width, axis=1).T
        mask = reset | blocked[rows,None] | blocked[None,:]
        identity = (rows[:,None] == columns[None,:]).astype(matrix.dtype)
        old = jnp.where(mask, identity, old)
        old_transpose = jnp.where(mask, identity, old_transpose)
        s = step[rows]
        h = hy[rows]
        row_update = old+factor*(s[:,None]*step[None,:])
        row_update -= (h[:,None]*step[None,:]+s[:,None]*hy[None,:])/ys
        transposed = old_transpose+factor*(step[None,:]*s[:,None])
        transposed -= (hy[None,:]*s[:,None]+step[None,:]*h[:,None])/ys
        updated = jnp.where(update, .5*(row_update+transposed), old)
        result = jax.lax.dynamic_update_slice_in_dim(result, updated, first, axis=0)
        include = rows[:,None] >= i*width
        delta += jnp.sum(jnp.where(include, jnp.abs(updated-old), 0.))
        magnitude += jnp.sum(jnp.where(include, jnp.abs(updated), 0.))
        finite &= jnp.all(jnp.isfinite(updated))
        return result, delta, magnitude, finite

    # Every element is overwritten by the row loop. Tie the otherwise arbitrary
    # initial contents to the new gradient so XLA cannot hoist this dense output
    # allocation before gradient evaluation, when its working set is largest.
    initial = (jnp.broadcast_to(change[0], matrix.shape), jnp.float64(0.), jnp.float64(0.), jnp.array(True))
    result, delta, magnitude, finite = jax.lax.fori_loop(0, (n+width-1)//width, body, initial)
    relative = delta/jnp.maximum(magnitude, jnp.finfo(matrix.dtype).tiny)
    return result, relative, finite
