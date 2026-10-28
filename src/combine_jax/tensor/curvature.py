"""Exact AD curvature assembled in bounded column batches on the device."""


def make_value_gradient_hvps(function, *, recompute_point=False):
    """Return NLL, gradient and exact products for runtime direction rows.

    Each product uses forward-over-reverse AD in the same invocation. No dense
    Hessian or linearization tape is retained between invocations. Directions
    have shape (batch, parameters); the caller can stream batches of width one.
    Model arrays remain immutable runtime arguments, and all arithmetic is f64.
    """
    import jax
    import jax.numpy as jnp
    if type(recompute_point) is not bool:raise ValueError('recompute_point must be boolean')

    def evaluate(theta, arrays, directions):
        if theta.ndim != 1 or theta.dtype != jnp.float64:
            raise ValueError('HVPs require a one-dimensional float64 parameter vector')
        if directions.ndim != 2 or directions.shape[1] != theta.size or not directions.shape[0] or directions.dtype != jnp.float64:
            raise ValueError('HVP directions must be nonempty float64 rows in parameter order')
        def objective(x):
            value = function(x, arrays)
            return value, value
        gradient_function = jax.grad(objective, has_aux=True)
        def product(direction):
            point=theta
            if recompute_point:point,direction=jax.lax.optimization_barrier((point,direction))
            gradient, hvp, value = jax.jvp(gradient_function, (point,), (direction,), has_aux=True)
            return value, gradient, hvp
        values, gradients, products = jax.vmap(product)(directions)
        return values[0], gradients[0], products
    return evaluate


def make_value_gradient_hessian(function,*,column_batch_size=4,linearization='reuse'):
    """Return a JIT-compatible (theta, arrays) -> (NLL, gradient, Hessian).

    Linearize the gradient once, then apply its linear map to batches of unit
    vectors in a device while loop. Neither parameters nor matrix blocks cross
    the host boundary. The returned matrix uses H[row, column] orientation.
    The caller controls JIT compilation and synchronization. ``linearization``
    defaults to reusing the saved gradient intermediates. ``'recompute'`` instead
    checkpoints the gradient before linearizing it, recomputing its intermediates
    inside each column batch. This changes the AD memory/time tradeoff, not the
    derivatives; whether it saves memory must be measured for the compiled model.
    """
    import jax
    import jax.numpy as jnp
    if type(column_batch_size) is not int or column_batch_size<1:
        raise ValueError('column_batch_size must be a positive integer')
    if linearization not in ('reuse','recompute'):
        raise ValueError('linearization must be reuse or recompute')
    width=column_batch_size
    def evaluate(theta,arrays):
        if theta.ndim!=1 or theta.dtype!=jnp.float64:
            raise ValueError('Curvature requires a one-dimensional float64 parameter vector')
        def objective(x):
            value=function(x,arrays)
            return value,value
        gradient_function=jax.grad(objective,has_aux=True)
        if linearization=='recompute':gradient_function=jax.checkpoint(gradient_function)
        gradient,linear,value=jax.linearize(gradient_function,theta,has_aux=True)
        n=theta.shape[0]
        if not n:return value,gradient,jnp.empty((0,0),dtype=theta.dtype)
        count=(n+width-1)//width
        def body(state):
            index,matrix=state
            columns=index*width+jnp.arange(width,dtype=jnp.int32)
            directions=(jnp.arange(n,dtype=jnp.int32)[None,:]==columns[:,None]).astype(theta.dtype)
            block=jax.vmap(linear)(directions)
            return index+1,jax.lax.dynamic_update_slice(matrix,block,(index*width,jnp.int32(0)))
        initial=(jnp.int32(0),jnp.zeros((count*width,n),dtype=theta.dtype))
        _,matrix=jax.lax.while_loop(lambda state:state[0]<count,body,initial)
        return value,gradient,matrix[:n,:].T
    return evaluate
