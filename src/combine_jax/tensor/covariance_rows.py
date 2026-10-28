"""Selected inverse-Hessian rows using exact HVPs and device-side CG.

No dense Hessian, inverse approximation, or saved linearization is constructed.
CG assumes positive-definite free curvature. Residual checks and encountered
nonpositive curvature are reported; success does not prove global definiteness
or certify an uncertainty at a nonsmooth point or away from a minimum.
"""
import math


STATUS = {0: 'residual_converged', 1: 'iteration_limit',
          2: 'nonpositive_curvature', 3: 'nonfinite',
          4: 'invalid_coordinates', 5: 'true_residual_failed'}


def make_covariance_rows(function, *, max_iterations=1000,
                         relative_tolerance=1e-8, absolute_tolerance=0., recompute_point=False):
    """Build a JIT-compatible selected-row solver with runtime coordinates.

    Call ``solve(theta, arrays, free_indices, scale, columns)``. ``columns``
    indexes positions in ``free_indices``, not the full physical parameter
    vector. ``scale`` contains positive coordinate scales, not uncertainties.
    Model/point/coordinate arrays are runtime inputs. Requested rows are solved
    sequentially inside ``lax.scan`` and CG runs in ``lax.while_loop``.

    The scaled system is (S H_free S) x = e_j. Physical inverse row j is
    S_jj * (S x), padded with exact zeros at fixed indices. Failed solves return
    NaN rows and an explicit status. The final residual is recomputed by an
    exact AD HVP, independently of the CG recurrence. Iteration products and
    the final residual use the same HVP body in a two-phase device loop; no
    second large AD graph is emitted for the residual check. No regularization,
    eigenvalue repair, preconditioner, or host iteration is used.

    ``recompute_point`` pairs the fixed point with each changing direction in
    an optimization barrier before AD. This discourages XLA from hoisting large
    point-dependent intermediates across CG iterations. It changes the memory/
    recomputation tradeoff, not the point, direction, or derivative formula.
    """
    if type(max_iterations) is not int or max_iterations<1:
        raise ValueError('max_iterations must be a positive integer')
    if type(recompute_point) is not bool:raise ValueError('recompute_point must be boolean')
    if not all(math.isfinite(v) and v>=0 for v in (relative_tolerance,absolute_tolerance)) or not relative_tolerance+absolute_tolerance>0:
        raise ValueError('Residual tolerances must be finite, nonnegative and not both zero')
    import jax
    import jax.numpy as jnp

    def solve(theta, arrays, free_indices, scale, columns):
        if theta.ndim!=1 or theta.dtype!=jnp.float64:
            raise ValueError('Covariance rows require a float64 parameter vector')
        if free_indices.ndim!=1 or not free_indices.size or not jnp.issubdtype(free_indices.dtype,jnp.integer):
            raise ValueError('free_indices must be a nonempty integer vector')
        if scale.shape!=free_indices.shape or scale.dtype!=jnp.float64:
            raise ValueError('scale must be float64 in free-coordinate order')
        if columns.ndim!=1 or not columns.size or not jnp.issubdtype(columns.dtype,jnp.integer):
            raise ValueError('columns must be a nonempty integer vector of free positions')
        count=free_indices.size
        sorted_indices=jnp.sort(free_indices)
        valid=(jnp.all(jnp.isfinite(theta)) & jnp.all(jnp.isfinite(scale)) & jnp.all(scale>0)
               & jnp.all(free_indices>=0) & jnp.all(free_indices<theta.size)
               & jnp.all(sorted_indices[1:]>sorted_indices[:-1]))
        gradient=jax.grad(lambda point:function(point,arrays))

        def multiply(vector):
            direction=jnp.zeros_like(theta).at[free_indices].set(scale*vector)
            point=theta
            if recompute_point:point,direction=jax.lax.optimization_barrier((point,direction))
            _,product=jax.jvp(gradient,(point,),(direction,))
            return scale*product[free_indices]

        def one_column(column):
            rhs=(jnp.arange(count)==column).astype(jnp.float64)
            target=jnp.maximum(absolute_tolerance,relative_tolerance*jnp.linalg.norm(rhs))
            admissible=valid & (column>=0) & (column<count)
            initial=dict(x=jnp.zeros_like(scale),r=rhs,p=rhs,rr=jnp.dot(rhs,rhs),
                iteration=jnp.int32(0),status=jnp.where(admissible,jnp.int32(0),jnp.int32(4)),
                check=jnp.dot(rhs,rhs)<=target**2,done=~admissible,
                residual_norm=jnp.float64(jnp.nan),hvp_evaluations=jnp.int32(0))

            def iteration(state,product):
                curvature=jnp.dot(state['p'],product)
                finite=jnp.all(jnp.isfinite(product)) & jnp.isfinite(curvature)
                positive=curvature>0
                alpha=state['rr']/jnp.where(finite & positive,curvature,1.)
                x=state['x']+alpha*state['p'];r=state['r']-alpha*product;rr=jnp.dot(r,r)
                p=r+(rr/state['rr'])*state['p']
                finite=finite & jnp.isfinite(alpha) & jnp.isfinite(rr) & jnp.all(jnp.isfinite(x)) & jnp.all(jnp.isfinite(p))
                accepted=finite & positive
                updated=dict(state,x=jnp.where(accepted,x,state['x']),r=jnp.where(accepted,r,state['r']),
                    p=jnp.where(accepted,p,state['p']),rr=jnp.where(accepted,rr,state['rr']),
                    iteration=state['iteration']+1,
                    status=jnp.where(~finite,jnp.int32(3),jnp.where(positive,jnp.int32(0),jnp.int32(2))))
                updated['check']=(updated['status']!=0) | (updated['iteration']>=max_iterations) | (updated['rr']<=target**2)
                return updated

            def check_residual(state,product):
                norm=jnp.linalg.norm(rhs-product)
                status=jnp.where(state['status']!=0,state['status'],
                    jnp.where(~jnp.isfinite(norm),jnp.int32(3),jnp.where(norm<=target,jnp.int32(0),
                    jnp.where(state['iteration']>=max_iterations,jnp.int32(1),jnp.int32(5)))))
                return dict(state,status=status,done=jnp.bool_(True),residual_norm=norm)

            def step(state):
                product=multiply(jnp.where(state['check'],state['x'],state['p']))
                updated=jax.lax.cond(state['check'],lambda:check_residual(state,product),
                    lambda:iteration(state,product))
                return dict(updated,hvp_evaluations=state['hvp_evaluations']+1)

            state=jax.lax.while_loop(lambda s:~s['done'],step,initial)
            index=jnp.clip(column,0,count-1)
            physical=scale*state['x']*scale[index];status=state['status']
            status=jnp.where((status==0) & ~jnp.all(jnp.isfinite(physical)),jnp.int32(3),status)
            status=jnp.where((status==0) & (physical[index]<=0),jnp.int32(2),status)
            row=jnp.zeros_like(theta).at[free_indices].set(physical)
            return dict(rows=jnp.where(status==0,row,jnp.full_like(theta,jnp.nan)),
                status=status,iterations=state['iteration'],hvp_evaluations=state['hvp_evaluations'],
                residual_norm=state['residual_norm'],recurrence_residual_norm=jnp.sqrt(state['rr']),residual_target=target)

        def scan(_,column):
            return None,one_column(column)
        return jax.lax.scan(scan,None,columns)[1]
    return solve
