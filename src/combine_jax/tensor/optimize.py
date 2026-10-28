"""Bounded BFGS with all numerical iterations inside JAX control flow."""
import math


OPTIMIZER_STATUS = {0: 'converged', 1: 'iteration_limit', 2: 'line_search_failed',
                    3: 'nonfinite', 4: 'invalid_initial_or_bounds', 5: 'edm_validation_failed'}


def make_bfgs(function, *, max_iterations=1000, max_linesearch=40,
              gradient_tolerance=1e-6,armijo_roundoff_multiplier=8.,
              stopping_criterion='projected_gradient', edm_tolerance=1e-4):
    """Return a JIT-compatible minimizer in caller-scaled free coordinates.

    Convergence uses the box-projected gradient by default. Optional ``edm``
    stopping compares .5 * projected_gradient.T @ inverse_BFGS @
    projected_gradient, multiplied by (1 + 3 * inverse_uncertainty), with a
    raw objective EDM threshold. The uncertainty averages the previous estimate
    with the relative L1 change of the inverse matrix, initially one. This is a BFGS
    estimate, not Minuit's implementation or an exact-Hessian certificate.
    At least one accepted secant update is required before using the estimate,
    except at an exactly stationary point. This prevents accepting a small
    gradient solely because the initial identity metric has the wrong scale.
    Feasible Armijo backtracking uses NLL-only trials; derivatives are evaluated
    at accepted points. The inverse BFGS approximation is a search metric, never
    a reported covariance. Equal-bound coordinates must be removed by the caller.
    Model buffers, starts and bounds are runtime arguments.
    Armijo comparisons allow a recorded multiple of eps*(1+abs(NLL)) to avoid
    rejecting unresolved decreases. This never changes the selected threshold.
    Bound-blocked coordinates are removed from both the search metric and its
    secant update. Their gradient changes must not enter free-space curvature.
    """
    import jax
    import jax.numpy as jnp
    if type(max_iterations) is not int or max_iterations < 0:
        raise ValueError('max_iterations must be a nonnegative integer')
    if type(max_linesearch) is not int or max_linesearch < 1:
        raise ValueError('max_linesearch must be a positive integer')
    if isinstance(gradient_tolerance, bool) or not math.isfinite(gradient_tolerance) or gradient_tolerance <= 0:
        raise ValueError('gradient_tolerance must be finite and positive')
    if stopping_criterion not in ('projected_gradient', 'edm'):
        raise ValueError('stopping_criterion must be projected_gradient or edm')
    if isinstance(edm_tolerance, bool) or not math.isfinite(edm_tolerance) or edm_tolerance <= 0:
        raise ValueError('edm_tolerance must be finite and positive')
    if isinstance(armijo_roundoff_multiplier,bool) or not math.isfinite(armijo_roundoff_multiplier) or armijo_roundoff_multiplier<0:
        raise ValueError('armijo_roundoff_multiplier must be finite and nonnegative')
    vg = jax.value_and_grad(function)

    def optimize(initial, arrays, lower, upper, initial_inverse=None,
                 initial_inverse_uncertainty=1., iteration_budget=None):
        if initial.ndim != 1 or initial.dtype != jnp.float64:
            raise ValueError('Optimization requires a one-dimensional float64 vector')
        if any(b.shape != initial.shape or b.dtype != initial.dtype for b in (lower, upper)):
            raise ValueError('Bounds must match the initial vector shape and dtype')
        n = initial.shape[0]
        identity = jnp.eye(n, dtype=initial.dtype)
        seed_inverse = identity if initial_inverse is None else initial_inverse
        if seed_inverse.shape != identity.shape or seed_inverse.dtype != initial.dtype:
            raise ValueError('Initial inverse must match the free-coordinate matrix shape and dtype')
        budget = max_iterations if iteration_budget is None else jnp.minimum(max_iterations,iteration_budget)
        valid_initial = (jnp.all(jnp.isfinite(initial)) & jnp.all(lower <= initial)
                         & jnp.all(initial <= upper) & jnp.all(lower < upper))

        def blocked_coordinates(x, g):
            eps = 32 * jnp.finfo(initial.dtype).eps * (1 + jnp.abs(x))
            return ((x <= lower + eps) & (g > 0)) | ((x >= upper - eps) & (g < 0))

        def projected(x, g):
            return jnp.where(blocked_coordinates(x, g), 0., g)

        value, gradient = vg(initial, arrays)
        state = dict(x=initial, value=value, gradient=gradient, inverse=seed_inverse,
                     iteration=jnp.int32(0), function_evaluations=jnp.int32(1),
                     gradient_evaluations=jnp.int32(1), line_search_evaluations=jnp.int32(0),
                     resets=jnp.int32(0), skipped_updates=jnp.int32(0),
                     inverse_uncertainty=jnp.float64(initial_inverse_uncertainty),
                     roundoff_accepts=jnp.int32(0),maximum_roundoff_allowance=jnp.float64(0.),
                     line_ok=jnp.array(True),
                     finite=jnp.isfinite(value) & jnp.all(jnp.isfinite(gradient)) & jnp.all(jnp.isfinite(seed_inverse)))

        def converged(s):
            pg = projected(s['x'], s['gradient'])
            if stopping_criterion == 'projected_gradient':
                return jnp.max(jnp.abs(pg), initial=0.) <= gradient_tolerance
            edm = .5*jnp.dot(pg, s['inverse']@pg)*(1+3*s['inverse_uncertainty'])
            updated = (s['iteration'] > s['skipped_updates']) | (initial_inverse_uncertainty == 0.)
            return jnp.all(pg == 0.) | (updated & jnp.isfinite(edm) & (edm >= 0.) & (edm < edm_tolerance))

        def condition(s):
            return valid_initial & (s['iteration'] < budget) & s['line_ok'] & s['finite'] & ~converged(s)

        def body(s):
            x, g = s['x'], s['gradient']
            pg = projected(x, g)
            blocked = blocked_coordinates(x, g)
            # Decouple currently blocked rows/columns; reset their diagonal so
            # a coordinate that later becomes free can move immediately.
            metric = jnp.where(blocked[:, None] | blocked[None, :], identity, s['inverse'])
            p = -metric @ pg
            eps = 32 * jnp.finfo(initial.dtype).eps * (1 + jnp.abs(x))
            p = jnp.where(((x <= lower + eps) & (p < 0)) | ((x >= upper - eps) & (p > 0)), 0., p)
            descent = jnp.dot(g, p) < 0
            p = jnp.where(descent, p, -pg)
            base = jnp.where(descent, metric, identity)
            safe = jnp.where(p != 0, p, 1.)
            bound = jnp.where(p > 0, (upper-x)/safe, jnp.where(p < 0, (lower-x)/safe, jnp.inf))
            alpha = jnp.minimum(1., jnp.min(bound, initial=jnp.inf))
            slope = jnp.dot(g, p)
            allowance=armijo_roundoff_multiplier*jnp.finfo(initial.dtype).eps*(1+jnp.abs(s['value']))
            search = dict(alpha=alpha, x=x, value=s['value'], accepted=jnp.array(False), evaluations=jnp.int32(0),roundoff_used=jnp.array(False))

            def search_condition(t):
                return (t['evaluations'] < max_linesearch) & ~t['accepted'] & (t['alpha'] > 0)

            def search_body(t):
                candidate = jnp.clip(x + t['alpha']*p, lower, upper)
                # A limiting step can round one ulp into the box. Preserve the
                # exact bound it reached before the active mask freezes it.
                candidate = jnp.where((p < 0) & (t['alpha'] == bound), lower, candidate)
                candidate = jnp.where((p > 0) & (t['alpha'] == bound), upper, candidate)
                v = function(candidate, arrays)
                strict=v-s['value']<=1e-4*t['alpha']*slope
                accepted = jnp.isfinite(v) & jnp.any(candidate!=x) & (v-s['value'] <= 1e-4*t['alpha']*slope+allowance)
                return dict(alpha=jnp.where(accepted, t['alpha'], .5*t['alpha']), x=candidate,
                            value=v, accepted=accepted, evaluations=t['evaluations']+1,roundoff_used=accepted&~strict)

            line = jax.lax.while_loop(search_condition, search_body, search)
            candidate = jnp.where(line['accepted'], line['x'], x)
            newvalue, newgradient = jax.lax.cond(line['accepted'], lambda q: vg(q, arrays),
                                                lambda q: (s['value'], g), candidate)
            step, change = candidate-x, jnp.where(blocked, 0., newgradient-g)
            ys = jnp.dot(step, change)
            adequate = (ys > 1e-18) & (ys > 1e-12*jnp.linalg.norm(step)*jnp.linalg.norm(change))

            def update(_):
                hy = base @ change
                result = base + (ys+jnp.dot(change, hy))/(ys*ys)*jnp.outer(step, step)
                result -= (jnp.outer(hy, step)+jnp.outer(step, hy))/ys
                return .5*(result+result.T)

            inverse = jax.lax.cond(adequate, update, lambda _: base, operand=None)
            uncertainty = s['inverse_uncertainty']
            if stopping_criterion == 'edm':
                relative_change = jnp.sum(jnp.abs(inverse-base))/jnp.maximum(jnp.sum(jnp.abs(inverse)),jnp.finfo(initial.dtype).tiny)
                previous_uncertainty = jnp.where(descent, uncertainty, 1.)
                uncertainty = jnp.where(adequate, .5*(previous_uncertainty+relative_change), previous_uncertainty)
            accepted = line['accepted'].astype(jnp.int32)
            return dict(x=candidate, value=newvalue, gradient=newgradient, inverse=inverse,
                        iteration=s['iteration']+1,
                        function_evaluations=s['function_evaluations']+line['evaluations']+accepted,
                        gradient_evaluations=s['gradient_evaluations']+accepted,
                        line_search_evaluations=s['line_search_evaluations']+line['evaluations'],
                        resets=s['resets']+(~descent).astype(jnp.int32),
                        skipped_updates=s['skipped_updates']+(~adequate).astype(jnp.int32),
                        inverse_uncertainty=uncertainty,
                        roundoff_accepts=s['roundoff_accepts']+line['roundoff_used'].astype(jnp.int32),
                        maximum_roundoff_allowance=jnp.maximum(s['maximum_roundoff_allowance'],allowance),
                        line_ok=line['accepted'], finite=jnp.isfinite(newvalue)
                        & jnp.all(jnp.isfinite(newgradient)) & jnp.all(jnp.isfinite(inverse)))

        result = jax.lax.while_loop(condition, body, state)
        status = jnp.where(~valid_initial, 4, jnp.where(~result['finite'], 3,
                 jnp.where(converged(result), 0, jnp.where(~result['line_ok'], 2, 1))))
        pg = projected(result['x'], result['gradient'])
        return dict(result, status=status, projected_gradient=pg,
                    estimated_edm=.5*jnp.dot(pg, result['inverse']@pg),
                    corrected_estimated_edm=.5*jnp.dot(pg, result['inverse']@pg)*(1+3*result['inverse_uncertainty']))

    return optimize
