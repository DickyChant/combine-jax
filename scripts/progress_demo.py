"""Show the fit progress display on small toy fits (no workspace needed).

    python scripts/progress_demo.py [--slow SECONDS] [--lanes] [--mode tty|plain]

Runs four toy fits with progress on: an uncapped smooth fit, a capped fit
that hits its cap, a fit with an absolute-value kink that needs the
directional EDM guard, and a batch of 6 fits under vmap. ``--slow`` sleeps
in the host sink after each record so the toy fits are slow enough to watch;
it does not change the fits.
"""
import argparse
import time

import jax
import jax.numpy as jnp

jax.config.update('jax_enable_x64', True)

from combine_jax.tensor import progress
from combine_jax.tensor.fitting import make_fit_function
from combine_jax.progress_display import ProgressDisplay


def rosenbrock(theta, _):
    return 100.*jnp.sum((theta[1:]-theta[:-1]**2)**2)+jnp.sum((1.-theta[:-1])**2)


def kinked(theta, _):
    return rosenbrock(theta, _)+.5*jnp.abs(theta[0]-1.)


def coordinates(n, start):
    reference = jnp.full(n, start)
    return dict(reference=reference, free_indices=jnp.arange(n), scale=jnp.ones(n),
                lower=jnp.full(n, -5.)-reference, upper=jnp.full(n, 5.)-reference,
                physical_lower=jnp.full(n, -5.), physical_upper=jnp.full(n, 5.))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slow', type=float, default=.03)
    parser.add_argument('--lanes', action='store_true')
    parser.add_argument('--mode', default='auto')
    parser.add_argument('--every', type=int, default=5)
    args = parser.parse_args()
    n = 30
    options = dict(covariance=False, stopping_criterion='edm', edm_tolerance=1e-4,
                   edm_validation='estimated', progress_every=args.every)
    display = ProgressDisplay(mode=args.mode, lanes=args.lanes, total=10,
                              redraw_interval=0., plain_interval=1.)

    def show(record):
        display.update(record)
        time.sleep(args.slow)

    with progress.sink(show):
        fit = jax.jit(make_fit_function(rosenbrock, max_iterations=10**8, **options))
        with progress.task('uncapped'):
            jax.block_until_ready(fit(coordinates(n, -1.2), {}))
        fit = jax.jit(make_fit_function(rosenbrock, max_iterations=60, **options))
        with progress.task('capped 60'):
            jax.block_until_ready(fit(coordinates(n, -1.2), {}))
        fit = jax.jit(make_fit_function(kinked, max_iterations=10**8,
                                        **dict(options, edm_validation='directional')))
        with progress.task('kink+guard'):
            jax.block_until_ready(fit(coordinates(n, -1.2), {}))
        fit = jax.jit(make_fit_function(rosenbrock, max_iterations=10**8, **options))
        batch = jax.jit(jax.vmap(fit, in_axes=(0, None)))
        starts = jnp.array([-1.2, -.5, 0., .5, 1.5, 2.])
        stacked = jax.vmap(lambda s: coordinates(n, s))(starts)
        with progress.task('batch'):
            jax.block_until_ready(batch(stacked, {}))
        # The batch counts as 6 of the 9 fits; one sequential fit is left.
        fit = jax.jit(make_fit_function(rosenbrock, max_iterations=10**8, **options))
        with progress.task('last'):
            jax.block_until_ready(fit(coordinates(n, 0.), {}))
    display.close()


if __name__ == '__main__':
    main()
