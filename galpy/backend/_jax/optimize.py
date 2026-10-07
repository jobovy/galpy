###############################################################################
#   galpy.backend._jax.optimize: jax half of galpy.backend.optimize.brentq.
#
#   Vectorised sign-preserving bisection (shared bisect_root) followed by the
#   one-Newton-step reparameterisation that gives exact implicit-function-theorem
#   gradients w.r.t. the parameters f closes over. See galpy.backend.optimize for
#   the math. The torch counterpart is galpy.backend._torch.optimize.
###############################################################################


def brentq_backend(
    f, a, b, xp, *, xtol, maxiter, width=None, guess=None, newton_steps=10
):
    """jax bracketed root of ``f`` on ``[a, b]``, differentiable in f's params.

    ``f`` is the single-argument closure ``x -> func(x, *args)`` in jax.numpy.
    Bisection localises the root with a piecewise-constant (gradient-free) value;
    the Newton reparameterisation

        x* = x0 - f(x0) / f'(x0),   x0 = stop_gradient(bisection root),

    keeps the forward value at the bisection root (f(x0) ~ 0) while propagating
    the exact implicit-function gradient dx*/dtheta = -(df/dtheta)/(df/dx). f'(x0)
    is computed by ``jax.jvp`` (forward-mode directional derivative along dx),
    which is itself differentiable, so reverse-mode (jax.grad) through x* works.
    No internal jit -- the returned value composes with the user's jit/grad/vmap.
    """
    import jax
    import jax.numpy as jnp

    from ..optimize import newton_polish

    if guess is None:
        x0 = _bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter, width=width)
    else:
        x0 = _newton_root(f, a, b, guess, xp, newton_steps)
    x0 = jax.lax.stop_gradient(x0)
    # df/dx at x0 via a forward-mode directional derivative along the all-ones
    # tangent (exact df/dx for an elementwise f); the value fx0 comes for free.
    fx0, dfx0 = jax.jvp(f, (x0,), (jnp.ones_like(x0),))
    # One Newton step: exact root for a locally-linear f, and -- since x0 is a
    # constant w.r.t. theta and fx0 ~ 0 -- its theta-gradient is the implicit
    # one. Guard a (near-)singular slope so AD never sees a 0/0.
    return newton_polish(x0, fx0, dfx0, xp)


def _newton_root(f, a, b, guess, xp, steps):
    """Safeguarded Newton from ``guess`` (see optimize.newton_step_bracketed),
    df/dx by jax.jvp; rolled into lax.fori_loop when tracing."""
    import jax
    import jax.numpy as jnp

    from .._namespaces import under_jax_trace
    from ..optimize import newton_step_bracketed

    def fs(x):
        return jax.jvp(f, (x,), (jnp.ones_like(x),))

    x = jnp.clip(xp.asarray(guess) * 1.0, a, b)
    lo, hi = xp.asarray(a) + 0.0 * x, xp.asarray(b) + 0.0 * x
    if under_jax_trace(x, lo, hi):
        x, lo, hi = jax.lax.fori_loop(
            0, steps, lambda _, c: newton_step_bracketed(fs, *c, xp), (x, lo, hi)
        )
        return x
    for _ in range(steps):
        x, lo, hi = newton_step_bracketed(fs, x, lo, hi, xp)
    return x


def _bisect_root(f, a, b, xp, *, xtol, maxiter, width=None):
    """Bisection root, rolling the halving loop into ``lax.fori_loop`` only when
    tracing (user jit/grad/vmap).

    Eager (concrete bracket): defer to the shared Python-loop ``bisect_root`` --
    bit-identical and ~9x faster eagerly than ``fori_loop`` (which compiles a loop
    primitive per call). Traced: the unrolled Python loop would bake ``n`` copies
    of ``f`` into the user's jaxpr (the dominant compile cost for the heavier
    closures, e.g. Staeckel turning points); ``fori_loop`` traces ``f`` once,
    shrinking the jaxpr ~17x and the jitted compile ~4x for the same result.
    """
    import jax

    from .._namespaces import under_jax_trace
    from ..optimize import bisect_root, bisect_step, n_bisect_steps

    if not under_jax_trace(a, b):  # entered directly with a concrete bracket
        return bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter, width=width)
    lo = xp.asarray(a) * 1.0
    hi = xp.asarray(b) * 1.0
    slo = xp.sign(f(lo))
    # a tracer has no width -> min(maxiter, _MAXITER), unless the caller bounds it
    n = n_bisect_steps(a, b, xtol, maxiter, width=width)
    lo, hi = jax.lax.fori_loop(
        0, n, lambda _, c: bisect_step(c[0], c[1], slo, f, xp), (lo, hi)
    )
    return 0.5 * (lo + hi)
