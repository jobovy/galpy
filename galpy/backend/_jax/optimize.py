###############################################################################
#   galpy.backend._jax.optimize: jax half of galpy.backend.optimize.brentq.
#
#   Vectorised sign-preserving bisection (shared bisect_root) followed by the
#   one-Newton-step reparameterisation that gives exact implicit-function-theorem
#   gradients w.r.t. the parameters f closes over. See galpy.backend.optimize for
#   the math. The torch counterpart is galpy.backend._torch.optimize.
###############################################################################


def brentq_backend(f, a, b, xp, *, xtol, maxiter):
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

    x0 = jax.lax.stop_gradient(_bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter))
    # df/dx at x0 via a forward-mode directional derivative along the all-ones
    # tangent (exact df/dx for an elementwise f); the value fx0 comes for free.
    fx0, dfx0 = jax.jvp(f, (x0,), (jnp.ones_like(x0),))
    # One Newton step: exact root for a locally-linear f, and -- since x0 is a
    # constant w.r.t. theta and fx0 ~ 0 -- its theta-gradient is the implicit
    # one. Guard a (near-)singular slope so AD never sees a 0/0.
    return x0 - fx0 / _nonzero(dfx0, xp)


def _bisect_root(f, a, b, xp, *, xtol, maxiter):
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
        return bisect_root(f, a, b, xp, xtol=xtol, maxiter=maxiter)
    lo = xp.asarray(a) * 1.0
    hi = xp.asarray(b) * 1.0
    slo = xp.sign(f(lo))
    n = n_bisect_steps(a, b, xtol, maxiter)  # tracer -> min(maxiter, _MAXITER)
    lo, hi = jax.lax.fori_loop(
        0, n, lambda _, c: bisect_step(c[0], c[1], slo, f, xp), (lo, hi)
    )
    return 0.5 * (lo + hi)


def _nonzero(d, xp):
    """Replace (near-)zero slopes by a tiny same-sign value so 1/d stays finite.

    A vanishing df/dx is a genuinely ill-posed root (tangential crossing); rather
    than emit inf/NaN that would poison reverse-mode AD, floor |d| at a tiny
    epsilon keeping its sign. Dead-branch guarded: both ``xp.where`` sides are
    evaluated eagerly, and the flooring keeps every branch finite.
    """
    ones = xp.ones_like(d)
    sign = xp.where(d >= 0, ones, -ones)
    return sign * xp.maximum(xp.abs(d), 1e-300 * ones)
