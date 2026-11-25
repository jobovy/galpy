###############################################################################
#   galpy.backend._jax.optimize: jax half of galpy.backend.optimize.brentq.
#
#   Vectorised sign-preserving bisection (shared bisect_root) followed by the
#   one-Newton-step reparameterisation that gives exact implicit-function-theorem
#   gradients w.r.t. the parameters f closes over. See galpy.backend.optimize for
#   the math. The torch counterpart is galpy.backend._torch.optimize.
###############################################################################


def brentq_backend(
    f,
    a,
    b,
    xp,
    *,
    xtol,
    maxiter,
    width=None,
    guess=None,
    newton_steps=10,
    sections=None,
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

    from ..optimize import newton_polish, section_root, sectionable

    if guess is None and sectionable(a, b, sections):
        x0, flo = section_root(
            f, a, b, xp, xtol=xtol, maxiter=maxiter, width=width, sections=sections
        )
        if not isinstance(flo, jax.core.Tracer):
            return x0  # nothing to differentiate; the secant value is polished
    elif guess is None:
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


def iterate_to_fixed_point(step, x0, n):
    """jax half of optimize.iterate_to_fixed_point: a lax.while_loop that stops
    once a step leaves x unchanged (or after n steps)."""
    import jax
    import jax.numpy as jnp

    def body(c):
        i, x, _ = c
        x1 = step(x)
        return i + 1, x1, jnp.all(x1 == x)

    _, x, _ = jax.lax.while_loop(
        lambda c: (c[0] < n) & ~c[2], body, (0, x0, jnp.asarray(False))
    )
    return x


def fixed_point_backend(func, x0, args, xp, *, xtol, maxiter):
    """jax half of optimize.fixed_point: del2 on stop_gradient'd args (a
    lax.while_loop when tracing), then the implicit-function step."""
    import jax

    from .._namespaces import is_backend_array, under_jax_trace
    from ..optimize import del2_iterate, del2_step

    sg = jax.lax.stop_gradient
    dargs = tuple(sg(a) if is_backend_array(a) else a for a in args)
    f = lambda x: func(x, *dargs)
    p = sg(xp.asarray(x0) * 1.0)
    if under_jax_trace(p, *dargs):

        def body(c):
            i, p0, _ = c
            p1, relerr = del2_step(f, p0, xp)
            return i + 1, p1, xp.all(xp.abs(relerr) < xtol)

        _, p, _ = jax.lax.while_loop(
            lambda c: (c[0] < maxiter) & ~c[2], body, (0, p, xp.asarray(False))
        )
    else:
        p = del2_iterate(f, p, xp, xtol, maxiter)
    return implicit_fixed_point_step(func, p, args, xp)


def implicit_fixed_point_step(func, p, args, xp, vector=False):
    """jax half of optimize.fixed_point_implicit_step: the value at p, with
    dx*/dtheta = (dg/dtheta) / (1 - dg/dx) through g = func(p, *args) (with
    vector, (I - dg/dx)^-1 dg/dtheta for each column's k x k dg/dx)"""
    import jax

    from .._namespaces import is_backend_array, under_jax_trace
    from ..optimize import nonzero_slope

    sg = jax.lax.stop_gradient
    dargs = tuple(sg(a) if is_backend_array(a) else a for a in args)
    p = sg(p)
    g = func(p, *args)
    if not under_jax_trace(g):  # no derivative to carry (as torch's)
        return p
    f = lambda x: func(x, *dargs)  # noqa: E731
    if not vector:
        _, slope = jax.jvp(f, (p,), (xp.ones_like(p),))
        return p + (g - sg(g)) / nonzero_slope(1.0 - slope, xp)
    eye = jax.numpy.eye(p.shape[0], dtype=p.dtype)
    J = columnwise_jacobian(f, p)[1]  # (n, k, k)
    rhs = jax.numpy.moveaxis(g - sg(g), 0, 1)[..., None]  # (n, k, 1)
    step = jax.numpy.linalg.solve(eye - J, rhs)[..., 0]
    return p + jax.numpy.moveaxis(step, 1, 0)


def columnwise_jacobian(f, p, has_aux=False):
    """jax half of optimize.columnwise_jacobian: the jvps of the k unknowns
    (the tangent of one unknown on every column), batched into one pass"""
    import jax

    tangents = jax.numpy.eye(p.shape[0], dtype=p.dtype)[:, :, None] + 0.0 * p
    out = jax.vmap(
        lambda t: jax.jvp(f, (p,), (t,), has_aux=has_aux),
        out_axes=(None, -1, None) if has_aux else (None, -1),
    )(tangents)
    J = jax.numpy.moveaxis(out[1], 1, 0)  # (n, m, k)
    return (out[0], J, out[2]) if has_aux else (out[0], J)


def columnwise_jvp(f, p, t, has_aux=False, reverse=False):
    """jax half of optimize.columnwise_jvp: jax.jvp, or with reverse one
    backward pass per output row (the columns are independent), contracted
    with t; reverse=None: jax.jvp unless f contains a custom_vjp (such as the
    C-STM), which only differentiates backward and to first order: then,
    for traced p or t, the rate comes from backward passes at the stopped p
    and t, and differentiating it raises"""
    import jax

    from .._namespaces import under_jax_trace

    first_order = False
    if reverse is None and under_jax_trace(p, t):
        # (traced, forward mode stages a custom_vjp as a custom_lin, which
        # only fails when compiled)
        first_order = _only_backward(f, p, t, has_aux)
        if not first_order:
            return jax.jvp(f, (p,), (t,), has_aux=has_aux)
    elif reverse is None:  # forward, unless f only differentiates backward
        try:
            return jax.jvp(f, (p,), (t,), has_aux=has_aux)
        except TypeError as e:
            if "custom_vjp" not in str(e):
                raise
    elif not reverse:
        return jax.jvp(f, (p,), (t,), has_aux=has_aux)
    sg = jax.lax.stop_gradient if first_order else (lambda x: x)
    out = jax.vjp(f, sg(p), has_aux=has_aux)
    fx, back = out[0], out[1]
    eye = jax.numpy.eye(fx.shape[0], dtype=fx.dtype)[:, :, None]
    rate = jax.numpy.stack(
        [
            jax.numpy.sum(back(eye[i] + 0.0 * fx)[0] * sg(t), axis=0)
            for i in range(fx.shape[0])
        ]
    )
    if first_order:
        out = f(p) if has_aux else (f(p),)
        fx, rate = out[0], _first_order_only(rate, (p, t))
    return (fx, rate, out[-1]) if has_aux else (fx, rate)


def _only_backward(f, p, t, has_aux):
    """Whether f only differentiates backward: whether forward mode stages
    a custom_lin"""
    import jax

    def staged(jaxpr):
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "custom_lin":
                return True
            for v in eqn.params.values():
                for sub in v if isinstance(v, (tuple, list)) else (v,):
                    sub = getattr(sub, "jaxpr", sub)
                    if hasattr(sub, "eqns") and staged(sub):
                        return True
        return False

    return staged(
        jax.make_jaxpr(lambda p, t: jax.jvp(f, (p,), (t,), has_aux=has_aux))(p, t).jaxpr
    )


def _first_order_only(x, deps):
    """x, whose derivative (with respect to deps) raises"""
    import jax

    @jax.custom_vjp
    def guard(x, deps):
        return x

    def bwd(res, ct):
        raise NotImplementedError(
            "second derivatives through a function that differentiates only "
            "to first order (a custom_vjp, such as galpy's C code) are not "
            "available"
        )

    guard.defvjp(lambda x, deps: (x, None), bwd)
    return guard(x, deps)
