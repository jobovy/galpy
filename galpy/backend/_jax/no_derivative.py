###############################################################################
#   galpy.backend._jax.no_derivative: a host callback with a NaN derivative.
###############################################################################
import jax


def no_derivative(fn, a):
    """``fn(a)`` for a host callback ``fn``, whose derivative is NaN: it has none,
    and under a trace a raise would also hit ``lax.cond`` arms that never run."""
    g = jax.custom_jvp(fn)
    g.defjvp(lambda p, t: (fn(*p), t[0] * jax.numpy.nan))
    return g(a)
