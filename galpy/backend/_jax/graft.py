###############################################################################
#   galpy.backend._jax.graft: jax half of galpy.backend.autodiff.graft_derivative.
###############################################################################


def graft_derivative(x, value, deriv, higher):
    """``value`` with d/dx = ``deriv``; higher orders from ``higher(x)``."""
    import jax
    import jax.numpy as jnp

    @jax.custom_jvp
    def f(x):
        return jnp.asarray(value)

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (dx,) = primals, tangents
        if isinstance(x, jax.core.Tracer):
            # x itself differentiated (second order and up): the backend
            # computation supplies every derivative, the value stays `value`
            hb, dhb = jax.jvp(higher, (x,), (dx,))
            return jnp.asarray(value) + (hb - jax.lax.stop_gradient(hb)), dhb
        return jnp.asarray(value), jnp.asarray(deriv) * dx

    return f(x)
