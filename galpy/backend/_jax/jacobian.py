###############################################################################
#   galpy.backend._jax.jacobian: jax half of galpy.backend.jacobian.
#
#   Reverse-mode Jacobian (jax.jacrev) by default: galpy's C-STM orbit integrator
#   is a custom_vjp, so forward-mode jacfwd cannot differentiate through it.
#   forward=True takes jax.jacfwd (e.g. over diffrax ForwardMode solves). Both
#   compose for higher-order AD (grad through the Jacobian works).
###############################################################################


def jacobian_backend(f, x, forward=False):
    """``jax.jacrev(f)(x)`` (``jax.jacfwd`` if ``forward``) -- dense Jacobian,
    composable for higher-order AD."""
    import jax

    return (jax.jacfwd if forward else jax.jacrev)(f)(x)
