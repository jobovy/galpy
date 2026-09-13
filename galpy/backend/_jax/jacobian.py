###############################################################################
#   galpy.backend._jax.jacobian: jax half of galpy.backend.jacobian.
#
#   Reverse-mode Jacobian (jax.jacrev): galpy's C-STM orbit integrator is a
#   custom_vjp, so forward-mode jacfwd cannot differentiate through it. jacrev is
#   naturally composable for higher-order AD (grad through the Jacobian works).
###############################################################################


def jacobian_backend(f, x):
    """``jax.jacrev(f)(x)`` -- dense Jacobian, composable for higher-order AD."""
    import jax

    return jax.jacrev(f)(x)
