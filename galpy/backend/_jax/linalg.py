###############################################################################
#   galpy.backend._jax.linalg: jax half of galpy.backend.linalg.real_eig.
###############################################################################


def safe_eigh(a):
    """``eigh`` whose eigenvector derivative is zero inside degenerate clusters."""
    import jax
    import jax.numpy as jnp

    from ..linalg import degenerate_safe_gap_inverse

    @jax.custom_jvp
    def _eigh(a):
        return jnp.linalg.eigh(a)

    @_eigh.defjvp
    def _eigh_jvp(primals, tangents):
        # written in differentiable ops of _eigh's own output -> every order
        (a,), (da,) = primals, tangents
        lam, v = _eigh(a)
        vt = jnp.swapaxes(v, -1, -2)
        x = vt @ (0.5 * (da + jnp.swapaxes(da, -1, -2))) @ v
        return (lam, v), (
            jnp.diagonal(x, axis1=-2, axis2=-1),
            v @ (degenerate_safe_gap_inverse(lam) * x),
        )

    # a concrete input stays concrete: under an outer jvp the custom rule would
    # lift even constants to (zero-tangent) tracers
    if not isinstance(a, jax.core.Tracer):
        return jnp.linalg.eigh(a)
    return _eigh(a)
