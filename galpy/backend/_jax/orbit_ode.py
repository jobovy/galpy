###############################################################################
#   galpy.backend._jax.orbit_ode: jax (diffrax) in-backend orbit integration.
#
#   The jax-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with diffrax. The torch counterpart
#   is galpy.backend._torch.orbit_ode.
###############################################################################


def integrate(pot, y0, ts, *, rtol, atol, max_steps):
    """Integrate the EOM with diffrax (Dopri8, adaptive). y0/ys in EOM variables
    [R, vR, phi, Omega, z, vz]. Reverse-mode differentiable (diffrax uses a
    custom_vjp -> forward-mode jacfwd is unavailable; use jacrev)."""
    import diffrax
    import jax.numpy as jnp

    from .._reference.inbackend_ode import _eom_rhs

    def field(t, y, args):
        return jnp.stack(_eom_rhs(y, pot, t))

    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(field),
        diffrax.Dopri8(),
        t0=ts[0],
        t1=ts[-1],
        dt0=None,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
        max_steps=max_steps,
    )
    return sol.ys
