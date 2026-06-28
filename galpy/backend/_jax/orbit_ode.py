###############################################################################
#   galpy.backend._jax.orbit_ode: jax (diffrax) in-backend orbit integration.
#
#   The jax-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with diffrax. The torch counterpart
#   is galpy.backend._torch.orbit_ode.
###############################################################################


def integrate(pot, y0, ts, *, dim, rtol, atol, max_steps):
    """Integrate the EOM with diffrax (Dopri8, adaptive). y0/ys in rectangular
    EOM variables [x, vx, y, vy, z, vz], shape (dim,) for one orbit or (N, dim) for
    a batch.

    ``ts`` shape (nt,): one shared output grid -> all N orbits in ONE solve (one
    shared adaptive controller). ``ts`` shape (N, nt): a PER-ORBIT grid -> jax.vmap
    over (y0, ts) so each orbit gets its own saveat/span and its own (independent)
    controller. Either way returns ys (nt, N, dim) for a batch ((nt, dim) single).
    Reverse-mode differentiable (diffrax uses a custom_vjp -> forward-mode jacfwd is
    unavailable; use jacrev)."""
    import diffrax
    import jax
    import jax.numpy as jnp

    from .._reference.inbackend_ode import _eom_rhs

    # stack on the trailing (component) axis so a batch (N, dim) state maps to an
    # (N, dim) derivative; for a single (dim,) state axis=-1 == axis=0.
    term = diffrax.ODETerm(
        lambda t, y, args: jnp.stack(_eom_rhs(y, pot, t, jnp, dim), axis=-1)
    )

    def _solve(y0i, tsi):
        return diffrax.diffeqsolve(
            term,
            diffrax.Dopri8(),
            t0=tsi[0],
            t1=tsi[-1],
            dt0=None,
            y0=y0i,
            saveat=diffrax.SaveAt(ts=tsi),
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=max_steps,
        ).ys

    if ts.ndim > 1:  # per-orbit grids (N, nt): map each (y0, ts) to its own solve
        return jax.vmap(_solve, in_axes=(0, 0), out_axes=1)(y0, ts)
    return _solve(y0, ts)  # shared (nt,) grid: native batch (or a single orbit)
