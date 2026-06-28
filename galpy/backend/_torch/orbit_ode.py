###############################################################################
#   galpy.backend._torch.orbit_ode: torch (torchdiffeq) in-backend orbit
#   integration.
#
#   The torch-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with torchdiffeq. The jax
#   counterpart is galpy.backend._jax.orbit_ode.
###############################################################################


def integrate(pot, y0, ts, *, dim, rtol, atol):
    """Integrate the EOM with torchdiffeq. y0/ys in rectangular EOM variables
    [x, vx, y, vy, z, vz], shape (dim,) for one orbit or (N, dim) for a batch.

    ``ts`` shape (nt,): one shared output grid -> all N orbits in ONE solve.
    ``ts`` shape (N, nt): a PER-ORBIT grid -> integrate each orbit on its own grid
    (torchdiffeq has no vmap, so a per-orbit loop) and stack on the orbit axis.
    Either way returns ys (nt, N, dim) for a batch ((nt, dim) single).

    Uses ``dopri5``, NOT ``dopri8``: torchdiffeq's ``dopri8`` *backward* pass is
    noticeably less accurate (~1e-5 relative gradient error vs ~1e-8 for
    dopri5/rk4), because its adaptive step controller is detached on the backward.
    ``dopri5`` gives accurate gradients (matching jax/diffrax Dopri8) while still
    matching the C integrator forward to ~1e-7. (diffrax's Dopri8 backward is fine;
    this is torchdiffeq-specific.)
    """
    import torch
    from torchdiffeq import odeint

    from .._reference.inbackend_ode import _eom_rhs

    def field(t, y):
        # stack on the trailing (component) axis so a batch (N, dim) state maps to
        # an (N, dim) derivative; for a single (dim,) state axis=-1 == axis=0.
        return torch.stack(_eom_rhs(y, pot, t, torch, dim), axis=-1)

    if ts.ndim > 1:  # per-orbit grids (N, nt): one solve per orbit, stack -> (nt,N,dim)
        return torch.stack(
            [
                odeint(field, y0[i], ts[i], method="dopri5", rtol=rtol, atol=atol)
                for i in range(y0.shape[0])
            ],
            dim=1,
        )
    return odeint(field, y0, ts, method="dopri5", rtol=rtol, atol=atol)
