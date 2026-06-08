###############################################################################
#   galpy.backend._torch.orbit_ode: torch (torchdiffeq) in-backend orbit
#   integration.
#
#   The torch-specific half of galpy.backend._reference.integrate_orbit. Integrates
#   the shared backend-agnostic EOM (_eom_rhs) with torchdiffeq. The jax
#   counterpart is galpy.backend._jax.orbit_ode.
###############################################################################


def integrate(pot, y0, ts, *, rtol, atol):
    """Integrate the EOM with torchdiffeq. y0/ys in rectangular EOM variables
    [x, vx, y, vy, z, vz].

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
        return torch.stack(_eom_rhs(y, pot, t, torch))

    return odeint(field, y0, ts, method="dopri5", rtol=rtol, atol=atol)
