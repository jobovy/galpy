###############################################################################
#   galpy.backend._reference.inbackend_ode
#
#   Differentiable orbit integration *inside* the backend: galpy's equations of
#   motion are evaluated through the backend-agnostic force layer
#   (evaluateRforces / evaluatephitorques / evaluatezforces) and integrated by
#   the backend's own ODE solver (diffrax for jax, torchdiffeq for torch), so
#   gradients of the orbit w.r.t. initial conditions AND potential parameters
#   fall out of autodiff -- no hand-coded variational equations.
#
#   This is the reference, higher-order-differentiable orbit path for the
#   jax/torch backends, and the independent cross-check for the fast C
#   state-transition-matrix path (Track B/D of the backend plan).
#
#   Convention: phase-space vectors use galpy's Orbit ordering
#   ``[R, vR, vT, z, vz, phi]``. Internally the EOM is integrated in
#   ``[R, vR, phi, Omega=dphi/dt, z, vz]`` (galpy's ``_EOM`` variables) and
#   mapped back, so the public input/output match ``Orbit``.
###############################################################################
from .. import get_namespace


def _eom_rhs(y, pot, t):
    """Backend-agnostic cylindrical EOM, state y = [R, vR, phi, Omega, z, vz].

    Mirrors galpy.orbit.integrateFullOrbit._EOM; the force evaluators dispatch on
    the array type of (R, z, phi), so this runs and differentiates under any
    backend. Returns a length-6 tuple of time-derivatives.
    """
    # Imported lazily so importing this module never forces the potential import
    # graph at galpy import time.
    from ...potential import (
        evaluatephitorques,
        evaluateRforces,
        evaluatezforces,
    )

    R, vR, phi, Om, z, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    Lz2 = (R**2 * Om) ** 2
    # v=[vR, vT, vz] matches galpy._EOM exactly (ignored for non-dissipative
    # potentials; required by velocity-dependent/dissipative forces).
    v = [vR, R * Om, vz]
    aR = Lz2 / R**3 + evaluateRforces(pot, R, z, phi=phi, t=t, v=v)
    dOm = (evaluatephitorques(pot, R, z, phi=phi, t=t, v=v) - 2.0 * R * vR * Om) / R**2
    az = evaluatezforces(pot, R, z, phi=phi, t=t, v=v)
    return vR, aR, Om, dOm, vz, az


def _to_eom(xp, vxvv):
    """[R,vR,vT,z,vz,phi] (Orbit order) -> [R,vR,phi,Omega,z,vz] (EOM order)."""
    R, vR, vT, z, vz, phi = (vxvv[i] for i in range(6))
    return xp.stack([R, vR, phi, vT / R, z, vz])


def _from_eom(xp, ys):
    """[...,R,vR,phi,Omega,z,vz] -> [...,R,vR,vT,z,vz,phi] (Orbit order)."""
    R, vR, phi, Om, z, vz = (ys[..., i] for i in range(6))
    return xp.stack([R, vR, R * Om, z, vz, phi], axis=-1)


def integrate_orbit(pot, vxvv, ts, *, rtol=1e-10, atol=1e-10, max_steps=100000):
    """Differentiably integrate a 3D orbit with the backend's ODE solver.

    Parameters
    ----------
    pot : Potential (or list) -- must be backend-migrated for the chosen backend.
    vxvv : backend array, shape (6,), ``[R, vR, vT, z, vz, phi]`` (Orbit order).
        Its namespace selects the integrator: jax -> diffrax, torch -> torchdiffeq.
    ts : backend array of output times (monotonic; ts[0] is the initial time).
    rtol, atol : solver tolerances.
    max_steps : diffrax step cap (jax only).

    Returns
    -------
    backend array, shape ``(len(ts), 6)``, the orbit in ``[R, vR, vT, z, vz, phi]``.

    Notes
    -----
    Differentiable w.r.t. ``vxvv`` (initial conditions) and, when the parameter is
    supplied as a backend array, w.r.t. the potential's parameters. ``phi`` is the
    unwrapped solver value (galpy's ``Orbit`` wraps it to [-pi, pi]).
    """
    xp = get_namespace(vxvv)
    name = xp.__name__
    y0 = _to_eom(xp, vxvv)
    # backend-specific integrators live in galpy.backend._jax / ._torch
    if "jax" in name:
        from .._jax.orbit_ode import integrate as _integrate_jax

        ys = _integrate_jax(pot, y0, ts, rtol=rtol, atol=atol, max_steps=max_steps)
    elif "torch" in name:
        from .._torch.orbit_ode import integrate as _integrate_torch

        ys = _integrate_torch(pot, y0, ts, rtol=rtol, atol=atol)
    else:  # pragma: no cover - numpy path uses galpy's C/scipy integrators instead
        raise NotImplementedError(
            "in-backend ODE integration requires a jax or torch input array; "
            "for numpy use Orbit.integrate (C / scipy integrators)"
        )
    return _from_eom(xp, ys)
