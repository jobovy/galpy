###############################################################################
#   galpy.backend._reference.inbackend_ode
#
#   Differentiable orbit integration *inside* the backend: galpy's equations of
#   motion are evaluated through the backend-agnostic force layer
#   (_evaluateRforces / _evaluatephitorques / _evaluatezforces -- the
#   underscored, decorator-free evaluators, as the Python integrators use) and
#   integrated by the backend's own ODE solver (diffrax for jax, torchdiffeq for
#   torch), so gradients of the orbit w.r.t. initial conditions AND potential
#   parameters fall out of autodiff -- no hand-coded variational equations.
#
#   This is the reference, higher-order-differentiable orbit path for the
#   jax/torch backends, and the independent cross-check for the fast C
#   state-transition-matrix path (Track B/D of the backend plan).
#
#   Convention: phase-space vectors use galpy's Orbit ordering
#   ``[R, vR, vT, z, vz, phi]``. Internally the EOM is integrated in
#   *rectangular* variables ``[x, vx, y, vy, z, vz]`` -- matching galpy's C
#   integrator (integrateFullOrbit.c) rather than the cylindrical Python _EOM --
#   which avoids the 1/R centrifugal term and the coordinate singularity at the
#   axis, and is naturally well-behaved for autodiff. The public input/output are
#   transformed to/from ``Orbit`` order so they match ``Orbit``.
###############################################################################
from .. import get_namespace


def _eom_rhs(y, pot, t, xp):
    """Backend-agnostic rectangular EOM, state y = [x, vx, y, vy, z, vz].

    Mirrors galpy.orbit.integrateFullOrbit._rectForce: recover (R, phi) from the
    Cartesian position, evaluate the (decorator-free) force layer -- which
    dispatches on the array type of (R, z, phi) so this runs and differentiates
    under any backend -- and rotate the planar force back to Cartesian. Returns a
    length-6 tuple of time-derivatives.
    """
    # Imported lazily so importing this module never forces the potential import
    # graph at galpy import time.
    from ...potential.Potential import (
        _evaluatephitorques,
        _evaluateRforces,
        _evaluatezforces,
    )

    x, vx, yy, vy, z, vz = y[0], y[1], y[2], y[3], y[4], y[5]
    R = xp.sqrt(x**2 + yy**2)
    phi = xp.arctan2(yy, x)
    cosphi, sinphi = x / R, yy / R
    # v=[vR, vT, vz]; only used by velocity-dependent/dissipative forces.
    vR = vx * cosphi + vy * sinphi
    vT = -vx * sinphi + vy * cosphi
    v = [vR, vT, vz]
    Rforce = _evaluateRforces(pot, R, z, phi=phi, t=t, v=v)
    phitorque = _evaluatephitorques(pot, R, z, phi=phi, t=t, v=v)
    ax = cosphi * Rforce - sinphi / R * phitorque
    ay = sinphi * Rforce + cosphi / R * phitorque
    az = _evaluatezforces(pot, R, z, phi=phi, t=t, v=v)
    return vx, ax, vy, ay, vz, az


def _to_eom(xp, vxvv):
    """[R,vR,vT,z,vz,phi] (Orbit order) -> [x,vx,y,vy,z,vz] (rectangular EOM)."""
    R, vR, vT, z, vz, phi = (vxvv[i] for i in range(6))
    cosphi, sinphi = xp.cos(phi), xp.sin(phi)
    return xp.stack(
        [
            R * cosphi,
            vR * cosphi - vT * sinphi,
            R * sinphi,
            vR * sinphi + vT * cosphi,
            z,
            vz,
        ]
    )


def _from_eom(xp, ys):
    """[...,x,vx,y,vy,z,vz] -> [...,R,vR,vT,z,vz,phi] (Orbit order)."""
    x, vx, yy, vy, z, vz = (ys[..., i] for i in range(6))
    R = xp.sqrt(x**2 + yy**2)
    phi = xp.arctan2(yy, x)  # in [-pi, pi], matching Orbit's wrapped convention
    cosphi, sinphi = x / R, yy / R
    vR = vx * cosphi + vy * sinphi
    vT = -vx * sinphi + vy * cosphi
    return xp.stack([R, vR, vT, z, vz, phi], axis=-1)


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
    supplied as a backend array, w.r.t. the potential's parameters. ``phi`` is
    recovered from the Cartesian position, so it is wrapped to [-pi, pi] as in
    ``Orbit``.
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
    else:  # numpy path uses galpy's C/scipy integrators instead
        raise NotImplementedError(
            "in-backend ODE integration requires a jax or torch input array; "
            "for numpy use Orbit.integrate (C / scipy integrators)"
        )
    return _from_eom(xp, ys)
